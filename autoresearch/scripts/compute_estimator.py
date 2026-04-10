"""Compute time estimation for H100 GPU training runs.

Wrapper around h100_time_guess logic. Provides functions to estimate
screen/promote wallclock times from training logs using TFLOPS-ratio
extrapolation.

If the actual h100_time_guess.py doesn't exist in scripts/, this module
implements a simplified version: parse "train_time:...ms" and "step:N/M"
from logs, use a TFLOPS lookup table for common GPUs, and extrapolate.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

__all__ = [
    "estimate_screen_time",
    "estimate_promote_time",
    "check_within_budget",
]

logger = logging.getLogger(__name__)

# Peak FP16/BF16 TFLOPS for common GPU SKUs (theoretical peak)
_TFLOPS_TABLE: dict[str, float] = {
    "H100 SXM": 1979.0,
    "H100 PCIe": 1513.0,
    "H100": 1979.0,
    "A100 SXM": 312.0,
    "A100 PCIe": 312.0,
    "A100": 312.0,
    "A6000": 155.0,
    "L40S": 733.0,
    "L40": 362.0,
    "RTX 4090": 330.0,
    "RTX 3090": 142.0,
}

# Default target: 8xH100 SXM total TFLOPS
_H100_8X_TFLOPS = 8 * 1979.0

# Regex patterns for log parsing
_STEP_RE = re.compile(
    r"step[:\s]*(\d+)\s*/\s*(\d+)",
    re.IGNORECASE,
)
_STEP_SIMPLE_RE = re.compile(
    r"step[:\s]*(\d+)",
    re.IGNORECASE,
)
_TRAIN_TIME_RE = re.compile(
    r"train_time[:\s]*([\d.]+)\s*ms",
    re.IGNORECASE,
)
_MS_STEP_RE = re.compile(
    r"ms_step[:\s]*([\d.eE+\-]+)",
    re.IGNORECASE,
)
_WALLCLOCK_RE = re.compile(
    r"total.*?wallclock.*?([\d.]+)\s*s",
    re.IGNORECASE,
)
_GPU_RE = re.compile(
    r"(H100|A100|A6000|L40S|L40|RTX 4090|RTX 3090)",
    re.IGNORECASE,
)
_NPROC_RE = re.compile(
    r"nproc_per_node[=:\s]*(\d+)",
    re.IGNORECASE,
)


def _parse_log_timing(log_path: Path) -> Optional[dict[str, float]]:
    """Extract timing information from a training log.

    Returns a dict with keys: ms_per_step, total_steps, completed_steps,
    wallclock_s, gpu_count. Any key may be absent if not found.
    """
    if not log_path.exists():
        return None

    info: dict[str, float] = {}
    max_step = 0
    total_steps = 0
    last_ms_step = 0.0
    ms_step_sum = 0.0
    ms_step_count = 0
    gpu_count = 0.0

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    for line in text.splitlines():
        # step N/M pattern
        m = _STEP_RE.search(line)
        if m:
            step = int(m.group(1))
            total = int(m.group(2))
            if step > max_step:
                max_step = step
            if total > total_steps:
                total_steps = total
        else:
            m = _STEP_SIMPLE_RE.search(line)
            if m:
                step = int(m.group(1))
                if step > max_step:
                    max_step = step

        # ms_step
        m = _MS_STEP_RE.search(line)
        if m:
            try:
                val = float(m.group(1))
                last_ms_step = val
                ms_step_sum += val
                ms_step_count += 1
            except ValueError:
                pass

        # train_time in ms
        m = _TRAIN_TIME_RE.search(line)
        if m:
            try:
                info["train_time_ms"] = float(m.group(1))
            except ValueError:
                pass

        # wallclock
        m = _WALLCLOCK_RE.search(line)
        if m:
            try:
                info["wallclock_s"] = float(m.group(1))
            except ValueError:
                pass

        # nproc (GPU count)
        m = _NPROC_RE.search(line)
        if m:
            gpu_count = float(m.group(1))

    if max_step > 0:
        info["completed_steps"] = float(max_step)
    if total_steps > 0:
        info["total_steps"] = float(total_steps)
    if ms_step_count > 0:
        info["ms_per_step"] = ms_step_sum / ms_step_count
    if last_ms_step > 0:
        info["last_ms_step"] = last_ms_step
    if gpu_count > 0:
        info["gpu_count"] = gpu_count

    return info if info else None


def _detect_gpu_type(log_path: Path) -> Optional[str]:
    """Try to detect the GPU type from log contents."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = _GPU_RE.search(text)
    if m:
        return m.group(1).upper()
    return None


def _extrapolate_time(
    ms_per_step: float,
    completed_steps: float,
    target_steps: float,
    source_gpu_count: float,
    target_gpu_count: float,
    source_tflops_per_gpu: float,
    target_tflops_per_gpu: float,
) -> float:
    """Extrapolate training time to a different GPU config.

    Uses the TFLOPS-ratio method: time scales inversely with total TFLOPS.
    """
    source_total_tflops = source_gpu_count * source_tflops_per_gpu
    target_total_tflops = target_gpu_count * target_tflops_per_gpu

    if source_total_tflops <= 0 or target_total_tflops <= 0:
        return 0.0

    tflops_ratio = source_total_tflops / target_total_tflops
    time_per_step_target = ms_per_step * tflops_ratio
    total_time_ms = time_per_step_target * target_steps
    return total_time_ms / 1000.0


def estimate_screen_time(parent_log: Path) -> Optional[float]:
    """Estimate screen-stage wallclock seconds on 2xH100 from a parent log.

    Args:
        parent_log: Path to a training log from a previous run.

    Returns:
        Estimated seconds for a 2-GPU screen run, or None if log is
        unparseable.
    """
    info = _parse_log_timing(parent_log)
    if info is None:
        return None

    ms_per_step = info.get("ms_per_step")
    if ms_per_step is None or ms_per_step <= 0:
        return None

    completed = info.get("completed_steps", 0)
    total = info.get("total_steps", completed)
    if total <= 0:
        # Fallback: use wallclock if available
        wallclock = info.get("wallclock_s")
        if wallclock and wallclock > 0:
            return wallclock
        return None

    source_gpus = info.get("gpu_count", 2.0)
    gpu_type = _detect_gpu_type(parent_log)
    source_tflops = _TFLOPS_TABLE.get(gpu_type or "H100", 1979.0)

    # Screen: 2 GPUs, H100
    return _extrapolate_time(
        ms_per_step=ms_per_step,
        completed_steps=completed,
        target_steps=total,
        source_gpu_count=source_gpus,
        target_gpu_count=2.0,
        source_tflops_per_gpu=source_tflops,
        target_tflops_per_gpu=1979.0,
    )


def estimate_promote_time(screen_log: Path) -> Optional[float]:
    """Estimate promote-stage wallclock seconds on 8xH100 from a screen log.

    Args:
        screen_log: Path to a screen-stage training log.

    Returns:
        Estimated seconds for an 8-GPU promote run, or None if unparseable.
    """
    info = _parse_log_timing(screen_log)
    if info is None:
        return None

    ms_per_step = info.get("ms_per_step")
    if ms_per_step is None or ms_per_step <= 0:
        return None

    source_gpus = info.get("gpu_count", 2.0)
    gpu_type = _detect_gpu_type(screen_log)
    source_tflops = _TFLOPS_TABLE.get(gpu_type or "H100", 1979.0)

    # Promote runs use 8 GPUs and typically more steps.
    # Grad accumulation: 8 // world_size. So 8 GPUs = 1x accum, 2 GPUs = 4x accum.
    # More GPUs = more steps in same wallclock (linear scaling minus overhead).
    # We estimate: same total compute, distributed across 8 GPUs.
    completed = info.get("completed_steps", 0)
    total = info.get("total_steps", completed)
    if total <= 0:
        return None

    # Scale step count: promote typically runs ~3.3x more steps than screen
    # (600s / 180s budget ratio), but with 4x more GPUs the per-step time
    # drops by ~4x. Net: roughly same wallclock per step-count-ratio.
    promote_steps = total * (600.0 / 180.0)

    return _extrapolate_time(
        ms_per_step=ms_per_step,
        completed_steps=completed,
        target_steps=promote_steps,
        source_gpu_count=source_gpus,
        target_gpu_count=8.0,
        source_tflops_per_gpu=source_tflops,
        target_tflops_per_gpu=1979.0,
    )


def check_within_budget(
    log: Path,
    budget_s: float,
) -> Tuple[bool, float]:
    """Check whether a run's extrapolated time fits within a budget.

    Args:
        log: Path to a training log.
        budget_s: Budget in seconds (e.g. 180 for screen, 600 for promote).

    Returns:
        Tuple of (within_budget, estimated_seconds). If the log is
        unparseable, returns (True, 0.0) — optimistic default.
    """
    info = _parse_log_timing(log)
    if info is None:
        return (True, 0.0)

    # If we have actual wallclock, use that directly
    wallclock = info.get("wallclock_s")
    if wallclock and wallclock > 0:
        return (wallclock <= budget_s, wallclock)

    ms_per_step = info.get("ms_per_step", 0.0)
    total = info.get("total_steps", info.get("completed_steps", 0.0))
    if ms_per_step <= 0 or total <= 0:
        return (True, 0.0)

    estimated_s = (ms_per_step * total) / 1000.0
    return (estimated_s <= budget_s, estimated_s)
