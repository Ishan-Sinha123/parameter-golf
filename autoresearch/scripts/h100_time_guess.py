"""Compute time estimator for GPU training runs.

Estimates wallclock time on target hardware by parsing training logs and
scaling via peak TFLOPS ratios. Integrated from the JianYan11/parameter-golf
tooling plane (§2.1) for use in QUEUE pre-flight (§3.1) and PROMOTE budget
verification (§6.2).

CLI::

    python3 -m autoresearch.scripts.h100_time_guess check <log_path>
    python3 -m autoresearch.scripts.h100_time_guess estimate <log_path> [--target-gpus N] [--target-gpu NAME]
"""

from __future__ import annotations

__all__ = [
    "GPU_TFLOPS",
    "TrainLogStats",
    "BudgetCheck",
    "parse_train_log",
    "estimate_h100_time",
    "check_within_budget",
]

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Peak FP16 TFLOPS for common GPU SKUs.
GPU_TFLOPS: Dict[str, float] = {
    "H100 SXM": 989.0,
    "H100 PCIe": 756.0,
    "H100 NVL": 835.0,
    "A100 80GB": 312.0,
    "A100 40GB": 312.0,
    "A100 PCIe": 312.0,
    "A10": 125.0,
    "L40S": 362.0,
    "L40": 181.0,
    "L4": 121.0,
    "4090": 330.0,
    "3090": 142.0,
}

# Patterns to extract step timing from training logs.
# Matches lines like: "step:42/1000 train_time:115.23ms" or "step 42 | 115.2 ms/step"
_STEP_TIME_PATTERNS: List[re.Pattern[str]] = [
    # train_gpt.py format: "step:N/M train_time:Xms"
    re.compile(
        r"step[:\s]+(?P<step>\d+)[/\s]+(?P<total>\d+)\s+.*?"
        r"train_time[:\s]+(?P<ms>[\d.]+)\s*ms"
    ),
    # Common format: "step N | X ms/step"
    re.compile(
        r"step\s+(?P<step>\d+)\s*\|\s*(?P<ms>[\d.]+)\s*ms/step"
    ),
    # Fallback: any line with "ms/step" or "ms_step"
    re.compile(
        r"step[:\s]*(?P<step>\d+).*?(?:ms[/_]step|ms_per_step)[:\s]+(?P<ms>[\d.]+)"
    ),
]

# Pattern for total step count: "step:N/M" or "steps: M"
_TOTAL_STEPS_PATTERN = re.compile(r"step[:\s]+\d+[/](?P<total>\d+)")

# Pattern to detect GPU type from log
_GPU_PATTERN = re.compile(
    r"(?P<gpu>H100 SXM|H100 PCIe|H100 NVL|A100 80GB|A100 40GB|A100 PCIe|"
    r"A10\b|L40S|L40\b|L4\b|4090|3090)",
    re.IGNORECASE,
)

# Pattern to detect GPU count from log
_GPU_COUNT_PATTERN = re.compile(
    r"(?:nproc_per_node|num_gpus|gpu_count|gpus)[=:\s]+(?P<count>\d+)"
)


@dataclass(frozen=True)
class TrainLogStats:
    """Parsed statistics from a training log file."""

    total_steps: int
    avg_ms_per_step: float
    total_time_s: float
    source_gpu: Optional[str] = None
    source_gpu_count: Optional[int] = None


@dataclass(frozen=True)
class BudgetCheck:
    """Result of a budget check against estimated time."""

    estimated_s: float
    budget_s: float
    passes: bool
    margin_pct: float


class ComputeEstimatorError(Exception):
    """Raised when log parsing or estimation fails."""


def parse_train_log(log_path: Path) -> TrainLogStats:
    """Extract training statistics from a log file.

    Parses step timing lines to compute total steps, average ms/step,
    and total training time. Also attempts to detect the source GPU
    type and count from log contents.

    Args:
        log_path: Path to the training log file.

    Returns:
        Parsed training log statistics.

    Raises:
        ComputeEstimatorError: If the log cannot be parsed or contains
            no step timing information.
    """
    if not log_path.exists():
        raise ComputeEstimatorError(f"Log file not found: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    step_times_ms: List[float] = []
    max_step: int = 0
    total_steps: Optional[int] = None

    for line in lines:
        # Try to extract total steps
        m_total = _TOTAL_STEPS_PATTERN.search(line)
        if m_total:
            total_steps = int(m_total.group("total"))

        # Try each pattern for step timing
        for pattern in _STEP_TIME_PATTERNS:
            m = pattern.search(line)
            if m:
                ms = float(m.group("ms"))
                step = int(m.group("step"))
                step_times_ms.append(ms)
                max_step = max(max_step, step)
                break

    if not step_times_ms:
        raise ComputeEstimatorError(
            f"No step timing lines found in {log_path}. "
            "Expected lines matching 'step:N/M train_time:Xms' or similar."
        )

    if total_steps is None:
        total_steps = max_step

    avg_ms = sum(step_times_ms) / len(step_times_ms)
    total_time_s = (total_steps * avg_ms) / 1000.0

    # Detect source GPU
    source_gpu: Optional[str] = None
    source_gpu_count: Optional[int] = None

    m_gpu = _GPU_PATTERN.search(text)
    if m_gpu:
        gpu_name = m_gpu.group("gpu")
        # Normalize to our lookup keys
        for key in GPU_TFLOPS:
            if key.lower() == gpu_name.lower():
                source_gpu = key
                break
        if source_gpu is None:
            source_gpu = gpu_name

    m_count = _GPU_COUNT_PATTERN.search(text)
    if m_count:
        source_gpu_count = int(m_count.group("count"))

    logger.info(
        "Parsed %s: %d steps, %.1f ms/step avg, %.1fs total, GPU=%s x%s",
        log_path.name,
        total_steps,
        avg_ms,
        total_time_s,
        source_gpu or "unknown",
        source_gpu_count or "?",
    )

    return TrainLogStats(
        total_steps=total_steps,
        avg_ms_per_step=avg_ms,
        total_time_s=total_time_s,
        source_gpu=source_gpu,
        source_gpu_count=source_gpu_count,
    )


def estimate_h100_time(
    log_path: Path,
    target_gpus: int = 8,
    target_gpu: str = "H100 SXM",
) -> float:
    """Estimate wallclock time on target hardware from a training log.

    Uses TFLOPS ratios to scale from the source hardware (detected from
    the log or defaulting to the same as target) to the target. Also
    scales by GPU count ratio if the source used fewer GPUs.

    Args:
        log_path: Path to the training log from a prior run.
        target_gpus: Number of GPUs on the target system.
        target_gpu: Target GPU model name (must be in GPU_TFLOPS).

    Returns:
        Estimated wallclock seconds on the target hardware.

    Raises:
        ComputeEstimatorError: If the target GPU is unknown or the log
            cannot be parsed.
    """
    if target_gpu not in GPU_TFLOPS:
        raise ComputeEstimatorError(
            f"Unknown target GPU: {target_gpu!r}. "
            f"Known GPUs: {', '.join(sorted(GPU_TFLOPS))}"
        )

    stats = parse_train_log(log_path)

    # Determine source TFLOPS
    source_gpu = stats.source_gpu or target_gpu
    if source_gpu not in GPU_TFLOPS:
        logger.warning(
            "Source GPU %r not in TFLOPS table; assuming same as target (%s)",
            source_gpu,
            target_gpu,
        )
        source_gpu = target_gpu

    source_tflops = GPU_TFLOPS[source_gpu]
    target_tflops = GPU_TFLOPS[target_gpu]

    # Scale by TFLOPS ratio (higher TFLOPS = less time)
    tflops_ratio = source_tflops / target_tflops

    # Scale by GPU count ratio (more GPUs = less time, assuming linear scaling)
    source_gpus = stats.source_gpu_count or target_gpus
    gpu_ratio = source_gpus / target_gpus

    estimated_s = stats.total_time_s * tflops_ratio * gpu_ratio

    logger.info(
        "Estimate: %.1fs on %d x %s (from %.1fs on %d x %s, "
        "TFLOPS ratio=%.2f, GPU ratio=%.2f)",
        estimated_s,
        target_gpus,
        target_gpu,
        stats.total_time_s,
        source_gpus,
        source_gpu,
        tflops_ratio,
        gpu_ratio,
    )

    return estimated_s


def check_within_budget(
    log_path: Path,
    budget_s: float = 600.0,
    target_gpus: int = 8,
    target_gpu: str = "H100 SXM",
) -> BudgetCheck:
    """Check whether estimated time fits within a wallclock budget.

    Args:
        log_path: Path to the training log.
        budget_s: Wallclock budget in seconds (default: 600 for PROMOTE).
        target_gpus: Number of GPUs on the target system.
        target_gpu: Target GPU model name.

    Returns:
        A BudgetCheck with the estimate, budget, pass/fail, and margin.
    """
    estimated_s = estimate_h100_time(
        log_path,
        target_gpus=target_gpus,
        target_gpu=target_gpu,
    )
    margin_pct = ((budget_s - estimated_s) / budget_s) * 100.0
    passes = estimated_s <= budget_s

    return BudgetCheck(
        estimated_s=estimated_s,
        budget_s=budget_s,
        passes=passes,
        margin_pct=margin_pct,
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU compute time estimator for training runs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # check subcommand
    check_p = sub.add_parser(
        "check",
        help="Check if a training log fits within a wallclock budget.",
    )
    check_p.add_argument("log_path", type=Path, help="Path to training log.")
    check_p.add_argument(
        "--budget",
        type=float,
        default=600.0,
        help="Wallclock budget in seconds (default: 600).",
    )
    check_p.add_argument(
        "--target-gpus",
        type=int,
        default=8,
        help="Number of target GPUs (default: 8).",
    )
    check_p.add_argument(
        "--target-gpu",
        type=str,
        default="H100 SXM",
        help="Target GPU model (default: 'H100 SXM').",
    )

    # estimate subcommand
    est_p = sub.add_parser(
        "estimate",
        help="Estimate wallclock time on target hardware.",
    )
    est_p.add_argument("log_path", type=Path, help="Path to training log.")
    est_p.add_argument(
        "--target-gpus",
        type=int,
        default=8,
        help="Number of target GPUs (default: 8).",
    )
    est_p.add_argument(
        "--target-gpu",
        type=str,
        default="H100 SXM",
        help="Target GPU model (default: 'H100 SXM').",
    )

    return parser


def main(argv: List[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "check":
        try:
            result = check_within_budget(
                log_path=args.log_path,
                budget_s=args.budget,
                target_gpus=args.target_gpus,
                target_gpu=args.target_gpu,
            )
        except ComputeEstimatorError as exc:
            logger.error("%s", exc)
            sys.exit(1)

        status = "PASS" if result.passes else "FAIL"
        print(f"{status}: estimated {result.estimated_s:.1f}s "
              f"vs {result.budget_s:.0f}s budget "
              f"(margin: {result.margin_pct:+.1f}%)")
        if not result.passes:
            sys.exit(1)

    elif args.command == "estimate":
        try:
            estimated = estimate_h100_time(
                log_path=args.log_path,
                target_gpus=args.target_gpus,
                target_gpu=args.target_gpu,
            )
        except ComputeEstimatorError as exc:
            logger.error("%s", exc)
            sys.exit(1)

        print(f"{estimated:.1f}s estimated on "
              f"{args.target_gpus}x {args.target_gpu}")


if __name__ == "__main__":
    main()
