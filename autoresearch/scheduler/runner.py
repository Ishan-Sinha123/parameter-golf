"""torchrun wrapper, log parsing, and metrics emission for AutoResearch v2."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

from autoresearch.config import SystemConfig
from autoresearch.gate.types import (
    MetricsParseError,
    ProcessLaunchError,
    RunnerError,
)
from autoresearch.scheduler.monitor import RunState
from autoresearch.db.models import (
    EnvOverrides,
    ExperimentConfig,
    ExperimentId,
    GPUSlot,
    MetricEvent,
    Stage,
)

__all__ = [
    "RunCommandBuilder",
    "RunnerContext",
    "MetricsParser",
    "parse_metric_line",
    "RunResult",
]

logger = logging.getLogger(__name__)

# Regex patterns for log line parsing
_STEP_RE = re.compile(
    r"step:(\d+).*?loss:([\d.eE+\-]+).*?lr:([\d.eE+\-]+)"
    r".*?ms_step:([\d.eE+\-]+).*?mem:([\d.eE+\-]+)GB",
    re.IGNORECASE,
)
_TOKENS_RE = re.compile(r"tokens_seen:(\d+)", re.IGNORECASE)
_TRAIN_BPB_RE = re.compile(r"train_bpb[:\s]+([\d.]+)", re.IGNORECASE)
_EMA_BPB_RE = re.compile(r"ema_bpb[:\s]+([\d.]+)", re.IGNORECASE)
_INT6_BPB_RE = re.compile(r"(?:int6_bpb|final_int8_zlib_roundtrip)[:\s]+([\d.]+)", re.IGNORECASE)
_QUANT_GAP_RE = re.compile(r"quant_gap[:\s]+([\d.]+)", re.IGNORECASE)
_ARTIFACT_MB_RE = re.compile(
    r"(?:artifact_mb|Total submission size[^:]*:)\s*([\d.]+)", re.IGNORECASE
)


@dataclass
class RunResult:
    """Summary of a completed training run."""

    experiment_id: ExperimentId
    stage: Stage
    exit_code: int
    wallclock_s: float
    train_bpb: Optional[float] = None
    ema_bpb: Optional[float] = None
    int6_bpb: Optional[float] = None
    quant_gap: Optional[float] = None
    artifact_mb: Optional[float] = None
    steps: Optional[int] = None
    ms_per_step: Optional[float] = None
    error_message: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


class RunCommandBuilder:
    """Builder for torchrun subprocess commands.

    Usage:
        cmd = (
            RunCommandBuilder(config, slot, stage, experiment_dir)
            .with_wallclock(180)
            .with_extra_env({"SEED": "42"})
            .build()
        )
    """

    def __init__(
        self,
        config: ExperimentConfig,
        slot: GPUSlot,
        stage: Stage,
        experiment_dir: Path,
    ) -> None:
        self._config = config
        self._slot = slot
        self._stage = stage
        self._experiment_dir = experiment_dir
        self._wallclock_s: Optional[int] = None
        self._extra_env: EnvOverrides = {}

    def with_wallclock(self, seconds: int) -> "RunCommandBuilder":
        self._wallclock_s = seconds
        return self

    def with_extra_env(self, env: EnvOverrides) -> "RunCommandBuilder":
        self._extra_env = {**self._extra_env, **env}
        return self

    def build(self) -> Tuple[List[str], Dict[str, str]]:
        """Return (argv, env_dict) ready for subprocess.Popen."""
        env = {**os.environ}
        env["CUDA_VISIBLE_DEVICES"] = self._slot.cuda_visible_devices
        env["EXPERIMENT_DESC"] = self._config.name
        env["RESULTS_TSV_PATH"] = str(
            self._experiment_dir / "results.tsv"
        )
        if self._wallclock_s is not None:
            env["MAX_WALLCLOCK_SECONDS"] = str(self._wallclock_s)

        env.update(self._config.env_overrides)
        env.update(self._extra_env)

        n_procs = str(self._slot.gpu_count)
        argv = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={n_procs}",
            self._config.script_path,
        ]
        return argv, env

    def build_gate_command(self, gate_script: str) -> Tuple[List[str], Dict[str, str]]:
        """Build a single-GPU gate (GPTQ) command."""
        env = {**os.environ}
        env["CUDA_VISIBLE_DEVICES"] = str(self._slot.gpu_indices[0])
        checkpoint = str(
            self._experiment_dir / "screen_checkpoint.pt"
        )
        output = str(self._experiment_dir / "gate_results.json")
        argv = [
            "python",
            gate_script,
            "--checkpoint", checkpoint,
            "--output", output,
        ]
        return argv, env


@contextmanager
def RunnerContext(
    argv: List[str],
    env: Dict[str, str],
    log_path: Path,
    run_state: RunState,
) -> Generator[subprocess.Popen, None, None]:  # type: ignore[type-arg]
    """Context manager wrapping a subprocess.Popen.

    Redirects stdout and stderr to log_path. Feeds stderr lines into
    run_state for OOM/NCCL detection. Kills the process on context exit
    if it has not already terminated.

    Raises:
        ProcessLaunchError: If the process cannot be started.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = None
    proc = None
    try:
        log_fh = log_path.open("wb")
        proc = subprocess.Popen(
            argv,
            env=env,
            stdout=log_fh,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        run_state.process_pid = proc.pid
        logger.info("Launched pid=%d: %s", proc.pid, " ".join(argv[:3]))

        def _drain_stderr() -> None:
            assert proc is not None
            assert proc.stderr is not None
            for raw_line in proc.stderr:
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                run_state.record_stderr_line(line)
                if log_fh and not log_fh.closed:
                    log_fh.write(raw_line)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        yield proc

        stderr_thread.join(timeout=5.0)
    except (OSError, ValueError) as exc:
        raise ProcessLaunchError(
            f"Failed to launch {argv[0]}: {exc}"
        ) from exc
    finally:
        if proc is not None and proc.poll() is None:
            logger.warning("Killing process pid=%d on context exit", proc.pid)
            proc.kill()
            proc.wait()
        if log_fh is not None:
            log_fh.close()


def parse_metric_line(line: str, timestamp: Optional[float] = None) -> Optional[MetricEvent]:
    """Parse a single log line into a MetricEvent, returning None if not a step line."""
    m = _STEP_RE.search(line)
    if m is None:
        return None
    try:
        step = int(m.group(1))
        train_loss = float(m.group(2))
        lr = float(m.group(3))
        ms_step = float(m.group(4))
        gpu_mem_gb = float(m.group(5))

        tokens_m = _TOKENS_RE.search(line)
        tokens_seen = int(tokens_m.group(1)) if tokens_m else 0

        return MetricEvent(
            timestamp=timestamp or time.time(),
            step=step,
            train_loss=train_loss,
            lr=lr,
            ms_step=ms_step,
            gpu_mem_gb=gpu_mem_gb,
            tokens_seen=tokens_seen,
        )
    except (ValueError, IndexError) as exc:
        raise MetricsParseError(f"Cannot parse step line: {exc!r} in: {line!r}") from exc


class MetricsParser:
    """Generator-based parser that yields MetricEvents from a log file.

    Usage:
        parser = MetricsParser(log_path)
        for event in parser.tail():
            process(event)
    """

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path

    def tail(
        self, poll_interval: float = 0.5
    ) -> Generator[MetricEvent, None, None]:
        """Yield MetricEvents as lines appear in the log file."""
        with self._log_path.open("r", encoding="utf-8", errors="replace") as fh:
            while True:
                line = fh.readline()
                if not line:
                    time.sleep(poll_interval)
                    continue
                event = parse_metric_line(line)
                if event is not None:
                    yield event

    def parse_final_metrics(self) -> Dict[str, Optional[float]]:
        """Scan the complete log for summary metrics after a run finishes."""
        result: Dict[str, Optional[float]] = {
            "train_bpb": None,
            "ema_bpb": None,
            "int6_bpb": None,
            "quant_gap": None,
            "artifact_mb": None,
        }
        if not self._log_path.exists():
            return result
        text = self._log_path.read_text(encoding="utf-8", errors="replace")
        for key, pattern in (
            ("train_bpb", _TRAIN_BPB_RE),
            ("ema_bpb", _EMA_BPB_RE),
            ("int6_bpb", _INT6_BPB_RE),
            ("quant_gap", _QUANT_GAP_RE),
            ("artifact_mb", _ARTIFACT_MB_RE),
        ):
            m = pattern.search(text)
            if m:
                try:
                    result[key] = float(m.group(1))
                except ValueError:
                    pass
        return result

    def parse_step_count(self) -> Optional[int]:
        """Return the highest step number seen in the log."""
        highest = None
        if not self._log_path.exists():
            return None
        with self._log_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _STEP_RE.search(line)
                if m:
                    try:
                        step = int(m.group(1))
                        if highest is None or step > highest:
                            highest = step
                    except ValueError:
                        pass
        return highest


class ExperimentRunner:
    """Orchestrates a single experiment stage run end-to-end."""

    def __init__(
        self,
        system_config: SystemConfig,
        workspace: Path,
    ) -> None:
        self._cfg = system_config
        self._workspace = workspace

    def run_screen(
        self,
        config: ExperimentConfig,
        slot: GPUSlot,
        run_state: RunState,
    ) -> RunResult:
        """Execute the screen stage for an experiment."""
        exp_dir = self._workspace / self._cfg.experiments_dir / config.id
        exp_dir.mkdir(parents=True, exist_ok=True)
        log_path = exp_dir / "train.log"
        metrics_path = exp_dir / "metrics.jsonl"

        builder = RunCommandBuilder(config, slot, Stage.SCREEN, exp_dir)
        builder.with_wallclock(self._cfg.screen_wallclock_s)
        argv, env = builder.build()

        start_time = time.monotonic()
        with RunnerContext(argv, env, log_path, run_state) as proc:
            self._stream_metrics(proc, log_path, metrics_path, run_state)
            proc.wait()
        elapsed = time.monotonic() - start_time

        parser = MetricsParser(log_path)
        metrics = parser.parse_final_metrics()
        step_count = parser.parse_step_count()

        return RunResult(
            experiment_id=config.id,
            stage=Stage.SCREEN,
            exit_code=proc.returncode,
            wallclock_s=elapsed,
            train_bpb=metrics["train_bpb"],
            ema_bpb=metrics["ema_bpb"],
            steps=step_count,
        )

    def run_gate(
        self,
        config: ExperimentConfig,
        slot: GPUSlot,
        run_state: RunState,
        gate_script: str = "gptq_gate.py",
    ) -> RunResult:
        """Execute the GPTQ gate stage for an experiment."""
        exp_dir = self._workspace / self._cfg.experiments_dir / config.id
        log_path = exp_dir / "gate.log"

        builder = RunCommandBuilder(config, slot, Stage.GATE, exp_dir)
        argv, env = builder.build_gate_command(gate_script)

        start_time = time.monotonic()
        with RunnerContext(argv, env, log_path, run_state) as proc:
            proc.wait()
        elapsed = time.monotonic() - start_time

        int6_bpb: Optional[float] = None
        quant_gap: Optional[float] = None
        artifact_mb: Optional[float] = None
        gate_results_path = exp_dir / "gate_results.json"
        if gate_results_path.exists():
            try:
                gate_data = json.loads(gate_results_path.read_text())
                int6_bpb = gate_data.get("int6_bpb")
                quant_gap = gate_data.get("quant_gap")
                artifact_mb = gate_data.get("artifact_mb")
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Cannot read gate results for %s: %s", config.id, exc)

        return RunResult(
            experiment_id=config.id,
            stage=Stage.GATE,
            exit_code=proc.returncode,
            wallclock_s=elapsed,
            int6_bpb=int6_bpb,
            quant_gap=quant_gap,
            artifact_mb=artifact_mb,
        )

    def run_promote(
        self,
        config: ExperimentConfig,
        slot: GPUSlot,
        run_state: RunState,
    ) -> RunResult:
        """Execute the promote (full-budget) stage for an experiment."""
        exp_dir = self._workspace / self._cfg.experiments_dir / config.id
        log_path = exp_dir / "promote.log"
        metrics_path = exp_dir / "promote_metrics.jsonl"

        builder = RunCommandBuilder(config, slot, Stage.PROMOTE, exp_dir)
        builder.with_wallclock(self._cfg.promote_wallclock_s)
        argv, env = builder.build()

        start_time = time.monotonic()
        with RunnerContext(argv, env, log_path, run_state) as proc:
            self._stream_metrics(proc, log_path, metrics_path, run_state)
            proc.wait()
        elapsed = time.monotonic() - start_time

        parser = MetricsParser(log_path)
        metrics = parser.parse_final_metrics()
        step_count = parser.parse_step_count()

        return RunResult(
            experiment_id=config.id,
            stage=Stage.PROMOTE,
            exit_code=proc.returncode,
            wallclock_s=elapsed,
            train_bpb=metrics["train_bpb"],
            ema_bpb=metrics["ema_bpb"],
            int6_bpb=metrics["int6_bpb"],
            quant_gap=metrics["quant_gap"],
            artifact_mb=metrics["artifact_mb"],
            steps=step_count,
        )

    def run_inspect(
        self,
        config: ExperimentConfig,
        run_state: RunState,
    ) -> RunResult:
        """Run generate_demo.py on a screen checkpoint for qualitative inspection.

        Captures output to inspect_samples.txt in the experiment directory.
        Runs on CPU (no GPU slot required). Never auto-rejects.
        """
        exp_dir = self._workspace / self._cfg.experiments_dir / config.id
        checkpoint = exp_dir / "screen_checkpoint.pt"
        output_path = exp_dir / "inspect_samples.txt"
        log_path = exp_dir / "inspect.log"

        generate_demo = self._workspace.parent / "scripts" / "generate_demo.py"
        if not generate_demo.exists():
            # Try relative to workspace
            generate_demo = self._workspace / "scripts" / "generate_demo.py"

        if not generate_demo.exists():
            logger.warning(
                "generate_demo.py not found; skipping inspect for %s",
                config.id,
            )
            return RunResult(
                experiment_id=config.id,
                stage=Stage.SCREEN,  # no INSPECT stage enum in Stage
                exit_code=0,
                wallclock_s=0.0,
                error_message="generate_demo.py not found; inspect skipped",
            )

        argv = [
            "python3",
            str(generate_demo),
            "--checkpoint", str(checkpoint),
            "--plain",
            "--prompt", "The most important thing about",
            "--max-new-tokens", "128",
        ]
        env = {**os.environ}
        env["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only

        start_time = time.monotonic()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                argv,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            elapsed = time.monotonic() - start_time

            # Write captured output to inspect_samples.txt
            output_path.write_text(
                result.stdout + result.stderr,
                encoding="utf-8",
            )
            # Also write to log
            log_path.write_text(
                f"argv: {' '.join(argv)}\n"
                f"exit_code: {result.returncode}\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}\n",
                encoding="utf-8",
            )

            logger.info(
                "Inspect completed for %s (exit=%d, %.1fs)",
                config.id, result.returncode, elapsed,
            )
            return RunResult(
                experiment_id=config.id,
                stage=Stage.SCREEN,
                exit_code=result.returncode,
                wallclock_s=elapsed,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - start_time
            logger.warning("Inspect timed out for %s after %.1fs", config.id, elapsed)
            return RunResult(
                experiment_id=config.id,
                stage=Stage.SCREEN,
                exit_code=-1,
                wallclock_s=elapsed,
                error_message="Inspect timed out after 120s",
            )
        except OSError as exc:
            elapsed = time.monotonic() - start_time
            logger.error("Inspect failed for %s: %s", config.id, exc)
            return RunResult(
                experiment_id=config.id,
                stage=Stage.SCREEN,
                exit_code=-1,
                wallclock_s=elapsed,
                error_message=str(exc),
            )

    def _stream_metrics(
        self,
        proc: subprocess.Popen,  # type: ignore[type-arg]
        log_path: Path,
        metrics_path: Path,
        run_state: RunState,
    ) -> None:
        """Background thread: parse log lines and write metrics.jsonl."""
        def _worker() -> None:
            if not log_path.exists():
                return
            with (
                log_path.open("r", encoding="utf-8", errors="replace") as log_fh,
                metrics_path.open("w", encoding="utf-8") as metrics_fh,
            ):
                while proc.poll() is None:
                    line = log_fh.readline()
                    if not line:
                        time.sleep(0.1)
                        continue
                    run_state.record_stderr_line(line)
                    event = parse_metric_line(line)
                    if event is not None:
                        run_state.record_metric(event)
                        metrics_fh.write(json.dumps({
                            "t": event.timestamp,
                            "step": event.step,
                            "train_loss": event.train_loss,
                            "lr": event.lr,
                            "ms_step": event.ms_step,
                            "gpu_mem_gb": event.gpu_mem_gb,
                            "tokens_seen": event.tokens_seen,
                        }) + "\n")
                        metrics_fh.flush()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
