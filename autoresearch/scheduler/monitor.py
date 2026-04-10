"""Watchdog health checks for running experiments.

Each check implements the HealthCheck Protocol. The default monitor composes
NaNDetector, OOMDetector, and StallDetector. Gate agents may provide their
own HealthCheck implementations and pass them to ExperimentMonitor.
"""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, runtime_checkable

from autoresearch.gate.types import HealthCheckError
from autoresearch.db.models import ExperimentId, MetricEvent

__all__ = [
    "HealthCheck",
    "HealthCheckResult",
    "CheckSeverity",
    "NaNDetector",
    "OOMDetector",
    "StallDetector",
    "DivergenceDetector",
    "ExperimentMonitor",
    "RunState",
]

logger = logging.getLogger(__name__)

NAN_LOSS_THRESHOLD = 100.0
STALL_TIMEOUT_S = 60.0
OOM_PATTERNS = ("CUDA out of memory", "RuntimeError: CUDA", "OutOfMemoryError")
NCCL_PATTERNS = ("NCCL timeout", "NCCL error", "ncclInternalError")
DIVERGENCE_WINDOW = 50


class CheckSeverity(str):
    """Severity constants returned by health checks."""

    OK = "ok"
    WARN = "warn"
    FATAL = "fatal"


@dataclass
class HealthCheckResult:
    """Result returned by a HealthCheck implementation."""

    severity: str
    message: str
    should_kill: bool = False
    should_retry: bool = False


@runtime_checkable
class HealthCheck(Protocol):
    """Protocol for pluggable experiment health checks."""

    @property
    def name(self) -> str:
        """Short identifier for this check."""
        ...

    def check(self, state: "RunState") -> HealthCheckResult:
        """Evaluate the current run state and return a result."""
        ...


@dataclass
class RunState:
    """Mutable snapshot of a running experiment's observable state."""

    experiment_id: ExperimentId
    last_log_time: float = field(default_factory=time.monotonic)
    last_metric: Optional[MetricEvent] = None
    recent_losses: List[float] = field(default_factory=list)
    stderr_tail: List[str] = field(default_factory=list)
    retry_count: int = 0
    process_pid: Optional[int] = None

    def record_metric(self, event: MetricEvent) -> None:
        self.last_metric = event
        self.last_log_time = time.monotonic()
        self.recent_losses.append(event.train_loss)
        if len(self.recent_losses) > DIVERGENCE_WINDOW * 2:
            self.recent_losses = self.recent_losses[-DIVERGENCE_WINDOW * 2:]

    def record_stderr_line(self, line: str) -> None:
        self.stderr_tail.append(line)
        if len(self.stderr_tail) > 200:
            self.stderr_tail = self.stderr_tail[-200:]
        self.last_log_time = time.monotonic()


class NaNDetector:
    """Kills experiments immediately on NaN or exploding loss."""

    @property
    def name(self) -> str:
        return "nan_detector"

    def check(self, state: RunState) -> HealthCheckResult:
        if state.last_metric is None:
            return HealthCheckResult(CheckSeverity.OK, "no metrics yet")
        loss = state.last_metric.train_loss
        import math

        if math.isnan(loss) or math.isinf(loss):
            return HealthCheckResult(
                CheckSeverity.FATAL,
                f"NaN/Inf loss detected at step {state.last_metric.step}",
                should_kill=True,
                should_retry=False,
            )
        if loss > NAN_LOSS_THRESHOLD:
            return HealthCheckResult(
                CheckSeverity.FATAL,
                f"Loss explosion: {loss:.4f} > {NAN_LOSS_THRESHOLD} "
                f"at step {state.last_metric.step}",
                should_kill=True,
                should_retry=False,
            )
        return HealthCheckResult(CheckSeverity.OK, f"loss={loss:.4f}")


class OOMDetector:
    """Detects CUDA OOM and NCCL timeout errors in stderr."""

    @property
    def name(self) -> str:
        return "oom_detector"

    def check(self, state: RunState) -> HealthCheckResult:
        stderr_text = "\n".join(state.stderr_tail)
        for pattern in NCCL_PATTERNS:
            if pattern in stderr_text:
                return HealthCheckResult(
                    CheckSeverity.FATAL,
                    f"NCCL error detected: {pattern}",
                    should_kill=True,
                    should_retry=True,
                )
        for pattern in OOM_PATTERNS:
            if pattern in stderr_text:
                return HealthCheckResult(
                    CheckSeverity.FATAL,
                    f"OOM detected: {pattern}",
                    should_kill=True,
                    should_retry=True,
                )
        return HealthCheckResult(CheckSeverity.OK, "no oom/nccl errors")


class StallDetector:
    """Detects experiments that stop emitting log lines."""

    def __init__(self, timeout_s: float = STALL_TIMEOUT_S) -> None:
        self._timeout_s = timeout_s

    @property
    def name(self) -> str:
        return "stall_detector"

    def check(self, state: RunState) -> HealthCheckResult:
        elapsed = time.monotonic() - state.last_log_time
        if elapsed > self._timeout_s:
            return HealthCheckResult(
                CheckSeverity.FATAL,
                f"Stalled: no log output for {elapsed:.0f}s "
                f"(timeout={self._timeout_s}s)",
                should_kill=True,
                should_retry=True,
            )
        return HealthCheckResult(CheckSeverity.OK, f"last log {elapsed:.0f}s ago")


class DivergenceDetector:
    """Warns (but does not kill) when loss is consistently increasing."""

    def __init__(self, window: int = DIVERGENCE_WINDOW) -> None:
        self._window = window

    @property
    def name(self) -> str:
        return "divergence_detector"

    def check(self, state: RunState) -> HealthCheckResult:
        losses = state.recent_losses
        if len(losses) < self._window:
            return HealthCheckResult(CheckSeverity.OK, "insufficient history")
        window_losses = losses[-self._window:]
        increasing = all(
            window_losses[i] < window_losses[i + 1]
            for i in range(len(window_losses) - 1)
        )
        if increasing:
            return HealthCheckResult(
                CheckSeverity.WARN,
                f"Loss increasing for {self._window} consecutive steps",
                should_kill=False,
                should_retry=False,
            )
        return HealthCheckResult(CheckSeverity.OK, "loss not monotonically increasing")


class ExperimentMonitor:
    """Runs a composed set of HealthChecks against a RunState.

    Can be instantiated with the default checks or a custom list for testing.
    """

    def __init__(
        self,
        checks: Optional[List[HealthCheck]] = None,
        max_retries: int = 2,
    ) -> None:
        self._checks: List[HealthCheck] = checks if checks is not None else [
            NaNDetector(),
            OOMDetector(),
            StallDetector(),
            DivergenceDetector(),
        ]
        self._max_retries = max_retries

    def evaluate(self, state: RunState) -> List[HealthCheckResult]:
        """Run all checks and return a list of results.

        Does not raise — callers inspect the returned results.
        """
        results: List[HealthCheckResult] = []
        for check in self._checks:
            try:
                result = check.check(state)
            except Exception as exc:
                logger.error(
                    "Health check '%s' raised unexpectedly: %s", check.name, exc
                )
                result = HealthCheckResult(
                    CheckSeverity.WARN,
                    f"Check error: {exc}",
                    should_kill=False,
                )
            results.append(result)
            if result.severity == CheckSeverity.FATAL:
                logger.warning(
                    "[%s] Fatal check '%s': %s",
                    state.experiment_id,
                    check.name,
                    result.message,
                )
            elif result.severity == CheckSeverity.WARN:
                logger.warning(
                    "[%s] Warning from '%s': %s",
                    state.experiment_id,
                    check.name,
                    result.message,
                )
        return results

    def should_kill(self, results: List[HealthCheckResult]) -> bool:
        return any(r.should_kill for r in results)

    def should_retry(
        self, results: List[HealthCheckResult], state: RunState
    ) -> bool:
        retry_requested = any(r.should_retry for r in results)
        return retry_requested and state.retry_count < self._max_retries
