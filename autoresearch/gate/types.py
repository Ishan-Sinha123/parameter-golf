"""Gate-specific types: GateResult, GateVerdict, VerdictCode."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = ["VerdictCode", "GateVerdict", "GateResult",
           "AutoResearchError", "HealthCheckError"]


class AutoResearchError(Exception):
    """Base exception for the autoresearch system."""

class HealthCheckError(AutoResearchError):
    """Error in health check evaluation."""

class RunnerError(AutoResearchError):
    """Error in experiment runner."""

class ProcessLaunchError(RunnerError):
    """Failed to launch training process."""

class ProcessTimeoutError(RunnerError):
    """Training process exceeded wallclock budget."""

class MetricsParseError(RunnerError):
    """Failed to parse metrics from training log."""


class VerdictCode(Enum):
    PASS = "pass"
    FAIL_QUANT_GAP = "fail_quant_gap"
    FAIL_ARTIFACT_SIZE = "fail_artifact_size"
    FAIL_BPB_REGRESSION = "fail_bpb_regression"
    FAIL_OOM = "fail_oom"
    FAIL_CORRUPT = "fail_corrupt"
    FAIL_CHECKPOINT_NOT_FOUND = "fail_checkpoint_not_found"
    FAIL_UNKNOWN = "fail_unknown"


@dataclass(frozen=True)
class GateVerdict:
    """Outcome of applying all rejection criteria to gate measurements."""

    passed: bool
    code: VerdictCode
    reason: Optional[str]


@dataclass(frozen=True)
class GateResult:
    """Full result record written to the output JSON after a gate run.

    quant_gap = int6_bpb - fp32_bpb (positive means quantization hurts BPB).
    """

    passed: bool
    int6_bpb: float
    quant_gap: float
    artifact_mb: float
    rejection_reason: Optional[str]
    gptq_time_s: float
    calibration_tokens: int
    peak_memory_mb: float
