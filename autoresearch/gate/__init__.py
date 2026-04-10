"""autoresearch.gate — GPTQ int6 gate pipeline.

Public surface
--------------
Types (from .types):
    VerdictCode, GateVerdict, GateResult

Criteria (from .criteria):
    RejectionCriteria

Compression (from .compress):
    CompressedArtifact, compress_artifact

Quantization (from .quantize):
    Quantizer            — Protocol for quantizers
    QuantizedModel       — Frozen value object produced by quantizers
    GPTQQuantizer        — Concrete int6 RTN quantizer with calibration hook
    dequantize_state_dict_int6

Evaluation (from .evaluate):
    evaluate_int6_bpb
    EvaluationError

CLI (from .gptq_gate):
    run_gate             — Programmatic entry point
    main                 — argparse CLI entry point
"""

from autoresearch.gate.compress import CompressedArtifact, compress_artifact
from autoresearch.gate.criteria import RejectionCriteria
from autoresearch.gate.evaluate import EvaluationError, evaluate_int6_bpb
from autoresearch.gate.gptq_gate import main, run_gate
from autoresearch.gate.quantize import (
    GPTQQuantizer,
    QuantizationError,
    QuantizedModel,
    Quantizer,
    dequantize_state_dict_int6,
)
from autoresearch.gate.types import GateResult, GateVerdict, VerdictCode

__all__ = [
    # types
    "VerdictCode",
    "GateVerdict",
    "GateResult",
    # criteria
    "RejectionCriteria",
    # compress
    "CompressedArtifact",
    "compress_artifact",
    # quantize
    "Quantizer",
    "QuantizedModel",
    "QuantizationError",
    "GPTQQuantizer",
    "dequantize_state_dict_int6",
    # evaluate
    "evaluate_int6_bpb",
    "EvaluationError",
    # cli
    "run_gate",
    "main",
]
