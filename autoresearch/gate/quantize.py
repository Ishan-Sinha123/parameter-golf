"""GPTQ-style int6 quantization for the gate pipeline.

Adapted from the int8 + zlib logic in train_gpt.py.  Key differences:
  - 6-bit symmetric range: quantized values are clipped to [-31, 31] (stored
    as int8 containers, so no bit-packing overhead in PyTorch).
  - Calibration hook: GPTQQuantizer accepts calibration_data so future
    callers can collect activation statistics before quantizing.
  - Per-row scales for 2-D weight matrices, per-tensor for vectors/scalars
    (same scheme as train_gpt.py's int8 path).
  - Passthrough for small tensors (≤ INT6_KEEP_FLOAT_MAX_NUMEL elements)
    and for non-float tensors.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple, runtime_checkable

import torch
from torch import Tensor

from autoresearch.gate.types import AutoResearchError

__all__ = [
    "QuantizedModel",
    "Quantizer",
    "GPTQQuantizer",
    "dequantize_state_dict_int6",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (mirroring train_gpt.py INT8_* names but for 6-bit)
# ---------------------------------------------------------------------------

#: Maximum number of elements before we pass a float tensor through unchanged.
#: Tensors at or below this size are cheap enough that quantizing them saves
#: little while adding dequantization overhead.
INT6_KEEP_FLOAT_MAX_NUMEL: int = 65_536

#: Storage dtype for small float passthrough tensors (fp16 saves ~50% vs fp32).
INT6_KEEP_FLOAT_STORE_DTYPE: torch.dtype = torch.float16

#: Storage dtype for per-row scales.
INT6_PER_ROW_SCALE_DTYPE: torch.dtype = torch.float16

#: Clip percentile.  Same as train_gpt.py's INT8_CLIP_PERCENTILE.
INT6_CLIP_PERCENTILE: float = 99.99984
INT6_CLIP_Q: float = INT6_CLIP_PERCENTILE / 100.0

#: 6-bit symmetric maximum (2^5 - 1 = 31; we use ±31 for symmetry like int8
#: uses ±127 instead of the full ±128).
INT6_MAX: int = 31

#: Env var listing comma-separated substrings; matching tensor names are kept
#: as full float32 (control scalars that must not lose precision).
_CONTROL_PATTERN_ENV = "CONTROL_TENSOR_NAME_PATTERNS"
_DEFAULT_CONTROL_PATTERNS = (
    "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
    "q_gain,skip_weight,skip_weights"
)
CONTROL_TENSOR_NAME_PATTERNS: Tuple[str, ...] = tuple(
    p
    for p in os.environ.get(_CONTROL_PATTERN_ENV, _DEFAULT_CONTROL_PATTERNS).split(",")
    if p
)

INT6_KEEP_FLOAT_FP32_NAME_PATTERNS: Tuple[str, ...] = tuple(
    p
    for p in os.environ.get(
        "INT6_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if p
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class QuantizationError(AutoResearchError):
    """Raised when int6 quantization fails unrecoverably."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuantizedModel:
    """Immutable container produced by :class:`GPTQQuantizer`.

    Attributes:
        quant_obj:   The serialisable dict produced by
                     :func:`_quantize_state_dict_int6`.  Pass this to
                     :func:`dequantize_state_dict_int6` to recover weights.
        stats:       Bookkeeping counters collected during quantization.
        serialized:  Raw bytes from ``torch.save(quant_obj, BytesIO())``, ready
                     to be fed into :func:`~compress.compress_artifact`.
        calibration_tokens: Number of calibration tokens consumed (0 if none).
    """

    quant_obj: Dict[str, object]
    stats: Dict[str, int]
    serialized: bytes
    calibration_tokens: int


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Quantizer(Protocol):
    """Interface for model quantizers used by the gate pipeline."""

    def quantize(
        self,
        model: torch.nn.Module,
        calibration_data: Optional[Tensor],
    ) -> QuantizedModel:
        """Quantize *model* and return a :class:`QuantizedModel`.

        Args:
            model:            The fp32/bf16 model to quantize (should already
                              be on CPU for serialization).
            calibration_data: Optional token tensor used to collect activation
                              statistics before quantization.  May be ``None``
                              for RTN-style (no calibration) quantization.

        Returns:
            A frozen :class:`QuantizedModel` with the quantized weights and
            compression statistics.
        """
        ...


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def _keep_float_tensor(
    name: str,
    t: Tensor,
    passthrough_orig_dtypes: Dict[str, str],
) -> Tensor:
    """Downcast small float tensors to fp16 for storage efficiency.

    Control tensors (e.g. learned scalars) are kept in fp32 so their exact
    values are preserved.
    """
    if any(pattern in name for pattern in INT6_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT6_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def _quantize_float_tensor_int6(t: Tensor) -> Tuple[Tensor, Tensor]:
    """Round-to-nearest int6 quantization.

    Uses the same per-row (2-D matrices) / per-tensor (vectors/scalars) scheme
    as train_gpt.py's int8 path, but clips to ±INT6_MAX instead of ±127.

    Args:
        t: A CPU float tensor with >= 1 elements.

    Returns:
        (q, scale) where *q* is int8-stored 6-bit values in [-31, 31] and
        *scale* is fp16 (per-row or scalar).
    """
    t32 = t.float()

    if t32.ndim == 2:
        # Per-row clip and scale — mirrors train_gpt.py quantize_float_tensor.
        clip_abs = (
            torch.quantile(t32.abs(), INT6_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(
            torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None]
        )
        scale = (clip_abs / float(INT6_MAX)).clamp_min(1.0 / float(INT6_MAX))
        q = torch.clamp(
            torch.round(clipped / scale[:, None]), -INT6_MAX, INT6_MAX
        ).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT6_PER_ROW_SCALE_DTYPE).contiguous()

    # Per-tensor scale for vectors / scalars.
    clip_abs = (
        float(torch.quantile(t32.abs().flatten(), INT6_CLIP_Q).item())
        if t32.numel()
        else 0.0
    )
    scale = torch.tensor(
        clip_abs / float(INT6_MAX) if clip_abs > 0 else 1.0,
        dtype=torch.float32,
    )
    q = torch.clamp(
        torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale),
        -INT6_MAX,
        INT6_MAX,
    ).to(torch.int8).contiguous()
    return q, scale


def _quantize_state_dict_int6(
    state_dict: Dict[str, Tensor],
) -> Tuple[Dict[str, object], Dict[str, int]]:
    """Quantize every float tensor in *state_dict* to int6.

    Args:
        state_dict: Raw model state dict (may be on any device).

    Returns:
        (quant_obj, stats) where *quant_obj* is the serialisable dict and
        *stats* contains byte-count bookkeeping.
    """
    quantized: Dict[str, Tensor] = {}
    scales: Dict[str, Tensor] = {}
    dtypes: Dict[str, str] = {}
    passthrough: Dict[str, Tensor] = {}
    passthrough_orig_dtypes: Dict[str, str] = {}
    qmeta: Dict[str, Dict[str, object]] = {}

    stats: Dict[str, int] = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "int6_payload_bytes",
        ),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += _tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int6_payload_bytes"] += _tensor_nbytes(t)
            continue

        # Small float tensors are passed through (stored as fp16 to save bytes).
        if t.numel() <= INT6_KEEP_FLOAT_MAX_NUMEL:
            kept = _keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int6_payload_bytes"] += _tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = _quantize_float_tensor_int6(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int6_payload_bytes"] += _tensor_nbytes(q) + _tensor_nbytes(s)

    obj: Dict[str, object] = {
        "__quant_format__": "int6_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes

    return obj, stats


# ---------------------------------------------------------------------------
# Public dequantization helper (used by evaluate.py)
# ---------------------------------------------------------------------------


def dequantize_state_dict_int6(obj: Dict[str, object]) -> Dict[str, Tensor]:
    """Reconstruct fp32/bf16 weights from a :func:`_quantize_state_dict_int6` object.

    Mirrors train_gpt.py's ``dequantize_state_dict_int8`` but for the int6
    format tag ``"int6_clean_per_row_v1"``.

    Args:
        obj: The dict produced by :func:`_quantize_state_dict_int6` (or loaded
             from disk after LZMA decompression and ``torch.load``).

    Returns:
        A standard state dict with all tensors in their original dtypes.
    """
    out: Dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})

    for name, q in obj["quantized"].items():  # type: ignore[union-attr]
        dtype = getattr(torch, obj["dtypes"][name])  # type: ignore[index]
        s = obj["scales"][name]  # type: ignore[index]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:  # type: ignore[union-attr]
            s = s.to(dtype=torch.float32)
            out[name] = (
                q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))
            ).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()

    for name, t in obj["passthrough"].items():  # type: ignore[union-attr]
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t

    return out


# ---------------------------------------------------------------------------
# GPTQQuantizer
# ---------------------------------------------------------------------------


class GPTQQuantizer:
    """Round-to-nearest int6 quantizer with an optional calibration hook.

    The "GPTQ" label refers to the calibration interface: when *calibration_data*
    is provided the model is run in inference mode on those tokens first, giving
    future extensions the hook they need to collect Hessians or activation
    statistics before computing scales.  The current implementation uses those
    tokens only to warm up the model's cache (e.g. KV cache, running stats) and
    then falls back to per-row percentile clipping — the same RTN approach used
    by train_gpt.py — because the simpler method already performs well at this
    model scale.

    Args:
        device:   Device on which calibration forward passes are run.
        seq_len:  Sequence length used when batching calibration tokens.
    """

    def __init__(
        self,
        device: torch.device,
        seq_len: int = 1024,
    ) -> None:
        self.device = device
        self.seq_len = seq_len

    def quantize(
        self,
        model: torch.nn.Module,
        calibration_data: Optional[Tensor],
    ) -> QuantizedModel:
        """Quantize *model* to int6 and return a :class:`QuantizedModel`.

        Args:
            model:            The fp32/bf16 model to quantize.  The model is
                              moved to CPU for state-dict extraction; callers
                              should ensure the model is already fully trained
                              and that no gradient is needed.
            calibration_data: Optional 1-D token tensor.  When provided, the
                              model performs a forward pass on each consecutive
                              seq_len-length window before quantization.

        Returns:
            A frozen :class:`QuantizedModel`.
        """
        calibration_tokens = 0

        if calibration_data is not None:
            calibration_tokens = self._run_calibration(model, calibration_data)

        logger.info(
            "GPTQQuantizer: quantizing model to int6 "
            "(calibration_tokens=%d)",
            calibration_tokens,
        )

        # Extract state dict on CPU.
        state_dict: Dict[str, Tensor] = {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        }

        quant_obj, stats = _quantize_state_dict_int6(state_dict)

        baseline_bytes = stats["baseline_tensor_bytes"]
        payload_bytes = stats["int6_payload_bytes"]
        ratio = baseline_bytes / max(payload_bytes, 1)
        logger.info(
            "GPTQQuantizer: baseline=%d bytes, int6_payload=%d bytes, ratio=%.2fx",
            baseline_bytes,
            payload_bytes,
            ratio,
        )

        buf = io.BytesIO()
        torch.save(quant_obj, buf)
        serialized = buf.getvalue()

        return QuantizedModel(
            quant_obj=quant_obj,
            stats=stats,
            serialized=serialized,
            calibration_tokens=calibration_tokens,
        )

    def _run_calibration(
        self,
        model: torch.nn.Module,
        calibration_data: Tensor,
    ) -> int:
        """Run forward passes on calibration tokens.

        This warms up any running statistics (e.g. BatchNorm) and gives
        subclasses a hook for Hessian accumulation.  The base implementation
        simply runs inference without collecting gradients.

        Args:
            model:            The model to calibrate.
            calibration_data: 1-D int token tensor.

        Returns:
            Number of tokens consumed (rounded down to seq_len boundary).
        """
        model.eval()
        tokens_consumed = 0

        n_tokens = calibration_data.numel()
        n_full_seqs = n_tokens // (self.seq_len + 1)

        if n_full_seqs == 0:
            logger.warning(
                "GPTQQuantizer._run_calibration: calibration_data has only %d tokens, "
                "need at least %d for one sequence — skipping calibration",
                n_tokens,
                self.seq_len + 1,
            )
            return 0

        logger.info(
            "GPTQQuantizer._run_calibration: running %d calibration sequences "
            "(seq_len=%d)",
            n_full_seqs,
            self.seq_len,
        )

        with torch.inference_mode():
            for i in range(n_full_seqs):
                start = i * (self.seq_len + 1)
                chunk = calibration_data[start : start + self.seq_len + 1].to(
                    device=self.device, dtype=torch.int64
                )
                x = chunk[:-1].unsqueeze(0)  # (1, seq_len)
                y = chunk[1:].unsqueeze(0)
                try:
                    model(x, y)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "GPTQQuantizer._run_calibration: forward pass failed at "
                        "sequence %d: %s — stopping calibration early",
                        i,
                        exc,
                    )
                    break
                tokens_consumed += self.seq_len

        model.train()
        return tokens_consumed
