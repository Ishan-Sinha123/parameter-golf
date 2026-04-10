"""Int6 roundtrip BPB evaluation for the gate pipeline.

Adapted from train_gpt.py's ``eval_val`` function.  Key differences:
  - Single-GPU / CPU only (the gate runs on one process).
  - No distributed all-reduce.
  - Accepts pre-built LUT tensors or builds trivial byte-count LUTs from
    vocab_size when sentencepiece is not available in the gate environment.
  - Returns only BPB (not val_loss) since that is the gate's decision metric.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch import Tensor

from autoresearch.gate.types import AutoResearchError
from autoresearch.gate.quantize import QuantizedModel, dequantize_state_dict_int6

__all__ = ["evaluate_int6_bpb", "EvaluationError"]

logger = logging.getLogger(__name__)


class EvaluationError(AutoResearchError):
    """Raised when BPB evaluation fails unrecoverably."""


def _make_uniform_byte_luts(
    vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Build trivial LUT tensors that count every token as 1 byte.

    When an accurate sentencepiece LUT is not available (e.g. in CI or unit
    tests), this gives a correct *relative* comparison between fp32 and int6 BPB
    because the same LUTs are applied to both runs.

    Args:
        vocab_size: Number of vocabulary entries.
        device:     Target device.

    Returns:
        (base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
        matching the signature expected by ``_compute_bpb``.
    """
    base_bytes = torch.ones(vocab_size, dtype=torch.int32, device=device)
    has_leading_space = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    is_boundary = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    return base_bytes, has_leading_space, is_boundary


def _compute_bpb(
    model: torch.nn.Module,
    val_tokens: Tensor,
    seq_len: int,
    batch_tokens: int,
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> float:
    """Run one full evaluation pass and return bits-per-byte.

    Mirrors the inner loop of train_gpt.py's ``eval_val`` but without the
    distributed all-reduce, multi-rank slicing, or world_size / grad_accum
    scaling.

    Args:
        model:                  The model to evaluate (already loaded with
                                dequantized weights).
        val_tokens:             1-D uint16/int64 token tensor.
        seq_len:                Sequence length (tokens per context window).
        batch_tokens:           Total tokens to feed per evaluation batch.
                                Must be >= seq_len.
        device:                 CUDA or CPU device.
        base_bytes_lut:         Shape (vocab_size,) int32 — bytes per token
                                (without leading-space adjustment).
        has_leading_space_lut:  Shape (vocab_size,) bool — True if token is
                                rendered with a leading space.
        is_boundary_token_lut:  Shape (vocab_size,) bool — True if token ends
                                a UTF-8 boundary (suppress leading-space byte
                                count for the *next* token).

    Returns:
        Bits-per-byte (float).
    """
    local_batch_seqs = max(batch_tokens // seq_len, 1)
    total_seqs = (val_tokens.numel() - 1) // seq_len

    if total_seqs == 0:
        raise EvaluationError(
            f"val_tokens has only {val_tokens.numel()} tokens — need at least "
            f"{seq_len + 1} for one sequence"
        )

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for seq_start in range(0, total_seqs, local_batch_seqs):
            seq_end = min(seq_start + local_batch_seqs, total_seqs)
            raw_start = seq_start * seq_len
            raw_end = seq_end * seq_len + 1

            local = val_tokens[raw_start:raw_end].to(
                device=device, dtype=torch.int64, non_blocking=True
            )
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)

            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
            ):
                batch_loss = model(x, y).detach()

            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count

            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()

    bpb = float(bits_per_token * tokens_per_byte)
    logger.debug(
        "_compute_bpb: val_loss=%.6f bpb=%.6f tokens=%.0f bytes=%.0f",
        val_loss.item(),
        bpb,
        val_token_count.item(),
        val_byte_count.item(),
    )
    return bpb


def evaluate_int6_bpb(
    model: torch.nn.Module,
    quantized_model: QuantizedModel,
    val_tokens: Tensor,
    device: torch.device,
    seq_len: int = 1024,
    batch_tokens: int = 524_288,
    base_bytes_lut: Optional[Tensor] = None,
    has_leading_space_lut: Optional[Tensor] = None,
    is_boundary_token_lut: Optional[Tensor] = None,
) -> float:
    """Load int6-dequantized weights into *model* and measure BPB.

    This is the full roundtrip: dequantize → load_state_dict → eval forward
    pass.  The model's weights are mutated in place; callers that need the
    original fp32 weights back should save a copy beforehand.

    Args:
        model:                  The model architecture (state dict will be
                                overwritten with dequantized int6 weights).
        quantized_model:        Output from :class:`~quantize.GPTQQuantizer`.
        val_tokens:             1-D token tensor for evaluation.
        device:                 Device to run evaluation on.
        seq_len:                Sequence length in tokens.
        batch_tokens:           Tokens per evaluation batch (lower → less VRAM).
        base_bytes_lut:         Optional pre-built LUT (see ``_compute_bpb``).
                                If None, uniform 1-byte-per-token LUTs are used.
        has_leading_space_lut:  Optional LUT (see above).
        is_boundary_token_lut:  Optional LUT (see above).

    Returns:
        Bits-per-byte after int6 dequantization.

    Raises:
        EvaluationError: If dequantization or the evaluation forward pass fails.
    """
    logger.info(
        "evaluate_int6_bpb: dequantizing and loading int6 weights onto %s",
        device,
    )

    try:
        recovered_state = dequantize_state_dict_int6(quantized_model.quant_obj)
    except Exception as exc:
        raise EvaluationError(
            f"Failed to dequantize int6 weights: {exc}"
        ) from exc

    try:
        model.load_state_dict(recovered_state, strict=True)
    except Exception as exc:
        raise EvaluationError(
            f"Failed to load dequantized state dict: {exc}"
        ) from exc

    model.to(device)

    # Build or validate LUTs.
    if base_bytes_lut is None:
        # Infer vocab size from the embedding weight or from the model attribute.
        vocab_size: int = getattr(model, "vocab_size", None) or _infer_vocab_size(model)
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = (
            _make_uniform_byte_luts(vocab_size, device)
        )
    else:
        if has_leading_space_lut is None or is_boundary_token_lut is None:
            raise EvaluationError(
                "base_bytes_lut was provided but has_leading_space_lut or "
                "is_boundary_token_lut is None — provide all three or none."
            )
        base_bytes_lut = base_bytes_lut.to(device)
        has_leading_space_lut = has_leading_space_lut.to(device)
        is_boundary_token_lut = is_boundary_token_lut.to(device)

    logger.info("evaluate_int6_bpb: running BPB evaluation pass")

    try:
        bpb = _compute_bpb(
            model=model,
            val_tokens=val_tokens,
            seq_len=seq_len,
            batch_tokens=batch_tokens,
            device=device,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
    except EvaluationError:
        raise
    except Exception as exc:
        raise EvaluationError(
            f"Evaluation forward pass failed: {exc}"
        ) from exc

    logger.info("evaluate_int6_bpb: int6_bpb=%.6f", bpb)
    return bpb


def _infer_vocab_size(model: torch.nn.Module) -> int:
    """Try to infer vocab_size from the model's embedding or output head weight."""
    for name, param in model.named_parameters():
        if "embed" in name.lower() or "wte" in name.lower():
            if param.ndim == 2:
                return param.shape[0]
        if ("lm_head" in name.lower() or "output" in name.lower()) and param.ndim == 2:
            return param.shape[0]
    # Last resort: largest first-dimension across all 2-D params.
    max_dim0 = max(
        (p.shape[0] for p in model.parameters() if p.ndim == 2),
        default=50_257,
    )
    logger.warning(
        "_infer_vocab_size: could not find embedding — using max first-dim %d as vocab_size",
        max_dim0,
    )
    return max_dim0
