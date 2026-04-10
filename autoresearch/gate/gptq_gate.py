"""GPTQ gate CLI entry point.

Pipeline:
  1. Load fp32 checkpoint  (--checkpoint)
  2. Load validation tokens and optionally calibration tokens
  3. Measure fp32 BPB (from --parent-results JSON or via a live eval pass)
  4. Quantize to int6 via GPTQQuantizer
  5. LZMA-compress the serialized quantized model
  6. Evaluate int6 roundtrip BPB
  7. Run RejectionCriteria checks
  8. Write GateResult JSON to --output

Usage:
    python -m autoresearch.gate.gptq_gate \\
        --checkpoint /workspace/experiments/exp-42/checkpoint/final_model.pt \\
        --val-tokens  /data/val_tokens.bin \\
        --output      /workspace/experiments/exp-42/artifacts/gate_result.json \\
        --parent-results /workspace/experiments/exp-42/screen_result.json \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import torch

from autoresearch.gate.types import AutoResearchError
from autoresearch.gate.compress import compress_artifact
from autoresearch.gate.criteria import RejectionCriteria
from autoresearch.gate.evaluate import EvaluationError, evaluate_int6_bpb
from autoresearch.gate.quantize import GPTQQuantizer, QuantizationError
from autoresearch.gate.types import GateResult, GateVerdict, VerdictCode

__all__ = ["run_gate", "main"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SEQ_LEN: int = 1024
_DEFAULT_BATCH_TOKENS: int = 524_288
_DEFAULT_CALIB_TOKENS: int = 131_072  # 128 K tokens for calibration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stderr,
    )


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a ``torch.save``'d checkpoint.

    The checkpoint may be:
    - A ``state_dict`` (dict[str, Tensor]) — we try to pair it with the model
      class available from train_gpt.py.
    - A full model object saved with ``torch.save(model, ...)``.

    Args:
        checkpoint_path: Path to the ``.pt`` file.
        device:          Device to map tensors to.

    Returns:
        A :class:`torch.nn.Module` in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        AutoResearchError: If the checkpoint cannot be loaded or is not a model.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    logger.info("Loading checkpoint from %s", checkpoint_path)
    obj = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    if isinstance(obj, torch.nn.Module):
        model: torch.nn.Module = obj
        model.eval()
        return model

    if isinstance(obj, dict):
        # Attempt to reconstruct the model from train_gpt.py.
        try:
            model = _reconstruct_model_from_state_dict(obj, device)
            model.eval()
            return model
        except Exception as exc:
            raise AutoResearchError(
                f"Checkpoint is a dict but could not reconstruct model: {exc}"
            ) from exc

    raise AutoResearchError(
        f"Unrecognised checkpoint type: {type(obj).__name__} in {checkpoint_path}"
    )


def _reconstruct_model_from_state_dict(
    state_dict: dict, device: torch.device
) -> torch.nn.Module:
    """Try to build a GPT model from train_gpt.py and load *state_dict* into it.

    This is a best-effort reconstruction.  It imports ``train_gpt`` at runtime
    so the gate can be used independently from the training environment.

    Raises:
        ImportError: If train_gpt cannot be imported.
        RuntimeError: If the state dict cannot be loaded.
    """
    import train_gpt  # type: ignore[import]

    # train_gpt exposes a Hyperparameters dataclass and a GPT model class.
    # We infer key dims from the state dict itself.
    hp = _infer_hyperparameters(state_dict, train_gpt)
    model = train_gpt.GPT(hp)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return model


def _infer_hyperparameters(state_dict: dict, train_gpt_module) -> object:  # type: ignore[return]
    """Infer Hyperparameters from state dict keys and tensor shapes.

    Falls back to defaults from train_gpt.Hyperparameters when a dimension
    cannot be determined from the state dict.
    """
    hp_cls = train_gpt_module.Hyperparameters

    # Try to read n_layer from the number of transformer blocks.
    block_keys = [k for k in state_dict if "blocks." in k or "h." in k]
    block_indices = set()
    for k in block_keys:
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p in ("blocks", "h") and i + 1 < len(parts):
                try:
                    block_indices.add(int(parts[i + 1]))
                except ValueError:
                    pass
    n_layer = max(block_indices) + 1 if block_indices else None

    # Try to read d_model from embedding weight shape.
    d_model = None
    for k, v in state_dict.items():
        if ("embed" in k.lower() or "wte" in k.lower()) and hasattr(v, "shape") and v.ndim == 2:
            d_model = int(v.shape[1])
            break

    kwargs = {}
    if n_layer is not None:
        kwargs["n_layer"] = n_layer
    if d_model is not None:
        kwargs["d_model"] = d_model

    try:
        return hp_cls(**kwargs)
    except TypeError:
        # If Hyperparameters doesn't accept those kwargs, just use defaults.
        return hp_cls()


def _load_val_tokens(val_tokens_path: Path, device: torch.device) -> torch.Tensor:
    """Load validation tokens from a binary shard file produced by train_gpt.py.

    The shard format: 256 int32 header words followed by uint16 token ids.
    Falls back to a plain ``torch.load`` if the magic number doesn't match.

    Args:
        val_tokens_path: Path to the validation token shard.
        device:          Device for the returned tensor.

    Returns:
        1-D token tensor (dtype int64).
    """
    import numpy as np  # type: ignore[import]

    header_bytes = 256 * 4  # 256 × int32
    raw = val_tokens_path.read_bytes()

    if len(raw) < header_bytes:
        # Too small for a shard file; try plain torch.load.
        logger.warning(
            "_load_val_tokens: %s too small for shard format, trying torch.load",
            val_tokens_path,
        )
        return torch.load(str(val_tokens_path), map_location=device).to(torch.int64)

    import struct

    magic = struct.unpack_from("<i", raw, 0)[0]
    if magic == 20240520:
        num_tokens = struct.unpack_from("<i", raw, 8)[0]
        tokens_np = np.frombuffer(
            raw, dtype="<u2", count=num_tokens, offset=header_bytes
        ).copy()
        return torch.from_numpy(tokens_np.astype(np.int64)).to(device)

    # Plain torch tensor.
    import io
    return torch.load(io.BytesIO(raw), map_location=device).to(torch.int64)


def _load_parent_fp32_bpb(parent_results_path: Optional[Path]) -> Optional[float]:
    """Parse the fp32 BPB from a parent screen/gate result JSON.

    Looks for keys ``screen_ema_bpb``, ``screen_train_bpb``, ``fp32_bpb``,
    or ``bpb`` in that order.

    Returns:
        The fp32 BPB float, or None if the file is absent or the key is missing.
    """
    if parent_results_path is None or not parent_results_path.exists():
        return None

    try:
        data = json.loads(parent_results_path.read_text())
    except Exception as exc:
        logger.warning(
            "_load_parent_fp32_bpb: could not parse %s: %s",
            parent_results_path,
            exc,
        )
        return None

    for key in ("screen_ema_bpb", "screen_train_bpb", "fp32_bpb", "bpb", "val_bpb"):
        if key in data and data[key] is not None:
            try:
                val = float(data[key])
                logger.info(
                    "_load_parent_fp32_bpb: using %s=%.6f from %s",
                    key,
                    val,
                    parent_results_path,
                )
                return val
            except (TypeError, ValueError):
                pass

    logger.warning(
        "_load_parent_fp32_bpb: no recognised BPB key in %s", parent_results_path
    )
    return None


def _peak_memory_mb(device: torch.device) -> float:
    """Return peak CUDA memory allocated in MiB, or 0 on CPU."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    return 0.0


def _write_gate_result(result: GateResult, output_path: Path) -> None:
    """Serialise *result* to JSON at *output_path*, creating parent dirs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": result.passed,
        "int6_bpb": result.int6_bpb,
        "quant_gap": result.quant_gap,
        "artifact_mb": result.artifact_mb,
        "rejection_reason": result.rejection_reason,
        "gptq_time_s": result.gptq_time_s,
        "calibration_tokens": result.calibration_tokens,
        "peak_memory_mb": result.peak_memory_mb,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("Gate result written to %s", output_path)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def run_gate(
    checkpoint_path: Path,
    output_path: Path,
    val_tokens_path: Optional[Path] = None,
    parent_results_path: Optional[Path] = None,
    device: torch.device = torch.device("cpu"),
    seq_len: int = _DEFAULT_SEQ_LEN,
    batch_tokens: int = _DEFAULT_BATCH_TOKENS,
    calib_tokens: int = _DEFAULT_CALIB_TOKENS,
    criteria: Optional[RejectionCriteria] = None,
    lzma_preset: int = 6,
) -> GateResult:
    """Full gate pipeline: load → quantize → evaluate → verdict → write.

    Args:
        checkpoint_path:    Path to the fp32 model checkpoint (.pt).
        output_path:        Where to write the gate_result.json.
        val_tokens_path:    Validation token shard for BPB evaluation.
        parent_results_path: Screen result JSON containing fp32 BPB.
        device:             Compute device.
        seq_len:            Sequence length for evaluation.
        batch_tokens:       Tokens per evaluation batch.
        calib_tokens:       Number of tokens to use for calibration.
        criteria:           Rejection criteria; uses defaults if None.
        lzma_preset:        LZMA compression preset [0–9].

    Returns:
        The :class:`GateResult` that was written to *output_path*.
    """
    if criteria is None:
        criteria = RejectionCriteria()

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Load checkpoint
    # ------------------------------------------------------------------
    try:
        model = _load_checkpoint(checkpoint_path, device)
    except FileNotFoundError as exc:
        logger.error("Checkpoint not found: %s", exc)
        result = GateResult(
            passed=False,
            int6_bpb=float("nan"),
            quant_gap=float("nan"),
            artifact_mb=float("nan"),
            rejection_reason=str(exc),
            gptq_time_s=time.perf_counter() - t_start,
            calibration_tokens=0,
            peak_memory_mb=0.0,
        )
        _write_gate_result(result, output_path)
        return result

    # ------------------------------------------------------------------
    # 2. Load validation tokens
    # ------------------------------------------------------------------
    val_tokens: Optional[torch.Tensor] = None
    if val_tokens_path is not None and val_tokens_path.exists():
        try:
            val_tokens = _load_val_tokens(val_tokens_path, device)
            logger.info(
                "Loaded %d validation tokens from %s",
                val_tokens.numel(),
                val_tokens_path,
            )
        except Exception as exc:
            logger.warning("Could not load val tokens from %s: %s", val_tokens_path, exc)

    # ------------------------------------------------------------------
    # 3. Get fp32 BPB from parent results (or live eval if tokens available)
    # ------------------------------------------------------------------
    fp32_bpb: Optional[float] = _load_parent_fp32_bpb(parent_results_path)

    if fp32_bpb is None and val_tokens is not None:
        logger.info("No parent fp32_bpb found — running fp32 eval pass")
        try:
            from autoresearch.gate.evaluate import _compute_bpb, _make_uniform_byte_luts

            vocab_size = getattr(model, "vocab_size", None) or 50_257
            base_lut, hs_lut, ib_lut = _make_uniform_byte_luts(vocab_size, device)
            fp32_bpb = _compute_bpb(
                model=model,
                val_tokens=val_tokens,
                seq_len=seq_len,
                batch_tokens=batch_tokens,
                device=device,
                base_bytes_lut=base_lut,
                has_leading_space_lut=hs_lut,
                is_boundary_token_lut=ib_lut,
            )
            logger.info("fp32_bpb (live eval) = %.6f", fp32_bpb)
        except Exception as exc:
            logger.warning("fp32 live eval failed: %s", exc)

    if fp32_bpb is None:
        # Cannot evaluate without a baseline — fail gracefully.
        reason = (
            "No fp32 BPB available: --parent-results not provided or missing BPB key, "
            "and val tokens unavailable for live eval"
        )
        logger.error(reason)
        result = GateResult(
            passed=False,
            int6_bpb=float("nan"),
            quant_gap=float("nan"),
            artifact_mb=float("nan"),
            rejection_reason=reason,
            gptq_time_s=time.perf_counter() - t_start,
            calibration_tokens=0,
            peak_memory_mb=_peak_memory_mb(device),
        )
        _write_gate_result(result, output_path)
        return result

    # ------------------------------------------------------------------
    # 4. Quantize to int6
    # ------------------------------------------------------------------
    t_quant_start = time.perf_counter()

    calib_data: Optional[torch.Tensor] = None
    if val_tokens is not None and calib_tokens > 0:
        calib_data = val_tokens[:calib_tokens].cpu()

    try:
        quantizer = GPTQQuantizer(device=device, seq_len=seq_len)
        quantized_model = quantizer.quantize(model, calib_data)
    except (QuantizationError, RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        verdict_code = (
            VerdictCode.FAIL_OOM
            if "out of memory" in str(exc).lower()
            else VerdictCode.FAIL_UNKNOWN
        )
        reason = f"Quantization failed ({verdict_code.value}): {exc}"
        logger.error(reason)
        result = GateResult(
            passed=False,
            int6_bpb=float("nan"),
            quant_gap=float("nan"),
            artifact_mb=float("nan"),
            rejection_reason=reason,
            gptq_time_s=time.perf_counter() - t_start,
            calibration_tokens=0,
            peak_memory_mb=_peak_memory_mb(device),
        )
        _write_gate_result(result, output_path)
        return result

    gptq_time_s = time.perf_counter() - t_quant_start
    logger.info("Quantization complete in %.1f s", gptq_time_s)

    # ------------------------------------------------------------------
    # 5. LZMA compress
    # ------------------------------------------------------------------
    compressed = compress_artifact(quantized_model.serialized, preset=lzma_preset)
    artifact_mb = compressed.compressed_mb
    logger.info(
        "LZMA compressed artifact: %.3f MB (ratio=%.2f×)",
        artifact_mb,
        compressed.ratio,
    )

    # ------------------------------------------------------------------
    # 6. Evaluate int6 roundtrip BPB
    # ------------------------------------------------------------------
    int6_bpb: float = float("nan")

    if val_tokens is not None:
        try:
            int6_bpb = evaluate_int6_bpb(
                model=model,
                quantized_model=quantized_model,
                val_tokens=val_tokens,
                device=device,
                seq_len=seq_len,
                batch_tokens=batch_tokens,
            )
            logger.info("int6_bpb = %.6f", int6_bpb)
        except (EvaluationError, RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            verdict_code = (
                VerdictCode.FAIL_OOM
                if "out of memory" in str(exc).lower()
                else VerdictCode.FAIL_CORRUPT
            )
            reason = f"int6 evaluation failed ({verdict_code.value}): {exc}"
            logger.error(reason)
            result = GateResult(
                passed=False,
                int6_bpb=float("nan"),
                quant_gap=float("nan"),
                artifact_mb=artifact_mb,
                rejection_reason=reason,
                gptq_time_s=time.perf_counter() - t_start,
                calibration_tokens=quantized_model.calibration_tokens,
                peak_memory_mb=_peak_memory_mb(device),
            )
            _write_gate_result(result, output_path)
            return result
    else:
        logger.warning(
            "No validation tokens — skipping int6 BPB eval; "
            "criteria checks will use NaN for int6_bpb"
        )

    # ------------------------------------------------------------------
    # 7. Rejection criteria
    # ------------------------------------------------------------------
    quant_gap = int6_bpb - fp32_bpb if not (
        float("nan") in (int6_bpb, fp32_bpb)  # type: ignore[operator]
    ) else float("nan")

    verdict: GateVerdict
    if float("nan") in (int6_bpb,):  # type: ignore[operator]
        verdict = GateVerdict(
            passed=False,
            code=VerdictCode.FAIL_CORRUPT,
            reason="int6_bpb is NaN — evaluation was skipped or failed",
        )
    else:
        verdict = criteria.evaluate(
            int6_bpb=int6_bpb,
            fp32_bpb=fp32_bpb,
            artifact_mb=artifact_mb,
        )

    # ------------------------------------------------------------------
    # 8. Write result
    # ------------------------------------------------------------------
    result = GateResult(
        passed=verdict.passed,
        int6_bpb=int6_bpb,
        quant_gap=quant_gap,
        artifact_mb=artifact_mb,
        rejection_reason=verdict.reason,
        gptq_time_s=time.perf_counter() - t_start,
        calibration_tokens=quantized_model.calibration_tokens,
        peak_memory_mb=_peak_memory_mb(device),
    )

    _write_gate_result(result, output_path)

    verdict_str = "PASS" if result.passed else f"FAIL ({verdict.code.value})"
    logger.info(
        "Gate verdict: %s | int6_bpb=%.6f quant_gap=%.6f artifact_mb=%.3f",
        verdict_str,
        int6_bpb,
        quant_gap,
        artifact_mb,
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gptq_gate",
        description=(
            "GPTQ gate: quantize a checkpoint to int6+LZMA and check against "
            "rejection criteria.  Writes a gate_result.json to --output."
        ),
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to fp32 model checkpoint (.pt file from train_gpt.py).",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the gate_result.json.",
    )
    p.add_argument(
        "--val-tokens",
        type=Path,
        default=None,
        metavar="PATH",
        help="Validation token shard (.bin) for BPB evaluation.",
    )
    p.add_argument(
        "--parent-results",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "JSON file from the screen stage containing the fp32 BPB "
            "(e.g. screen_result.json with key 'screen_ema_bpb').  "
            "If absent, a live fp32 eval pass is run."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="PyTorch device string (default: cuda:0 if available, else cpu).",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=_DEFAULT_SEQ_LEN,
        help=f"Sequence length for evaluation (default: {_DEFAULT_SEQ_LEN}).",
    )
    p.add_argument(
        "--batch-tokens",
        type=int,
        default=_DEFAULT_BATCH_TOKENS,
        help=f"Tokens per evaluation batch (default: {_DEFAULT_BATCH_TOKENS}).",
    )
    p.add_argument(
        "--calib-tokens",
        type=int,
        default=_DEFAULT_CALIB_TOKENS,
        help=f"Calibration token count (default: {_DEFAULT_CALIB_TOKENS}; 0 to disable).",
    )
    p.add_argument(
        "--max-quant-gap-ratio",
        type=float,
        default=2.0,
        help="RejectionCriteria.max_quant_gap_ratio (default: 2.0).",
    )
    p.add_argument(
        "--max-artifact-mb",
        type=float,
        default=15.5,
        help="RejectionCriteria.max_artifact_mb in MB (default: 15.5).",
    )
    p.add_argument(
        "--max-bpb-regression",
        type=float,
        default=0.05,
        help="RejectionCriteria.max_bpb_regression (default: 0.05).",
    )
    p.add_argument(
        "--lzma-preset",
        type=int,
        default=6,
        choices=range(10),
        metavar="[0-9]",
        help="LZMA compression preset (default: 6).",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.  Returns 0 on gate PASS, 1 on FAIL, 2 on unhandled error."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.log_level)

    device = torch.device(args.device)
    criteria = RejectionCriteria(
        max_quant_gap_ratio=args.max_quant_gap_ratio,
        max_artifact_mb=args.max_artifact_mb,
        max_bpb_regression=args.max_bpb_regression,
    )

    try:
        result = run_gate(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            val_tokens_path=args.val_tokens,
            parent_results_path=args.parent_results,
            device=device,
            seq_len=args.seq_len,
            batch_tokens=args.batch_tokens,
            calib_tokens=args.calib_tokens,
            criteria=criteria,
            lzma_preset=args.lzma_preset,
        )
    except Exception:
        logger.error("Unhandled exception in run_gate:\n%s", traceback.format_exc())
        return 2

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
