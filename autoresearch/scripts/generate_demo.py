"""Checkpoint token-streaming tool for qualitative inspection.

Loads a training checkpoint and generates tokens for human review.
Integrated from JianYan11/parameter-golf for the INSPECT stage (§3.1).

CLI::

    python3 -m autoresearch.scripts.generate_demo \\
        --checkpoint PATH --tokenizer PATH \\
        --prompt "The most important thing" \\
        --max-new-tokens 128 --temperature 0.9 --plain

In ``--plain`` mode, outputs generated text to stdout (machine-parseable).
Without ``--plain``, enters an interactive prompt loop.
"""

from __future__ import annotations

__all__ = ["load_checkpoint", "generate_tokens"]

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]


def load_checkpoint(
    path: Path,
    device: str = "cpu",
    model_class: Optional[Any] = None,
) -> Tuple[Any, Optional[Any]]:
    """Load a model checkpoint from disk.

    Handles common checkpoint formats:
    - Dict with ``model_state_dict`` key (training checkpoint)
    - Dict with ``model`` key
    - Raw state dict
    - DDP-wrapped state dicts with ``module.`` prefixes

    Supports both ``.pt`` and ``.ptz`` (compressed) files.

    Args:
        path: Path to the checkpoint file (``.pt`` or ``.ptz``).
        device: Device to load tensors onto.
        model_class: Optional model class to instantiate. If *None*,
            returns the raw state dict as the "model".

    Returns:
        Tuple of (model_or_state_dict, None). The second element is
        reserved for a tokenizer when bundled in the checkpoint.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        RuntimeError: If torch is not available or loading fails.
    """
    if torch is None:
        raise RuntimeError("torch is required for generate_demo but is not installed")

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info("Loading checkpoint from %s onto %s", path, device)

    suffix = path.suffix.lower()
    if suffix == ".ptz":
        # Compressed checkpoint: read bytes then decompress
        import zlib

        compressed = path.read_bytes()
        raw = zlib.decompress(compressed)
        import io

        checkpoint = torch.load(
            io.BytesIO(raw), map_location=device, weights_only=False
        )
    else:
        checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Extract state dict from various checkpoint formats
    state_dict: Dict[str, Any]
    tokenizer = None

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            tokenizer = checkpoint.get("tokenizer")
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            tokenizer = checkpoint.get("tokenizer")
        else:
            state_dict = checkpoint
    else:
        # Assume it's a full model object
        return checkpoint, None

    # Strip DDP 'module.' prefixes
    cleaned: Dict[str, Any] = {}
    for key, value in state_dict.items():
        clean_key = key.removeprefix("module.")
        cleaned[clean_key] = value

    if model_class is not None:
        model = model_class()
        model.load_state_dict(cleaned, strict=False)
        model.to(device)
        model.eval()
        return model, tokenizer

    logger.info(
        "Loaded state dict with %d parameters (no model class provided)",
        len(cleaned),
    )
    return cleaned, tokenizer


def _load_tokenizer(path: Path) -> Any:
    """Load a SentencePiece or tiktoken tokenizer from *path*.

    Returns an object with ``encode(str) -> list[int]`` and
    ``decode(list[int]) -> str`` methods.
    """
    suffix = path.suffix.lower()

    if suffix == ".model":
        # SentencePiece model
        try:
            import sentencepiece as spm
        except ImportError:
            raise RuntimeError(
                "sentencepiece is required for .model tokenizers. "
                "Install with: pip install sentencepiece"
            )
        sp = spm.SentencePieceProcessor()
        sp.Load(str(path))
        return sp

    if suffix == ".json":
        # Try tiktoken or HF tokenizer
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(str(path.parent))
        except ImportError:
            pass
        raise RuntimeError(
            f"Cannot load tokenizer from {path}. "
            "Install transformers: pip install transformers"
        )

    raise RuntimeError(f"Unsupported tokenizer format: {suffix}")


def generate_tokens(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
    device: str = "cpu",
) -> str:
    """Generate tokens from a model given a prompt.

    Args:
        model: A PyTorch model with a forward pass that accepts input_ids
            and returns logits, OR a raw state dict (in which case a
            simple embedding-based generation is attempted).
        tokenizer: Object with ``encode`` and ``decode`` methods.
        prompt: Text prompt to condition generation on.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        device: Device for inference.

    Returns:
        The generated text (prompt + completion).
    """
    if torch is None:
        raise RuntimeError("torch is required for generate_demo but is not installed")

    # Encode prompt
    token_ids: List[int] = tokenizer.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    # If model is a state dict, we can't do forward passes
    if isinstance(model, dict):
        logger.warning(
            "Model is a raw state dict — cannot generate tokens without "
            "a model class. Returning prompt only."
        )
        return prompt

    model.eval()
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(generated)

            # Handle different output formats
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs

            # Get logits for the last token
            next_logits = logits[:, -1, :]

            if temperature <= 0:
                # Greedy
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            # Stop on EOS if tokenizer has one
            eos_id = getattr(tokenizer, "eos_id", None)
            if eos_id is None:
                eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None and next_token.item() == eos_id:
                break

    output_ids = generated[0].tolist()
    return tokenizer.decode(output_ids)


def _interactive_loop(
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    device: str,
) -> None:
    """Run an interactive prompt loop for token exploration."""
    print("=" * 60)
    print("  AutoResearch v2 — Checkpoint Inspector")
    print("  Type a prompt and press Enter to generate.")
    print("  Type 'quit' or Ctrl-C to exit.")
    print("=" * 60)
    print()

    while True:
        try:
            prompt = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if prompt.strip().lower() in ("quit", "exit", "q"):
            break

        if not prompt.strip():
            continue

        result = generate_tokens(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )
        print()
        print(result)
        print()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate tokens from a training checkpoint for qualitative inspection.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt or .ptz).",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="Path to tokenizer file (.model for SentencePiece, .json for HF).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The most important thing about",
        help="Text prompt for generation (default: 'The most important thing about').",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate (default: 128).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature (default: 0.9).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (default: cpu).",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Plain output mode: print generated text to stdout and exit.",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv if argv is not None else sys.argv[1:])

    if torch is None:
        logger.error("torch is required but not installed. Install with: pip install torch")
        sys.exit(1)

    try:
        model, bundled_tokenizer = load_checkpoint(args.checkpoint, device=args.device)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Failed to load checkpoint: %s", exc)
        sys.exit(1)

    try:
        tokenizer = _load_tokenizer(args.tokenizer)
    except RuntimeError as exc:
        logger.error("Failed to load tokenizer: %s", exc)
        sys.exit(1)

    if args.plain:
        result = generate_tokens(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device,
        )
        print(result)
    else:
        _interactive_loop(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device,
        )


if __name__ == "__main__":
    main()
