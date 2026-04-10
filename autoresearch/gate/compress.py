"""LZMA compression wrapper for gate artifacts.

train_gpt.py uses zlib level-9 for the int8 artifact.  The gate pipeline
upgrades to LZMA for better compression at the cost of slightly more CPU time.
"""

from __future__ import annotations

import lzma
import logging
from dataclasses import dataclass

__all__ = ["CompressedArtifact", "compress_artifact"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompressedArtifact:
    """Immutable result of a compression operation.

    Attributes:
        compressed_bytes: Raw LZMA-compressed payload (ready to write to disk).
        original_size:    Size of the input data in bytes.
        compressed_size:  Size of the compressed output in bytes.
        ratio:            original_size / compressed_size  (>1.0 means compression
                          helped; 1.0 means no change).
    """

    compressed_bytes: bytes
    original_size: int
    compressed_size: int
    ratio: float

    @property
    def compressed_mb(self) -> float:
        """Compressed size expressed in mebibytes (1 MiB = 1 048 576 bytes)."""
        return self.compressed_size / (1024 * 1024)


def compress_artifact(data: bytes, preset: int = 6) -> CompressedArtifact:
    """Compress *data* with LZMA and return a :class:`CompressedArtifact`.

    The LZMA preset controls the compression level:
    - 0 is fastest / largest output
    - 6 (default) balances speed and size
    - 9 is slowest / smallest output

    Args:
        data:   Raw bytes to compress (e.g. the output of ``torch.save`` to a
                ``BytesIO`` buffer).
        preset: LZMA compression preset in [0, 9].  Passed directly to
                :func:`lzma.compress`.

    Returns:
        A frozen :class:`CompressedArtifact` with the compressed payload and
        size statistics.

    Raises:
        ValueError: If *preset* is outside [0, 9].
    """
    if not (0 <= preset <= 9):
        raise ValueError(f"LZMA preset must be in [0, 9], got {preset}")

    original_size = len(data)

    logger.debug(
        "compress_artifact: compressing %d bytes with LZMA preset=%d",
        original_size,
        preset,
    )

    compressed = lzma.compress(data, preset=preset)
    compressed_size = len(compressed)

    ratio = original_size / compressed_size if compressed_size > 0 else 1.0

    logger.debug(
        "compress_artifact: %d → %d bytes (ratio=%.3f, preset=%d)",
        original_size,
        compressed_size,
        ratio,
        preset,
    )

    return CompressedArtifact(
        compressed_bytes=compressed,
        original_size=original_size,
        compressed_size=compressed_size,
        ratio=ratio,
    )
