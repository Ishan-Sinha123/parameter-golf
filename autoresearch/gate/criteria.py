"""RejectionCriteria: pass/fail rules for the GPTQ gate.

Each check method returns None on pass or a human-readable rejection reason string.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from autoresearch.gate.types import GateVerdict, VerdictCode

__all__ = ["RejectionCriteria"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RejectionCriteria:
    """Configurable thresholds for the GPTQ gate.

    Attributes:
        max_quant_gap_ratio: Maximum allowed ratio of quant_gap to fp32_bpb.
            quant_gap = int6_bpb - fp32_bpb; ratio = quant_gap / fp32_bpb.
            Default 2.0 means the gap may not exceed 200% of the baseline BPB.
        max_artifact_mb: Maximum allowed compressed artifact size in megabytes.
            Default 15.5 MB keeps us well under the 16 MB submission cap.
        max_bpb_regression: Maximum allowed absolute BPB increase after
            int6 quantization relative to fp32.  Default 0.05 BPB.
    """

    max_quant_gap_ratio: float = 2.0
    max_artifact_mb: float = 15.5
    max_bpb_regression: float = 0.05

    def check_quant_gap(self, int6_bpb: float, fp32_bpb: float) -> Optional[str]:
        """Return a rejection reason if the quantization gap ratio is too large.

        The *ratio* is quant_gap / fp32_bpb, so a gap that is a large fraction
        of the baseline BPB is penalised proportionally.

        Args:
            int6_bpb: Bits-per-byte after int6 quantization + roundtrip.
            fp32_bpb:  Bits-per-byte of the fp32 checkpoint (parent result).

        Returns:
            None if the check passes, or an explanatory string if it fails.
        """
        if fp32_bpb <= 0.0:
            reason = f"fp32_bpb must be positive, got {fp32_bpb:.6f}"
            logger.warning("check_quant_gap: %s", reason)
            return reason

        quant_gap = int6_bpb - fp32_bpb
        ratio = quant_gap / fp32_bpb

        logger.debug(
            "check_quant_gap: int6_bpb=%.6f fp32_bpb=%.6f gap=%.6f ratio=%.4f limit=%.4f",
            int6_bpb,
            fp32_bpb,
            quant_gap,
            ratio,
            self.max_quant_gap_ratio,
        )

        if ratio > self.max_quant_gap_ratio:
            return (
                f"quant_gap_ratio {ratio:.4f} exceeds max {self.max_quant_gap_ratio:.4f} "
                f"(int6_bpb={int6_bpb:.6f}, fp32_bpb={fp32_bpb:.6f}, gap={quant_gap:.6f})"
            )
        return None

    def check_artifact_size(self, artifact_mb: float) -> Optional[str]:
        """Return a rejection reason if the compressed artifact is too large.

        Args:
            artifact_mb: Size of the LZMA-compressed int6 artifact in megabytes.

        Returns:
            None if the check passes, or an explanatory string if it fails.
        """
        logger.debug(
            "check_artifact_size: artifact_mb=%.3f limit=%.3f",
            artifact_mb,
            self.max_artifact_mb,
        )

        if artifact_mb > self.max_artifact_mb:
            return (
                f"artifact_mb {artifact_mb:.3f} exceeds max {self.max_artifact_mb:.3f}"
            )
        return None

    def check_bpb_regression(self, int6_bpb: float, fp32_bpb: float) -> Optional[str]:
        """Return a rejection reason if absolute BPB regression is too large.

        Unlike check_quant_gap (which is ratio-based), this check measures the
        absolute BPB increase so a model with an already-bad fp32 baseline
        cannot sneak through a large absolute regression.

        Args:
            int6_bpb: Bits-per-byte after int6 quantization + roundtrip.
            fp32_bpb:  Bits-per-byte of the fp32 checkpoint.

        Returns:
            None if the check passes, or an explanatory string if it fails.
        """
        regression = int6_bpb - fp32_bpb

        logger.debug(
            "check_bpb_regression: int6_bpb=%.6f fp32_bpb=%.6f regression=%.6f limit=%.6f",
            int6_bpb,
            fp32_bpb,
            regression,
            self.max_bpb_regression,
        )

        if regression > self.max_bpb_regression:
            return (
                f"bpb_regression {regression:.6f} exceeds max {self.max_bpb_regression:.6f} "
                f"(int6_bpb={int6_bpb:.6f}, fp32_bpb={fp32_bpb:.6f})"
            )
        return None

    def evaluate(
        self,
        int6_bpb: float,
        fp32_bpb: float,
        artifact_mb: float,
    ) -> GateVerdict:
        """Run all rejection checks and return an aggregated GateVerdict.

        Checks are evaluated in priority order; the first failure determines
        the VerdictCode so the most actionable rejection surfaces first.

        Args:
            int6_bpb:    BPB measured after int6 roundtrip.
            fp32_bpb:    BPB of the fp32 checkpoint (parent result).
            artifact_mb: Compressed artifact size in MB.

        Returns:
            GateVerdict with passed=True and code=PASS, or the first failure.
        """
        failures: List[tuple[VerdictCode, str]] = []

        quant_gap_reason = self.check_quant_gap(int6_bpb, fp32_bpb)
        if quant_gap_reason is not None:
            failures.append((VerdictCode.FAIL_QUANT_GAP, quant_gap_reason))

        artifact_reason = self.check_artifact_size(artifact_mb)
        if artifact_reason is not None:
            failures.append((VerdictCode.FAIL_ARTIFACT_SIZE, artifact_reason))

        bpb_reason = self.check_bpb_regression(int6_bpb, fp32_bpb)
        if bpb_reason is not None:
            failures.append((VerdictCode.FAIL_BPB_REGRESSION, bpb_reason))

        if failures:
            first_code, first_reason = failures[0]
            logger.info(
                "GateVerdict FAIL code=%s reason=%s", first_code.value, first_reason
            )
            return GateVerdict(passed=False, code=first_code, reason=first_reason)

        logger.info("GateVerdict PASS")
        return GateVerdict(passed=True, code=VerdictCode.PASS, reason=None)
