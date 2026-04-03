#!/usr/bin/env bash
set -euo pipefail
#
# Full experiment runner: SOTA baseline at full budget, then all ablations at 180s.
#
# Usage:
#   ./scripts/run_all.sh              # auto-detect GPUs
#   NGPUS=4 ./scripts/run_all.sh      # explicit GPU count
#   DRY_RUN=1 ./scripts/run_all.sh    # print commands without running
#
# The SOTA baseline runs at full budget (1200s on 4 GPUs, 600s on 8 GPUs).
# All ablations run at 180s training + 10 min eval timeout.
#
# Launch the dashboard in a separate terminal:
#   python3 scripts/dashboard.py
#

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

NGPUS="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NGPUS="${NGPUS:-8}"
[ "$NGPUS" -eq 0 ] && NGPUS=8
DRY_RUN="${DRY_RUN:-0}"

# Scale baseline wallclock for GPU count: 600s * (8 / NGPUS)
BASELINE_WALLCLOCK=$(( 600 * 8 / NGPUS ))
ABLATION_WALLCLOCK=180
EVAL_TIMEOUT=600  # 10 minutes

echo ""
echo "================================================================"
echo "  Parameter Golf Full Experiment Run"
echo "  GPUs: $NGPUS"
echo "  Baseline: ${BASELINE_WALLCLOCK}s training"
echo "  Ablations: ${ABLATION_WALLCLOCK}s training"
echo "  Eval timeout: ${EVAL_TIMEOUT}s (10 min)"
echo "  Mode: $([ "$DRY_RUN" = "1" ] && echo "DRY RUN" || echo "LIVE")"
echo "================================================================"
echo ""
echo "TIP: Launch dashboard in another terminal:"
echo "  python3 scripts/dashboard.py"
echo ""

export NGPUS
export EVAL_TIMEOUT
export DRY_RUN

# ── Phase 1: SOTA Baseline (full budget, full eval) ──
echo "========================================"
echo "  PHASE 1: SOTA Baseline"
echo "  Wallclock: ${BASELINE_WALLCLOCK}s"
echo "========================================"
MAX_WALLCLOCK_SECONDS=$BASELINE_WALLCLOCK FULL_EVAL=1 \
    "$REPO_ROOT/scripts/run_ablations.sh" 0 || true

# ── Phase 2: Architecture Ablations (180s, no sliding window) ──
echo "========================================"
echo "  PHASE 2: Architecture (A1-A4)"
echo "  Wallclock: ${ABLATION_WALLCLOCK}s"
echo "========================================"
MAX_WALLCLOCK_SECONDS=$ABLATION_WALLCLOCK \
    "$REPO_ROOT/scripts/run_ablations.sh" A || true

# ── Phase 3: Training Dynamics (180s, no sliding window) ──
echo "========================================"
echo "  PHASE 3: Training Dynamics (B1-B9)"
echo "  Wallclock: ${ABLATION_WALLCLOCK}s"
echo "========================================"
MAX_WALLCLOCK_SECONDS=$ABLATION_WALLCLOCK \
    "$REPO_ROOT/scripts/run_ablations.sh" B || true

# ── Phase 4: Eval Stride (180s training, with sliding window) ──
echo "========================================"
echo "  PHASE 4: Eval Stride (C1-C2)"
echo "  Wallclock: ${ABLATION_WALLCLOCK}s"
echo "========================================"
MAX_WALLCLOCK_SECONDS=$ABLATION_WALLCLOCK \
    "$REPO_ROOT/scripts/run_ablations.sh" C || true

# ── Phase 5: Combinations (180s, no sliding window) ──
echo "========================================"
echo "  PHASE 5: Combinations (D1-D5)"
echo "  Wallclock: ${ABLATION_WALLCLOCK}s"
echo "========================================"
MAX_WALLCLOCK_SECONDS=$ABLATION_WALLCLOCK \
    "$REPO_ROOT/scripts/run_ablations.sh" D || true

echo ""
echo "================================================================"
echo "  ALL PHASES COMPLETE"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""

# Print final summary
MAX_WALLCLOCK_SECONDS=$ABLATION_WALLCLOCK "$REPO_ROOT/scripts/run_ablations.sh" --summary 2>/dev/null || true
