#!/usr/bin/env bash
# ============================================================================
# Parameter Golf Ablation Runner
# ============================================================================
#
# Runs experiments using the SOTA train_gpt.py (PR #1019) with env var overrides.
# No code modifications — every experiment is the exact same script.
#
# Usage:
#   ./scripts/run_ablations.sh                    # run all experiments
#   ./scripts/run_ablations.sh 0                  # SOTA baseline only
#   ./scripts/run_ablations.sh A1 A2 B1           # specific experiments
#   ./scripts/run_ablations.sh A                  # all architecture experiments
#   DRY_RUN=1 ./scripts/run_ablations.sh          # print commands without running
#   NGPUS=1 ./scripts/run_ablations.sh 0          # single GPU (for testing)
#   MAX_WALLCLOCK_SECONDS=90 ./scripts/run_ablations.sh 0  # quick 90s run
#
# Environment:
#   NGPUS               Number of GPUs (default: auto-detect, fallback 8)
#   MAX_WALLCLOCK_SECONDS  Training time budget (default: 600)
#   DRY_RUN             If 1, print commands without executing
#   LOG_DIR             Where to save logs (default: experiment_logs/ablations)
#   SOTA_SCRIPT         Path to SOTA train_gpt.py (default: auto-detect)
#
# ============================================================================

set -euo pipefail

# --- Configuration ---
NGPUS="${NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NGPUS="${NGPUS:-8}"
[ "$NGPUS" -eq 0 ] && NGPUS=8

WALLCLOCK="${MAX_WALLCLOCK_SECONDS:-600}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-experiment_logs/ablations}"
SOTA_SCRIPT="${SOTA_SCRIPT:-records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py}"

# Resolve paths relative to repo root
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [ ! -f "$SOTA_SCRIPT" ]; then
    echo "ERROR: SOTA script not found at $SOTA_SCRIPT"
    exit 1
fi

mkdir -p "$LOG_DIR"

# --- Experiment definitions ---
# Each experiment: NAME|ENV_OVERRIDES|DESCRIPTION
# ENV_OVERRIDES are space-separated KEY=VALUE pairs
# The SOTA defaults (from the script) are the baseline — we only list overrides.

EXPERIMENTS=(
    # === Experiment 0: SOTA Baseline (reproduce PR #1019 exactly) ===
    "0_sota_baseline||SOTA baseline — reproduce PR #1019 (1.1147 BPB)"

    # === Category A: Architecture ===
    "A1_9L_mlp3.5x|NUM_LAYERS=9 MLP_MULT=3.5 XSA_LAST_N=9 VE_LAYERS=7,8|9L + MLP 3.5x (PR #1105 width, fewer layers)"
    "A2_8L_mlp4x|NUM_LAYERS=8 MLP_MULT=4.0 XSA_LAST_N=8 VE_LAYERS=6,7|8L + MLP 4x (prior best combo)"
    "A3_7L_mlp4x|NUM_LAYERS=7 MLP_MULT=4.0 XSA_LAST_N=7 VE_LAYERS=5,6|7L + MLP 4x (scaled test winner)"
    "A4_11L_mlp3.5x|NUM_LAYERS=11 MLP_MULT=3.5|11L + MLP 3.5x (PR #1105 config, same depth)"

    # === Category B: Training dynamics ===
    "B1_muon_lr_0.03|MATRIX_LR=0.03 SCALAR_LR=0.03|Muon LR 0.03 (autoresearch optimal)"
    "B2_warmdown_4500|WARMDOWN_ITERS=4500|Longer warmdown (smoother quant transition)"
    "B3_warmdown_5000|WARMDOWN_ITERS=5000|Even longer warmdown"
    "B4_bigram_3072x112|BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112|BigramHash 3072x112 (match submission.json)"
    "B5_muon_wd_0.06|MUON_WD=0.06 ADAM_WD=0.06|Higher weight decay"
    "B6_head_lr_0.01|HEAD_LR=0.01|Higher unembedding LR (autoresearch: 0.008 > 0.004)"

    # === Category C: Eval-time (no training change) ===
    "C1_stride_32|EVAL_STRIDE=32|Sliding window stride 32 (more overlap)"
    "C2_stride_16|EVAL_STRIDE=16|Sliding window stride 16 (max overlap)"

    # === Category D: Combinations (run after A/B/C show signal) ===
    "D1_7L_mlp4x_muon03|NUM_LAYERS=7 MLP_MULT=4.0 XSA_LAST_N=7 VE_LAYERS=5,6 MATRIX_LR=0.03 SCALAR_LR=0.03|7L/4x + Muon 0.03"
    "D2_9L_mlp3.5x_muon03|NUM_LAYERS=9 MLP_MULT=3.5 XSA_LAST_N=9 VE_LAYERS=7,8 MATRIX_LR=0.03 SCALAR_LR=0.03|9L/3.5x + Muon 0.03"
    "D3_11L_mlp3.5x_bigram3072|NUM_LAYERS=11 MLP_MULT=3.5 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112|11L/3.5x + bigger bigram"
    "D4_best_combo_stride16|NUM_LAYERS=7 MLP_MULT=4.0 XSA_LAST_N=7 VE_LAYERS=5,6 MATRIX_LR=0.03 SCALAR_LR=0.03 EVAL_STRIDE=16|D1 + stride 16 eval"
)

# --- Helper functions ---

run_experiment() {
    local spec="$1"
    local name envs desc
    IFS='|' read -r name envs desc <<< "$spec"

    local logfile="$LOG_DIR/${name}.log"

    echo "============================================================"
    echo "  Experiment: $name"
    echo "  Description: $desc"
    echo "  Overrides: ${envs:-<none (SOTA defaults)>}"
    echo "  Log: $logfile"
    echo "============================================================"

    # Build env var string
    local env_cmd=""
    env_cmd+="RUN_ID=$name "
    env_cmd+="MAX_WALLCLOCK_SECONDS=$WALLCLOCK "
    env_cmd+="VAL_LOSS_EVERY=200 "
    env_cmd+="TRAIN_LOG_EVERY=100 "

    if [ -n "$envs" ]; then
        env_cmd+="$envs "
    fi

    local cmd="env $env_cmd torchrun --standalone --nproc_per_node=$NGPUS $SOTA_SCRIPT"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY RUN] $cmd"
        echo ""
        return 0
    fi

    echo "Running: $cmd"
    echo "Started: $(date -Iseconds)"
    echo ""

    # Run and tee to log
    eval "$cmd" 2>&1 | tee "$logfile"
    local exit_code=${PIPESTATUS[0]}

    echo ""
    echo "Finished: $(date -Iseconds) (exit code: $exit_code)"

    # Extract key metrics from log
    if [ -f "$logfile" ]; then
        echo "--- Key metrics ---"
        grep -E "final_int6_sliding_window_exact|ttt_eval_exact|model_params|step_avg" "$logfile" | tail -5
        echo ""
    fi

    return $exit_code
}

match_experiment() {
    local filter="$1"
    local name
    IFS='|' read -r name _ _ <<< "$2"

    # Exact match (e.g., "0", "A1", "D3")
    if [ "$name" = "${filter}_sota_baseline" ] || [[ "$name" == "${filter}_"* ]] || [ "$name" = "$filter" ]; then
        return 0
    fi

    # Category match (e.g., "A" matches A1, A2, A3, A4)
    if [[ "$name" == "${filter}"[0-9]* ]]; then
        return 0
    fi

    return 1
}

# --- Main ---

echo ""
echo "Parameter Golf Ablation Runner"
echo "  SOTA script: $SOTA_SCRIPT"
echo "  GPUs: $NGPUS"
echo "  Wallclock: ${WALLCLOCK}s"
echo "  Log dir: $LOG_DIR"
echo "  Mode: $([ "$DRY_RUN" = "1" ] && echo "DRY RUN" || echo "LIVE")"
echo ""

# Determine which experiments to run
filters=("$@")
if [ ${#filters[@]} -eq 0 ]; then
    # No args = run all
    for spec in "${EXPERIMENTS[@]}"; do
        run_experiment "$spec"
    done
else
    for spec in "${EXPERIMENTS[@]}"; do
        for filter in "${filters[@]}"; do
            if match_experiment "$filter" "$spec"; then
                run_experiment "$spec"
                break
            fi
        done
    done
fi

# --- Summary ---
echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
echo ""
printf "%-30s %-15s %-15s %-10s\n" "Experiment" "Training BPB" "Final BPB" "Steps"
printf "%-30s %-15s %-15s %-10s\n" "----------" "------------" "---------" "-----"

for logfile in "$LOG_DIR"/*.log; do
    [ -f "$logfile" ] || continue
    name="$(basename "$logfile" .log)"

    # Extract last training BPB
    train_bpb=$(grep "val_bpb:" "$logfile" 2>/dev/null | tail -1 | sed 's/.*val_bpb:\([0-9.]*\).*/\1/' || echo "N/A")

    # Extract final sliding window BPB
    final_bpb=$(grep "final_int6_sliding_window_exact" "$logfile" 2>/dev/null | sed 's/.*val_bpb:\([0-9.]*\).*/\1/' || echo "N/A")

    # Extract steps
    steps=$(grep "^step:" "$logfile" 2>/dev/null | tail -1 | sed 's/step:\([0-9]*\).*/\1/' || echo "N/A")

    printf "%-30s %-15s %-15s %-10s\n" "$name" "$train_bpb" "$final_bpb" "$steps"
done

echo ""
echo "Done. Logs saved to $LOG_DIR/"
