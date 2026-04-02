#!/usr/bin/env bash
set -euo pipefail
# Re-run Phase 3 experiments that were contaminated by GPU contention.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_ROOT/experiment_logs_fullscale"
VENV="$REPO_ROOT/.venv/bin/activate"
[[ -f "$VENV" ]] && source "$VENV"

export DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_sp1024/"
export TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/hf_home
export VOCAB_SIZE=1024
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export TRAIN_SEQ_LEN=1024
export TRAIN_BATCH_TOKENS=524288
export BIGRAM_VOCAB_SIZE=3072
export BIGRAM_DIM=112
export TARGET_MB=15.9
export MAX_WALLCLOCK_SECONDS=90
export VAL_LOSS_EVERY=200
export WARMDOWN_ITERS=4000
export WARMUP_STEPS=20

NPROC=8
MASTER_PORT=29500

run_one() {
    local name="$1"
    shift
    local env_vars=("$@")
    local log="$LOG_DIR/${name}.log"

    echo "━━━ $name | started: $(date '+%H:%M:%S') ━━━"

    # Kill any leftover workers first
    pkill -9 -f "train_sota_exp.py" 2>/dev/null || true
    sleep 3

    (
        cd "$REPO_ROOT"
        for ev in "${env_vars[@]+"${env_vars[@]}"}"; do
            export "$ev"
        done
        export RUN_ID="$name"
        timeout 3600 torchrun \
            --standalone \
            --master_port=$MASTER_PORT \
            --nproc_per_node=$NPROC \
            train_sota_exp.py
    ) >"$log" 2>&1
    local rc=$?

    # Cleanup
    pkill -f "train_sota_exp.py" 2>/dev/null || true
    sleep 2
    pkill -9 -f "train_sota_exp.py" 2>/dev/null || true
    sleep 1

    local bpb
    bpb=$(grep -aoP 'val_bpb:\K[\d.]+' "$log" | tail -1 || echo "N/A")
    echo "  [$name] done (exit $rc) | final val_bpb: $bpb | ended: $(date '+%H:%M:%S')"
}

# Wait for any running experiments to finish
echo "Waiting for current experiments to finish..."
while pgrep -f "train_sota_exp.py" > /dev/null 2>&1; do
    sleep 30
done
echo "GPUs free. Starting re-runs at $(date '+%H:%M:%S')"
sleep 5

# Re-run contaminated experiments
run_one "p3_11L_swiglu_trigram" \
    "TTT_MODE=none" "NUM_LAYERS=11" "MLP_MULT=3" \
    "MLP_ACTIVATION=swiglu" "TRIGRAM=1"

run_one "p3_9L_mlp4x_leaky_trigram" \
    "TTT_MODE=none" "NUM_LAYERS=9" "MLP_MULT=4" \
    "MLP_ACTIVATION=leaky_relu2" "TRIGRAM=1"

echo ""
echo "Re-runs complete at $(date '+%H:%M:%S')"
