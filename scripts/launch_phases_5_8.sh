#!/usr/bin/env bash
set -euo pipefail
# Wait for the current run_experiments_scaled.sh (phases 1-4) to finish,
# then run bigram hash comparison + phases 5-8.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Waiting for phases 1-4 to finish..."
while pgrep -f "run_experiments_scaled.sh" >/dev/null 2>&1; do
    sleep 10
done

echo "Phases 1-4 complete."

# Run bigram hash experiment (comparison for existing trigram)
echo "━━━ Running p2_bigram_hash..."
(
    cd "$REPO_ROOT"
    export NUM_LAYERS=6 MODEL_DIM=384 MLP_MULT=2 TRAIN_SEQ_LEN=512
    export TRAIN_BATCH_TOKENS=262144 MAX_WALLCLOCK_SECONDS=90
    export VAL_LOSS_EVERY=10 WARMDOWN_ITERS=800 WARMUP_STEPS=10
    export XSA_LAST_N=6 VE_LAYERS="4,5" BIGRAM_VOCAB_SIZE=1024 BIGRAM_DIM=64
    export VOCAB_SIZE=1024 TORCHDYNAMO_DISABLE=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export TTT_MODE=none HASH_MODE=bigram RUN_ID=p2_bigram_hash
    timeout 600 torchrun --standalone --master_port=29500 --nproc_per_node=1 train_alpha.py
) >"$REPO_ROOT/experiment_logs_scaled/p2_bigram_hash.log" 2>&1
echo "    [p2_bigram_hash] done (exit $?)"

# Now launch phases 8, 6, 7, 5
echo "Launching phases 8 6 7 5..."
"$REPO_ROOT/scripts/run_experiments_scaled.sh" 8 6 7 5
