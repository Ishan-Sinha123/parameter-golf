#!/usr/bin/env bash
set -euo pipefail
# Fully detached runner for remaining experiments
# Writes to its own log file independent of any parent process

exec > /tmp/remaining_runner2.log 2>&1

REPO_ROOT="/workspace/parameter-golf"
LOG_DIR="$REPO_ROOT/experiment_logs_fullscale"
mkdir -p "$LOG_DIR"

source "$REPO_ROOT/.venv/bin/activate"

export DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_sp1024/"
export TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/hf_home
export VOCAB_SIZE=1024
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3
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

    echo "$(date '+%H:%M:%S') START $name"

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
    ) >"$log" 2>&1 || true

    local final_bpb
    final_bpb=$(grep -oP 'val_bpb:\K[\d.]+' "$log" 2>/dev/null | tail -1 || echo "N/A")
    echo "$(date '+%H:%M:%S') DONE  $name | val_bpb=$final_bpb"
}

# Phase 3 remaining
run_one "p3_11L_swiglu_trigram" \
    "TTT_MODE=none" "NUM_LAYERS=11" "MLP_MULT=3" "MLP_ACTIVATION=swiglu" "TRIGRAM=1"

run_one "p3_9L_mlp4x_leaky_trigram" \
    "TTT_MODE=none" "NUM_LAYERS=9" "MLP_MULT=4" "MLP_ACTIVATION=leaky_relu2" "TRIGRAM=1"

# Phase 4: TTT
run_one "p4_lora_r4"  "TTT_MODE=lora" "TTT_LORA_RANK=4"
run_one "p4_lora_r8"  "TTT_MODE=lora" "TTT_LORA_RANK=8"
run_one "p4_lora_r16" "TTT_MODE=lora" "TTT_LORA_RANK=16"
run_one "p4_lora_r32" "TTT_MODE=lora" "TTT_LORA_RANK=32" "TTT_BATCH_SIZE=32"

run_one "p4_fft_last2" "TTT_MODE=fft2"
run_one "p4_fft_last4" "TTT_MODE=fft4"
run_one "p4_fft_all"   "TTT_MODE=fft_all"

run_one "p4_lora_r16_3step"    "TTT_MODE=lora" "TTT_LORA_RANK=16" "TTT_STEPS=3" "TTT_LORA_LR=0.03"
run_one "p4_lora_r16_chunk128" "TTT_MODE=lora" "TTT_LORA_RANK=16" "TTT_CHUNK_SIZE=128"
run_one "p4_lora_r16_qvk"     "TTT_MODE=lora" "TTT_LORA_RANK=16" "TTT_LORA_TARGETS=qvk"
run_one "p4_bias_ttt"          "TTT_MODE=bias"
run_one "p4_lora_r16_qv_mlp"  "TTT_MODE=lora" "TTT_LORA_RANK=16" "TTT_LORA_TARGETS=qv_mlp"
run_one "p4_lora_r16_qvk_mlp" "TTT_MODE=lora" "TTT_LORA_RANK=16" "TTT_LORA_TARGETS=qvk_mlp"

echo "$(date '+%H:%M:%S') ALL DONE"
