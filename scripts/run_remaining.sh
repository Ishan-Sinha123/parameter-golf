#!/usr/bin/env bash
set -euo pipefail
# Run remaining Phase 3 + all Phase 4 experiments

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_ROOT/experiment_logs_fullscale"
mkdir -p "$LOG_DIR"

VENV="$REPO_ROOT/.venv/bin/activate"
[[ -f "$VENV" ]] && source "$VENV"

# Common env
export DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_sp1024/"
export TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/hf_home

# Full-scale model config (SOTA defaults)
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

# Training budget: 90s
export MAX_WALLCLOCK_SECONDS=90
export VAL_LOSS_EVERY=200
export WARMDOWN_ITERS=4000
export WARMUP_STEPS=20

NPROC=8
MASTER_PORT=29500

run_one() {
    local name="$1"
    local script="$2"
    shift 2
    local env_vars=("$@")
    local log="$LOG_DIR/${name}.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $name"
    echo "  env: ${env_vars[*]:-<defaults>}"
    echo "  started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

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
            "$script"
    ) >"$log" 2>&1
    local rc=$?

    local final_bpb
    final_bpb=$(grep -oP 'val_bpb:\K[\d.]+' "$log" | tail -1 || echo "N/A")
    echo "  [$name] exit=$rc | final val_bpb: $final_bpb | ended: $(date '+%H:%M:%S')"
}

echo "Started remaining experiments: $(date)"

# ── Phase 3 remaining ──
echo "=== PHASE 3 (remaining) ==="

run_one "p3_8L_mlp4x_swiglu_trigram" "train_sota_exp.py" \
    "TTT_MODE=none" "NUM_LAYERS=8" "MLP_MULT=4" \
    "MLP_ACTIVATION=swiglu" "TRIGRAM=1"

run_one "p3_11L_swiglu_trigram" "train_sota_exp.py" \
    "TTT_MODE=none" "NUM_LAYERS=11" "MLP_MULT=3" \
    "MLP_ACTIVATION=swiglu" "TRIGRAM=1"

run_one "p3_9L_mlp4x_leaky_trigram" "train_sota_exp.py" \
    "TTT_MODE=none" "NUM_LAYERS=9" "MLP_MULT=4" \
    "MLP_ACTIVATION=leaky_relu2" "TRIGRAM=1"

# ── Phase 4: TTT ──
echo "=== PHASE 4: TTT ==="

run_one "p4_lora_r4" "train_sota_exp.py" \
    "TTT_MODE=lora" "TTT_LORA_RANK=4"
run_one "p4_lora_r8" "train_sota_exp.py" \
    "TTT_MODE=lora" "TTT_LORA_RANK=8"
run_one "p4_lora_r16" "train_sota_exp.py" \
    "TTT_MODE=lora" "TTT_LORA_RANK=16"
run_one "p4_lora_r32" "train_sota_exp.py" \
    "TTT_MODE=lora" "TTT_LORA_RANK=32" "TTT_BATCH_SIZE=32"

run_one "p4_fft_last2" "train_sota_exp.py" \
    "TTT_MODE=fft2"
run_one "p4_fft_last4" "train_sota_exp.py" \
    "TTT_MODE=fft4"
run_one "p4_fft_all" "train_sota_exp.py" \
    "TTT_MODE=fft_all"

run_one "p4_lora_r16_3step" "train_sota_exp.py" \
    "TTT_MODE=lora" "TTT_LORA_RANK=16" "TTT_STEPS=3" "TTT_LORA_LR=0.03"
run_one "p4_lora_r16_chunk128" "train_sota_exp.py" \
    "TTT_MODE=lora" "TTT_LORA_RANK=16" "TTT_CHUNK_SIZE=128"
run_one "p4_lora_r16_qvk" "train_sota_exp.py" \
    "TTT_MODE=lora" "TTT_LORA_RANK=16" "TTT_LORA_TARGETS=qvk"
run_one "p4_bias_ttt" "train_sota_exp.py" \
    "TTT_MODE=bias"
run_one "p4_lora_r16_qv_mlp" "train_sota_exp.py" \
    "TTT_MODE=lora" "TTT_LORA_RANK=16" "TTT_LORA_TARGETS=qv_mlp"
run_one "p4_lora_r16_qvk_mlp" "train_sota_exp.py" \
    "TTT_MODE=lora" "TTT_LORA_RANK=16" "TTT_LORA_TARGETS=qvk_mlp"

echo ""
echo "All remaining experiments complete: $(date)"

# Summary
echo ""
echo "FULL RESULTS (all phases):"
echo "────────────────────────────────────────"
printf "%-35s %10s %8s\n" "Experiment" "Final BPB" "Steps"
echo "────────────────────────────────────────"
for log in "$LOG_DIR"/p[1234]_*.log; do
    name=$(basename "$log" .log)
    bpb=$(grep -oP 'val_bpb:\K[\d.]+' "$log" | tail -1 || echo "N/A")
    step=$(grep -oP 'stopping_early.*step:\K\d+' "$log" || echo "?")
    printf "%-35s %10s %8s\n" "$name" "$bpb" "$step"
done | sort -t' ' -k2 -n
echo "────────────────────────────────────────"
