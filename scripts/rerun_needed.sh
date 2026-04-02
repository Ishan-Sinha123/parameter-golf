#!/usr/bin/env bash
set -euo pipefail
# Rerun all incomplete/corrupted experiments
# NO pkill between experiments — that was killing successor runs

REPO_ROOT="/workspace/parameter-golf"
LOG_DIR="$REPO_ROOT/experiment_logs_fullscale"
RUNNER_LOG="$LOG_DIR/runner.log"
mkdir -p "$LOG_DIR"

source "$REPO_ROOT/.venv/bin/activate"

export DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_sp1024/"
export TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/hf_home
export VOCAB_SIZE=1024 NUM_LAYERS=11 MODEL_DIM=512
export NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3
export TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=524288
export BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112 TARGET_MB=15.9
export MAX_WALLCLOCK_SECONDS=90 VAL_LOSS_EVERY=200
export WARMDOWN_ITERS=4000 WARMUP_STEPS=20

NPROC=8
MASTER_PORT=29500

run_one() {
    local name="$1"; shift
    local env_vars=("$@")
    local log="$LOG_DIR/${name}.log"

    # Skip if completed (has final sliding window result)
    if [[ -f "$log" ]] && grep -q 'final_int6_sliding_window_exact' "$log" 2>/dev/null; then
        echo "$(date '+%H:%M:%S') SKIP  $name (already complete)" | tee -a "$RUNNER_LOG"
        return 0
    fi

    echo "$(date '+%H:%M:%S') START $name" | tee -a "$RUNNER_LOG"

    (
        cd "$REPO_ROOT"
        for ev in "${env_vars[@]+"${env_vars[@]}"}"; do export "$ev"; done
        export RUN_ID="$name"
        timeout 3600 torchrun --standalone --master_port=$MASTER_PORT --nproc_per_node=$NPROC train_sota_exp.py
    ) >"$log" 2>&1
    local rc=$?

    local final_bpb
    final_bpb=$(grep -oP 'val_bpb:\K[\d.]+' "$log" 2>/dev/null | tail -1 || echo "N/A")
    local steps
    steps=$(grep -oP 'stopping_early.*step:\K\d+' "$log" 2>/dev/null || echo "?")
    echo "$(date '+%H:%M:%S') DONE  $name | exit=$rc | steps=$steps | val_bpb=$final_bpb" | tee -a "$RUNNER_LOG"
}

echo "" >> "$RUNNER_LOG"
echo "$(date '+%H:%M:%S') === RERUN batch ===" | tee -a "$RUNNER_LOG"

# Phase 3 remaining (clean reruns)
run_one "p3_11L_swiglu_trigram" \
    "TTT_MODE=none" "NUM_LAYERS=11" "MLP_MULT=3" "MLP_ACTIVATION=swiglu" "TRIGRAM=1"
run_one "p3_9L_mlp4x_leaky_trigram" \
    "TTT_MODE=none" "NUM_LAYERS=9" "MLP_MULT=4" "MLP_ACTIVATION=leaky_relu2" "TRIGRAM=1"

# Phase 4: TTT — all of them
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

echo "$(date '+%H:%M:%S') === ALL DONE ===" | tee -a "$RUNNER_LOG"

# Final summary
echo ""
echo "RESULTS (all experiments):"
printf "%-35s %6s %8s %15s\n" "Experiment" "Steps" "ms/step" "Final BPB"
echo "────────────────────────────────────────────────────────────────────"
for log_f in "$LOG_DIR"/p[1234]_*.log; do
    n=$(basename "$log_f" .log)
    s=$(grep -oP 'stopping_early.*step:\K\d+' "$log_f" 2>/dev/null || echo "?")
    m=$(grep 'step_avg' "$log_f" 2>/dev/null | tail -1 | grep -oP 'step_avg:\K[\d.]+' || echo "?")
    b=$(grep -oP 'final_int6_sliding_window_exact.*val_bpb:\K[\d.]+' "$log_f" 2>/dev/null | tail -1 || echo "N/A")
    printf "%-35s %6s %8s %15s\n" "$n" "$s" "$m" "$b"
done | sort -t' ' -k4 -n
