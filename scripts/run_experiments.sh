#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_ROOT/experiment_logs"
mkdir -p "$LOG_DIR"

# Activate venv if present
VENV="$REPO_ROOT/.venv/bin/activate"
[[ -f "$VENV" ]] && source "$VENV"

# Common env vars
export DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_sp1024/"
export TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model"
export VOCAB_SIZE=1024
export MAX_WALLCLOCK_SECONDS=90
export VAL_LOSS_EVERY=10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable torch.compile to avoid OOM from compilation caches
export TORCHDYNAMO_DISABLE=1

# Which phases to run (pass as args, e.g. ./run_experiments.sh 1 or ./run_experiments.sh 1 2 3)
if [[ $# -gt 0 ]]; then
    PHASES=("$@")
else
    PHASES=(1)
fi

run_one() {
    local name="$1"
    local script="$2"
    shift 2
    local env_vars=("$@")
    local log="$LOG_DIR/${name}.log"

    echo "━━━ $name"
    echo "    script: $script"
    echo "    env: ${env_vars[*]:-<defaults>}"
    echo "    log: $log"

    (
        cd "$REPO_ROOT"
        # Export experiment-specific env vars
        for ev in "${env_vars[@]}"; do
            export "$ev"
        done
        export RUN_ID="$name"
        timeout 600 torchrun --standalone --master_port=29500 --nproc_per_node=1 "$script"
    ) >"$log" 2>&1 && status="ok" || status="FAILED (exit $?)"

    echo "    [$name] $status"
    echo ""
}

# ─────────────────────────────────────────
# PHASE 1: Env-var only (train_gpt.py)
# ─────────────────────────────────────────
run_phase1() {
    echo "╔══════════════════════════════════════╗"
    echo "║  PHASE 1: Architecture Variants      ║"
    echo "╚══════════════════════════════════════╝"

    run_one "p1_baseline" "train_gpt.py"

    run_one "p1_mlp4x_4h4kv" "train_gpt.py" \
        "NUM_HEADS=4" "NUM_KV_HEADS=4" "MLP_MULT=4"

    run_one "p1_16q_4kv" "train_gpt.py" \
        "NUM_HEADS=16" "NUM_KV_HEADS=4"

    run_one "p1_16q_8kv_mlp2x" "train_gpt.py" \
        "NUM_HEADS=16" "NUM_KV_HEADS=8" "MLP_MULT=2"
}

# ─────────────────────────────────────────
# PHASE 2: TTT + small arch (train_alpha.py)
# ─────────────────────────────────────────
run_phase2() {
    echo "╔══════════════════════════════════════╗"
    echo "║  PHASE 2: TTT + Architecture Mods    ║"
    echo "╚══════════════════════════════════════╝"

    # TTT variants
    run_one "p2_ttt_lora_r8" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_LORA_RANK=8"

    run_one "p2_ttt_chunk128" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_CHUNK_SIZE=128"

    run_one "p2_ttt_multistep3" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_STEPS=3" "TTT_LORA_LR=0.03"

    run_one "p2_ttt_bias_only" "train_alpha.py" \
        "TTT_MODE=bias"

    # FFT variants (priority)
    run_one "p2_fft_last2" "train_alpha.py" \
        "TTT_MODE=fft2"

    # Architecture mods
    run_one "p2_residual_sqrt" "train_alpha.py" \
        "TTT_MODE=none" "RESIDUAL_MODE=sqrt"

    run_one "p2_residual_gated" "train_alpha.py" \
        "TTT_MODE=none" "RESIDUAL_MODE=gated"

    run_one "p2_ln_inv_sqrt" "train_alpha.py" \
        "TTT_MODE=none" "LN_SCHEDULE=inv_sqrt"

    run_one "p2_trigram_hash" "train_alpha.py" \
        "TTT_MODE=none" "HASH_MODE=trigram"

    run_one "p2_local_conv3" "train_alpha.py" \
        "TTT_MODE=none" "LOCAL_CONV=3"
}

# ─────────────────────────────────────────
# PHASE 3: LoRA vs FFT (train_alpha.py)
# ─────────────────────────────────────────
run_phase3() {
    echo "╔══════════════════════════════════════╗"
    echo "║  PHASE 3: LoRA vs FFT Showdown       ║"
    echo "╚══════════════════════════════════════╝"

    run_one "p3_lora_rank16" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_LORA_RANK=16"

    run_one "p3_lora_rank32" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_LORA_RANK=32"

    run_one "p3_fft_last4" "train_alpha.py" \
        "TTT_MODE=fft4"

    run_one "p3_fft2_multistep3" "train_alpha.py" \
        "TTT_MODE=fft2" "TTT_STEPS=3"
}

# ─────────────────────────────────────────
# Run requested phases
# ─────────────────────────────────────────
for phase in "${PHASES[@]}"; do
    case "$phase" in
        1) run_phase1 ;;
        2) run_phase2 ;;
        3) run_phase3 ;;
        *) echo "Unknown phase: $phase" ;;
    esac
done

echo ""
echo "════════════════════════════════════════"
echo "All requested phases complete."
echo "Logs in: $LOG_DIR/"
echo "════════════════════════════════════════"

# Generate plots
if [[ -f "$REPO_ROOT/scripts/plot_experiments.py" ]]; then
    echo "Generating plots..."
    cd "$REPO_ROOT" && python3 scripts/plot_experiments.py
fi
