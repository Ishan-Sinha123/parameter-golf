#!/usr/bin/env bash
set -euo pipefail
#
# Scaled-down ablation runner.
#
# The original run_experiments.sh uses full-size models (11L/512d/MLP3x for
# train_gpt.py, 9L/512d/MLP2x for train_alpha.py) with a 90-second cap.
# At ~1900-4100 ms/step that yields only ~25-49 training steps — too few for
# loss curves to differentiate between ablations.
#
# This script shrinks the model so each run completes ~200-400 steps in 90s,
# enough to see real curve separation.
#
# Scaled-down config:
#   NUM_LAYERS=6  MODEL_DIM=384  MLP_MULT=2  TRAIN_SEQ_LEN=512
#   (targets ~250-400 ms/step → 200-350 steps in 90s)
#
# Usage:
#   ./run_experiments_scaled.sh          # run phase 1 only (default)
#   ./run_experiments_scaled.sh 1 2 3    # run all phases
#   ./run_experiments_scaled.sh 2        # run phase 2 only

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$REPO_ROOT/experiment_logs_scaled"
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
export TORCHDYNAMO_DISABLE=1

# ── Scaled-down model config ──
# These override the defaults in both train_gpt.py and train_alpha.py
export NUM_LAYERS=6
export MODEL_DIM=384
export MLP_MULT=2
export TRAIN_SEQ_LEN=512
export TRAIN_BATCH_TOKENS=262144   # smaller batch to fit reduced model

# train_gpt.py-specific overrides (features that reference layer count)
export XSA_LAST_N=6
export BIGRAM_VOCAB_SIZE=1024
export BIGRAM_DIM=64
export VE_LAYERS="4,5"

# Faster warmdown proportional to fewer expected steps
export WARMDOWN_ITERS=800
export WARMUP_STEPS=10

# Which phases to run
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
        # Export experiment-specific env vars (these override the scaled defaults)
        for ev in "${env_vars[@]+"${env_vars[@]}"}"; do
            export "$ev"
        done
        export RUN_ID="$name"
        timeout 600 torchrun --standalone --master_port=29500 --nproc_per_node=1 "$script"
    ) >"$log" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        status="ok"
    elif [[ $rc -eq 124 ]]; then
        status="ok (timeout — expected)"
    else
        status="FAILED (exit $rc)"
    fi

    echo "    [$name] $status"
    echo ""
}

# ─────────────────────────────────────────
# PHASE 1: Architecture Variants (train_alpha.py)
# ─────────────────────────────────────────
run_phase1() {
    echo "╔══════════════════════════════════════╗"
    echo "║  PHASE 1: Architecture Variants      ║"
    echo "║  (scaled: 6L/384d/MLP2x/seq512)      ║"
    echo "╚══════════════════════════════════════╝"

    run_one "p1_baseline" "train_alpha.py" \
        "TTT_MODE=none"

    run_one "p1_mlp4x_4h4kv" "train_alpha.py" \
        "TTT_MODE=none" "NUM_HEADS=4" "NUM_KV_HEADS=4" "MLP_MULT=4"

    run_one "p1_16q_4kv" "train_alpha.py" \
        "TTT_MODE=none" "NUM_HEADS=16" "NUM_KV_HEADS=4"

    run_one "p1_16q_8kv_mlp2x" "train_alpha.py" \
        "TTT_MODE=none" "NUM_HEADS=16" "NUM_KV_HEADS=8" "MLP_MULT=2"
}

# ─────────────────────────────────────────
# PHASE 2: TTT + small arch (train_alpha.py)
# ─────────────────────────────────────────
run_phase2() {
    echo "╔══════════════════════════════════════╗"
    echo "║  PHASE 2: TTT + Architecture Mods    ║"
    echo "║  (scaled: 6L/384d/MLP2x/seq512)      ║"
    echo "╚══════════════════════════════════════╝"

    # Alpha baseline (no TTT, no arch mods) to isolate framework differences
    run_one "p2_alpha_baseline" "train_alpha.py" \
        "TTT_MODE=none"

    # TTT variants
    run_one "p2_ttt_lora_r8" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_LORA_RANK=8"

    run_one "p2_ttt_chunk128" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_CHUNK_SIZE=128"

    run_one "p2_ttt_multistep3" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_STEPS=3" "TTT_LORA_LR=0.03"

    run_one "p2_ttt_bias_only" "train_alpha.py" \
        "TTT_MODE=bias"

    # FFT variants
    run_one "p2_fft_last2" "train_alpha.py" \
        "TTT_MODE=fft2"

    # Architecture mods
    run_one "p2_residual_sqrt" "train_alpha.py" \
        "TTT_MODE=none" "RESIDUAL_MODE=sqrt"

    run_one "p2_residual_gated" "train_alpha.py" \
        "TTT_MODE=none" "RESIDUAL_MODE=gated"

    run_one "p2_ln_inv_sqrt" "train_alpha.py" \
        "TTT_MODE=none" "LN_SCHEDULE=inv_sqrt"

    run_one "p2_bigram_hash" "train_alpha.py" \
        "TTT_MODE=none" "HASH_MODE=bigram"

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
    echo "║  (scaled: 6L/384d/MLP2x/seq512)      ║"
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
# PHASE 4: Heavy infrastructure (train_alpha.py)
# ─────────────────────────────────────────
run_phase4() {
    echo "╔══════════════════════════════════════╗"
    echo "║  PHASE 4: Heavy Infrastructure       ║"
    echo "║  (scaled: 6L/384d/MLP2x/seq512)      ║"
    echo "╚══════════════════════════════════════╝"

    run_one "p4_fft_all" "train_alpha.py" \
        "TTT_MODE=fft_all"
}

# ─────────────────────────────────────────
# PHASE 5: Extended TTT (train_alpha.py)
# ─────────────────────────────────────────
run_phase5() {
    echo "╔══════════════════════════════════════╗"
    echo "║  PHASE 5: Extended TTT               ║"
    echo "║  (scaled: 6L/384d/MLP2x/seq512)      ║"
    echo "╚══════════════════════════════════════╝"

    run_one "p5_ttt_chunk64" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_CHUNK_SIZE=64"

    run_one "p5_ttt_steps5" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_STEPS=5" "TTT_LORA_LR=0.01"

    run_one "p5_ttt_qvk" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_LORA_TARGETS=qvk"

    run_one "p5_ttt_no_reset" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_LORA_DECAY=0.0"

    run_one "p5_ttt_partial_decay" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_LORA_DECAY=0.1"

    # LoRA rank sweep (complements phase 3's rank 16/32)
    run_one "p5_lora_rank2" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_LORA_RANK=2"

    run_one "p5_lora_rank4" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_LORA_RANK=4"

    run_one "p5_lora_rank64" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_LORA_RANK=64"

    # LoRA on MLP layers (fc + proj) in addition to attention
    run_one "p5_lora_qv_mlp" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_LORA_TARGETS=qv_mlp"

    run_one "p5_lora_qvk_mlp" "train_alpha.py" \
        "TTT_MODE=lora" "TTT_LORA_TARGETS=qvk_mlp"
}

# ─────────────────────────────────────────
# PHASE 6: MLP expansion + activation (train_alpha.py)
# ─────────────────────────────────────────
run_phase6() {
    echo "╔══════════════════════════════════════╗"
    echo "║  PHASE 6: MLP Expansion + Activation ║"
    echo "║  (scaled: 6L/384d/MLP2x/seq512)      ║"
    echo "╚══════════════════════════════════════╝"

    run_one "p6_mlp3x" "train_alpha.py" \
        "TTT_MODE=none" "MLP_MULT=3"

    run_one "p6_leaky_relu2" "train_alpha.py" \
        "TTT_MODE=none" "MLP_ACTIVATION=leaky_relu2"

    run_one "p6_leaky_relu2_mlp3x" "train_alpha.py" \
        "TTT_MODE=none" "MLP_ACTIVATION=leaky_relu2" "MLP_MULT=3"

    run_one "p6_swiglu_mlp3x" "train_alpha.py" \
        "TTT_MODE=none" "MLP_ACTIVATION=swiglu" "MLP_MULT=3"

    run_one "p6_deep_narrow" "train_alpha.py" \
        "TTT_MODE=none" "NUM_LAYERS=8" "MLP_MULT=2" "XSA_LAST_N=8" "VE_LAYERS=6,7"

    run_one "p6_shallow_wide" "train_alpha.py" \
        "TTT_MODE=none" "NUM_LAYERS=4" "MLP_MULT=4" "XSA_LAST_N=4" "VE_LAYERS=2,3"
}

# ─────────────────────────────────────────
# PHASE 7: Layer importance (train_alpha.py)
# ─────────────────────────────────────────
run_phase7() {
    echo "╔══════════════════════════════════════╗"
    echo "║  PHASE 7: Layer Importance           ║"
    echo "║  (scaled: 6L/384d/MLP2x/seq512)      ║"
    echo "╚══════════════════════════════════════╝"

    run_one "p7_layer_stats" "train_alpha.py" \
        "TTT_MODE=none" "LOG_LAYER_STATS=1"

    run_one "p7_layer_drop_0.1" "train_alpha.py" \
        "TTT_MODE=none" "LAYER_DROP_RATE=0.1"

    run_one "p7_layer_drop_0.2" "train_alpha.py" \
        "TTT_MODE=none" "LAYER_DROP_RATE=0.2"

    run_one "p7_learnable_ln" "train_alpha.py" \
        "TTT_MODE=none" "LEARNABLE_LN_GAIN=1" "LN_SCHEDULE=inv_sqrt"

    run_one "p7_resid_gated_learn" "train_alpha.py" \
        "TTT_MODE=none" "RESIDUAL_MODE=gated" "LEARNABLE_LN_GAIN=1"
}

# ─────────────────────────────────────────
# PHASE 8: Gram Newton-Schulz (train_alpha.py)
# ─────────────────────────────────────────
run_phase8() {
    echo "╔══════════════════════════════════════╗"
    echo "║  PHASE 8: Gram Newton-Schulz         ║"
    echo "║  (scaled: 6L/384d/MLP2x/seq512)      ║"
    echo "╚══════════════════════════════════════╝"

    run_one "p8_gram_ns_baseline" "train_alpha.py" \
        "TTT_MODE=none" "MUON_BACKEND=gram_ns"

    run_one "p8_gram_ns_steps3" "train_alpha.py" \
        "TTT_MODE=none" "MUON_BACKEND=gram_ns" "MUON_BACKEND_STEPS=3"

    run_one "p8_gram_ns_steps5" "train_alpha.py" \
        "TTT_MODE=none" "MUON_BACKEND=gram_ns" "MUON_BACKEND_STEPS=5"
}

# ─────────────────────────────────────────
# Run requested phases
# ─────────────────────────────────────────
for phase in "${PHASES[@]}"; do
    case "$phase" in
        1) run_phase1 ;;
        2) run_phase2 ;;
        3) run_phase3 ;;
        4) run_phase4 ;;
        5) run_phase5 ;;
        6) run_phase6 ;;
        7) run_phase7 ;;
        8) run_phase8 ;;
        *) echo "Unknown phase: $phase" ;;
    esac
done

echo ""
echo "════════════════════════════════════════"
echo "All requested phases complete."
echo "Logs in: $LOG_DIR/"
echo "════════════════════════════════════════"

# Generate plots from the scaled logs
if [[ -f "$REPO_ROOT/scripts/plot_experiments.py" ]]; then
    echo "Generating plots..."
    cd "$REPO_ROOT" && python3 scripts/plot_experiments.py --log-dir "$LOG_DIR" --out-dir "${LOG_DIR}/plots"
fi
