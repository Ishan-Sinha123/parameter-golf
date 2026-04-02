#!/usr/bin/env bash
set -euo pipefail
#
# Full-scale experiment runner for 8×H100.
#
# Runs architecture experiments + TTT comparison experiments.
# Each run: 90s training + full TTT evaluation (no wallclock cap on eval).
#
# Architecture experiments use TTT_MODE=none (no post-training eval → ~2 min each).
# TTT experiments train the same base model but evaluate different TTT strategies
# (~10-15 min each including eval).
#
# Total estimated time: ~2 hours
#
# Usage:
#   ./scripts/run_fullscale.sh              # run all phases sequentially
#   ./scripts/run_fullscale.sh 1            # run Phase 1 only (architecture)
#   ./scripts/run_fullscale.sh 2            # run Phase 2 only (depth/width)
#   ./scripts/run_fullscale.sh 3            # run Phase 3 only (combined winners)
#   ./scripts/run_fullscale.sh 4            # run Phase 4 only (TTT: LoRA vs FFT)
#   ./scripts/run_fullscale.sh 1 2 3 4      # run all phases

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_ROOT/experiment_logs_fullscale"
mkdir -p "$LOG_DIR"

# Activate venv if present
VENV="$REPO_ROOT/.venv/bin/activate"
[[ -f "$VENV" ]] && source "$VENV"

# ── Common env vars ──
export DATA_PATH="$REPO_ROOT/data/datasets/fineweb10B_sp1024/"
export TOKENIZER_PATH="$REPO_ROOT/data/tokenizers/fineweb_1024_bpe.model"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1

# ── Full-scale model config ──
# Matches SOTA exactly (PR #1019): 11L/512d/MLP3x + BigramHash 3072×112
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

# ── Training budget (SOTA uses 600s, we use 90s for fast iteration) ──
export MAX_WALLCLOCK_SECONDS=90
export VAL_LOSS_EVERY=200
export WARMDOWN_ITERS=4000
export WARMUP_STEPS=20

# ── GPU config ──
NPROC=8
MASTER_PORT=29500

# Which phases to run
if [[ $# -gt 0 ]]; then
    PHASES=("$@")
else
    PHASES=(1 2 3 4)
fi

run_one() {
    local name="$1"
    local script="$2"
    shift 2
    local env_vars=("$@")
    local log="$LOG_DIR/${name}.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $name"
    echo "  script: $script"
    echo "  env: ${env_vars[*]:-<defaults>}"
    echo "  log: $log"
    echo "  started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    (
        cd "$REPO_ROOT"
        # Export experiment-specific env vars
        for ev in "${env_vars[@]+"${env_vars[@]}"}"; do
            export "$ev"
        done
        export RUN_ID="$name"
        # 3600s timeout = 60 min max (90s training + GPTQ + LZMA + TTT eval)
        timeout 3600 torchrun \
            --standalone \
            --master_port=$MASTER_PORT \
            --nproc_per_node=$NPROC \
            "$script"
    ) >"$log" 2>&1
    local rc=$?
    local end_time
    end_time=$(date '+%H:%M:%S')

    if [[ $rc -eq 0 ]]; then
        status="ok"
    elif [[ $rc -eq 124 ]]; then
        status="ok (timeout — expected)"
    else
        status="FAILED (exit $rc)"
    fi

    # Extract final val_bpb from log
    local final_bpb
    final_bpb=$(grep -oP 'val_bpb:\K[\d.]+' "$log" | tail -1 || echo "N/A")

    echo "  [$name] $status | final val_bpb: $final_bpb | ended: $end_time"
    echo ""
}

# ─────────────────────────────────────────────────────────────
# PHASE 1: Isolate Individual Gains (Architecture Experiments)
#
# TTT_MODE=none → no post-training eval → fast (~2 min each)
# ─────────────────────────────────────────────────────────────
run_phase1() {
    echo ""
    echo "╔═══════════════════════════════════════════════╗"
    echo "║  PHASE 1: Isolate Individual Gains            ║"
    echo "║  (SOTA config, 8×H100, 600s training)          ║"
    echo "║  TTT_MODE=none — architecture only            ║"
    echo "╚═══════════════════════════════════════════════╝"

    # 1A. Baseline — control run
    run_one "p1_baseline" "train_sota_exp.py" \
        "TTT_MODE=none"
    # 1B. SwiGLU activation (replace ReLU²)
    run_one "p1_swiglu" "train_sota_exp.py" \
        "TTT_MODE=none" "MLP_ACTIVATION=swiglu"
    # 1C. Trigram hash embedding
    run_one "p1_trigram_hash" "train_sota_exp.py" \
        "TTT_MODE=none" "TRIGRAM=1"
    # 1D. SwiGLU + Trigram (both features together)
    run_one "p1_swiglu_trigram" "train_sota_exp.py" \
        "TTT_MODE=none" "MLP_ACTIVATION=swiglu" "TRIGRAM=1"
    # 1E. LeakyReLU² (free activation swap)
    run_one "p1_leaky_relu2" "train_sota_exp.py" \
        "TTT_MODE=none" "MLP_ACTIVATION=leaky_relu2"
}

# ─────────────────────────────────────────────────────────────
# PHASE 2: Depth/Width Tradeoff
#
# Test whether shallow+wide beats deep+narrow at full scale.
# TTT_MODE=none — architecture only.
# ─────────────────────────────────────────────────────────────
run_phase2() {
    echo ""
    echo "╔═══════════════════════════════════════════════╗"
    echo "║  PHASE 2: Depth/Width Tradeoff                ║"
    echo "║  (8×H100, 90s training, TTT_MODE=none)        ║"
    echo "╚═══════════════════════════════════════════════╝"

    # 2A. Conservative shallow: 9L/MLP4x
    run_one "p2_9L_mlp4x" "train_sota_exp.py" \
        "TTT_MODE=none" "NUM_LAYERS=9" "MLP_MULT=4"
    # 2B. Moderate shallow: 8L/MLP4x
    run_one "p2_8L_mlp4x" "train_sota_exp.py" \
        "TTT_MODE=none" "NUM_LAYERS=8" "MLP_MULT=4"
    # 2C. 9L/MLP3x (reduced depth, same width as SOTA)
    run_one "p2_9L_mlp3x" "train_sota_exp.py" \
        "TTT_MODE=none" "NUM_LAYERS=9" "MLP_MULT=3"
    # 2D. 7L/MLP4x (aggressive shallow)
    run_one "p2_7L_mlp4x" "train_sota_exp.py" \
        "TTT_MODE=none" "NUM_LAYERS=7" "MLP_MULT=4"
}

# ─────────────────────────────────────────────────────────────
# PHASE 3: Combined Winners
#
# Stack best features from Phase 1 onto best arch from Phase 2.
# TTT_MODE=none — architecture only.
# ─────────────────────────────────────────────────────────────
run_phase3() {
    echo ""
    echo "╔═══════════════════════════════════════════════╗"
    echo "║  PHASE 3: Combined Winners                    ║"
    echo "║  (8×H100, 90s training, TTT_MODE=none)        ║"
    echo "╚═══════════════════════════════════════════════╝"

    # 3A. 9L/MLP4x + SwiGLU + Trigram
    run_one "p3_9L_mlp4x_swiglu_trigram" "train_sota_exp.py" \
        "TTT_MODE=none" "NUM_LAYERS=9" "MLP_MULT=4" \
        "MLP_ACTIVATION=swiglu" "TRIGRAM=1"
    # 3B. 8L/MLP4x + SwiGLU + Trigram
    run_one "p3_8L_mlp4x_swiglu_trigram" "train_sota_exp.py" \
        "TTT_MODE=none" "NUM_LAYERS=8" "MLP_MULT=4" \
        "MLP_ACTIVATION=swiglu" "TRIGRAM=1"
    # 3C. 11L (SOTA depth) + SwiGLU + Trigram (safe combo)
    run_one "p3_11L_swiglu_trigram" "train_sota_exp.py" \
        "TTT_MODE=none" "NUM_LAYERS=11" "MLP_MULT=3" \
        "MLP_ACTIVATION=swiglu" "TRIGRAM=1"
    # 3D. Best arch + LeakyReLU² + Trigram (fallback if SwiGLU fails)
    run_one "p3_9L_mlp4x_leaky_trigram" "train_sota_exp.py" \
        "TTT_MODE=none" "NUM_LAYERS=9" "MLP_MULT=4" \
        "MLP_ACTIVATION=leaky_relu2" "TRIGRAM=1"
}

# ─────────────────────────────────────────────────────────────
# PHASE 4: TTT Comparison — LoRA vs FFT + Rank Sweep
#
# All experiments train the SAME base model (11L/MLP3x baseline).
# They differ ONLY in the post-training TTT evaluation strategy.
# Each run takes ~10-15 min (90s training + 5-12 min TTT eval).
#
# NOTE: These all produce identical training curves. The signal
# is in the FINAL val_bpb after TTT eval, not the training curve.
# ─────────────────────────────────────────────────────────────
run_phase4() {
    echo ""
    echo "╔═══════════════════════════════════════════════╗"
    echo "║  PHASE 4: TTT — LoRA vs FFT + Rank Sweep     ║"
    echo "║  (11L/512d/MLP3x, 8×H100, 90s + full eval)   ║"
    echo "║  Same training, different post-train eval     ║"
    echo "╚═══════════════════════════════════════════════╝"

    # ── LoRA Rank Sweep ──

    # 4A. LoRA rank 4 (small adapter)
    run_one "p4_lora_r4" "train_sota_exp.py" \
        "TTT_MODE=lora" "TTT_LORA_RANK=4"

    # 4B. LoRA rank 8 (current default)
    run_one "p4_lora_r8" "train_sota_exp.py" \
        "TTT_MODE=lora" "TTT_LORA_RANK=8"

    # 4C. LoRA rank 16 (sweet spot from scaled tests)
    run_one "p4_lora_r16" "train_sota_exp.py" \
        "TTT_MODE=lora" "TTT_LORA_RANK=16"

    # 4D. LoRA rank 32
    run_one "p4_lora_r32" "train_sota_exp.py" \
        "TTT_MODE=lora" "TTT_LORA_RANK=32" "TTT_BATCH_SIZE=32"

    # ── FFT (Full Fine-Tuning) variants ──

    # 4E. FFT last 2 layers
    run_one "p4_fft_last2" "train_sota_exp.py" \
        "TTT_MODE=fft2"

    # 4F. FFT last 4 layers
    run_one "p4_fft_last4" "train_sota_exp.py" \
        "TTT_MODE=fft4"

    # 4G. FFT all layers
    run_one "p4_fft_all" "train_sota_exp.py" \
        "TTT_MODE=fft_all"

    # ── LoRA Variations ──

    # 4H. LoRA r16 + 3-step inner loop
    run_one "p4_lora_r16_3step" "train_sota_exp.py" \
        "TTT_MODE=lora" "TTT_LORA_RANK=16" "TTT_STEPS=3" "TTT_LORA_LR=0.03"

    # 4I. LoRA r16 + chunk size 128 (finer-grained adaptation)
    run_one "p4_lora_r16_chunk128" "train_sota_exp.py" \
        "TTT_MODE=lora" "TTT_LORA_RANK=16" "TTT_CHUNK_SIZE=128"

    # 4J. LoRA r16 targeting Q, V, K (instead of just Q, V)
    run_one "p4_lora_r16_qvk" "train_sota_exp.py" \
        "TTT_MODE=lora" "TTT_LORA_RANK=16" "TTT_LORA_TARGETS=qvk"

    # 4K. Bias-only TTT (minimal adapter)
    run_one "p4_bias_ttt" "train_sota_exp.py" \
        "TTT_MODE=bias"

    # 4L. LoRA r16 targeting Q, V + MLP (attention + MLP adapters)
    run_one "p4_lora_r16_qv_mlp" "train_sota_exp.py" \
        "TTT_MODE=lora" "TTT_LORA_RANK=16" "TTT_LORA_TARGETS=qv_mlp"

    # 4M. LoRA r16 targeting Q, V, K + MLP (full adapter coverage)
    run_one "p4_lora_r16_qvk_mlp" "train_sota_exp.py" \
        "TTT_MODE=lora" "TTT_LORA_RANK=16" "TTT_LORA_TARGETS=qvk_mlp"
}

# ─────────────────────────────────────────────────────────────
# Run requested phases
# ─────────────────────────────────────────────────────────────
echo "╔═══════════════════════════════════════════════╗"
echo "║  Full-Scale Experiment Runner                 ║"
echo "║  8×H100 | 90s training | train_sota_exp.py       ║"
echo "║  Phases: ${PHASES[*]}                              ║"
echo "║  Logs: $LOG_DIR/"
echo "╚═══════════════════════════════════════════════╝"
echo ""
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

for phase in "${PHASES[@]}"; do
    case "$phase" in
        1) run_phase1 ;;
        2) run_phase2 ;;
        3) run_phase3 ;;
        4) run_phase4 ;;
        *) echo "Unknown phase: $phase" ;;
    esac
done

echo ""
echo "════════════════════════════════════════════════"
echo "All requested phases complete."
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Logs in: $LOG_DIR/"
echo "════════════════════════════════════════════════"

# ── Summary table ──
echo ""
echo "RESULTS SUMMARY (final val_bpb per experiment):"
echo "────────────────────────────────────────────────"
printf "%-35s %10s %8s\n" "Experiment" "Val BPB" "Steps"
echo "────────────────────────────────────────────────"
for log in "$LOG_DIR"/*.log; do
    name=$(basename "$log" .log)
    bpb=$(grep -oP 'val_bpb:\K[\d.]+' "$log" | tail -1 || echo "N/A")
    step=$(grep -oP 'step:\K\d+' "$log" | tail -1 || echo "0")
    printf "%-35s %10s %8s\n" "$name" "$bpb" "$step"
done | sort -t' ' -k2 -n
echo "────────────────────────────────────────────────"

# Generate plots if available
if [[ -f "$REPO_ROOT/scripts/plot_experiments.py" ]]; then
    echo ""
    echo "Generating plots..."
    python3 "$REPO_ROOT/scripts/plot_experiments.py" \
        --log-dir "$LOG_DIR" \
        --out-dir "$LOG_DIR/plots"
fi
