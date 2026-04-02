# Full-Scale Experiment Plan (v3)

**Goal:** Beat current SOTA of 1.11473 val_bpb
**Script:** Unmodified SOTA `train_gpt.py` from `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/`
**Runner:** `./scripts/run_ablations.sh`
**Approach:** Every experiment runs the exact same SOTA code with different env var overrides. No code forks.

---

## Quick Start

### 1. Setup (RunPod 8×H100 SXM)

```bash
git clone git@github.com:Ishan-Sinha123/parameter-golf.git
cd parameter-golf
pip install -r requirements.txt
python3 data/cached_challenge_fineweb.py --variant sp1024
```

### 2. Run SOTA baseline first (validate your setup)

```bash
./scripts/run_ablations.sh 0
```

This reproduces PR #1019 exactly. Expected: ~1.1147 BPB on 8×H100 SXM, ~6900 steps in 600s.

### 3. Run ablation categories

```bash
# Architecture experiments (most promising)
./scripts/run_ablations.sh A

# Training dynamics
./scripts/run_ablations.sh B

# Eval-time (stride changes, no retraining needed)
./scripts/run_ablations.sh C

# Combinations (run after A/B/C show signal)
./scripts/run_ablations.sh D
```

### 4. Run specific experiments

```bash
./scripts/run_ablations.sh A1 A3 B1    # cherry-pick
```

### 5. Quick test runs (short budget, fewer GPUs)

```bash
NGPUS=1 MAX_WALLCLOCK_SECONDS=90 ./scripts/run_ablations.sh 0 A1
```

### 6. Dry run (see commands without executing)

```bash
DRY_RUN=1 ./scripts/run_ablations.sh
```

Logs go to `experiment_logs/ablations/`. Summary printed at end of each run.

---

## Experiment Matrix

All experiments use the unmodified SOTA `train_gpt.py`. Only env vars change.

### Experiment 0: SOTA Baseline

| ID | Env Overrides | Description |
|----|--------------|-------------|
| **0** | *(none)* | Reproduce PR #1019 exactly (1.1147 BPB) |

### Category A: Architecture

These test the depth/width tradeoff — our strongest finding from prior experiments. Note: `XSA_LAST_N` and `VE_LAYERS` are adjusted to match the new layer count.

| ID | Env Overrides | Description |
|----|--------------|-------------|
| **A1** | `NUM_LAYERS=9 MLP_MULT=3.5 XSA_LAST_N=9 VE_LAYERS=7,8` | 9L + MLP 3.5x (PR #1105 width, fewer layers) |
| **A2** | `NUM_LAYERS=8 MLP_MULT=4.0 XSA_LAST_N=8 VE_LAYERS=6,7` | 8L + MLP 4x (prior best combo) |
| **A3** | `NUM_LAYERS=7 MLP_MULT=4.0 XSA_LAST_N=7 VE_LAYERS=5,6` | 7L + MLP 4x (scaled test winner) |
| **A4** | `NUM_LAYERS=11 MLP_MULT=3.5` | 11L + MLP 3.5x (wider MLP, same depth as SOTA) |

**Rationale:** Prior experiments showed 7L/MLP4x gets 44% more training steps than 11L/MLP3x. The question is whether this transfers at full 600s budget on SXM hardware.

### Category B: Training Dynamics

| ID | Env Overrides | Description |
|----|--------------|-------------|
| **B1** | `MATRIX_LR=0.03 SCALAR_LR=0.03` | Muon LR 0.03 (autoresearch found 0.03 > 0.025) |
| **B2** | `WARMDOWN_ITERS=4500` | Longer warmdown (smoother quant transition) |
| **B3** | `WARMDOWN_ITERS=5000` | Even longer warmdown |
| **B4** | `BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112` | BigramHash 3072×112 (match submission.json) |
| **B5** | `MUON_WD=0.06 ADAM_WD=0.06` | Higher weight decay |
| **B6** | `HEAD_LR=0.01` | Higher unembedding LR |

**Rationale:** Autoresearch sweep found Muon LR 0.03 and higher head LR are slightly better. The SOTA submission.json says 3072×112 bigram but the script defaults to 2048×128 — B4 checks if the submission values actually matter.

### Category C: Eval-Time

No retraining needed — these only change the sliding window eval stride. Run on any trained checkpoint.

| ID | Env Overrides | Description |
|----|--------------|-------------|
| **C1** | `EVAL_STRIDE=32` | Sliding window stride 32 (2× more overlap) |
| **C2** | `EVAL_STRIDE=16` | Sliding window stride 16 (4× more overlap) |

**Rationale:** More overlap = better BPB but slower eval. PR #641 uses stride 16. SOTA uses stride 64. Need to check if the improvement justifies the eval time within the 10-minute eval budget.

### Category D: Combinations

Stack the winners from A + B + C. Run these after individual experiments show signal.

| ID | Env Overrides | Description |
|----|--------------|-------------|
| **D1** | `NUM_LAYERS=7 MLP_MULT=4.0 XSA_LAST_N=7 VE_LAYERS=5,6 MATRIX_LR=0.03 SCALAR_LR=0.03` | 7L/4x + Muon 0.03 |
| **D2** | `NUM_LAYERS=9 MLP_MULT=3.5 XSA_LAST_N=9 VE_LAYERS=7,8 MATRIX_LR=0.03 SCALAR_LR=0.03` | 9L/3.5x + Muon 0.03 |
| **D3** | `NUM_LAYERS=11 MLP_MULT=3.5 BIGRAM_VOCAB_SIZE=3072 BIGRAM_DIM=112` | 11L/3.5x + bigger bigram |
| **D4** | `NUM_LAYERS=7 MLP_MULT=4.0 XSA_LAST_N=7 VE_LAYERS=5,6 MATRIX_LR=0.03 SCALAR_LR=0.03 EVAL_STRIDE=16` | D1 + stride 16 eval |

---

## How the Pipeline Works

Each experiment runs the full SOTA pipeline:

1. **Train** — Parallel Muon + Adam optimizers, EMA + SWA weight averaging
2. **GPTQ int6** — Full Hessian quantization with AR self-generated calibration data
3. **LZMA compress** — Artifact must be ≤16MB
4. **Sliding window eval** — Final val_bpb score

No TTT — PR #1019 dropped it after 25 failed attempts. The GPTQ improvement more than compensated.

---

## What to Look For

1. **Training BPB** — printed every `VAL_LOSS_EVERY` steps. Lower is better.
2. **`final_int6_sliding_window_exact val_bpb:`** — this is the real score. Compare to SOTA 1.1147.
3. **`artifact_bytes`** — must be ≤16,000,000. If over, need more aggressive quantization.
4. **`step_avg`** — ms/step. Fewer layers = faster steps = more training. The depth/width tradeoff.

---

## Prior Results Summary (from v2 experiments)

These used `train_sota_exp.py` (a modified fork) with 90s training. Directional findings only:

| Finding | Evidence |
|---------|----------|
| **SwiGLU > LeakyReLU²** | 8% faster + better loss across all 3 tracks |
| **7L/MLP4x > 11L/MLP3x** | 44% more steps, best training + post-quant BPB |
| **Trigram hash is redundant** | No improvement on top of BigramHash 3072×112 |
| **LoRA r16 QVK best for TTT** | Marginal over bias-only TTT |
| **Muon LR 0.03 > 0.025** | Autoresearch sweep (36 experiments) |

**Key caveat:** These were run on a modified script with 90s budget. The v3 experiments use the unmodified SOTA script at full budget to get definitive results.

---

## Do NOT Run (Failed in Prior Tests)

| Experiment | Why Skip |
|-----------|----------|
| Residual gating | Destabilizes training (+0.13 bpb) |
| LN inverse sqrt | Aggressive decay hurts (+0.04 bpb) |
| Gram Newton-Schulz | Wrong optimizer (+0.61 bpb) |
| Causal conv (k=3) | No signal (+0.01 bpb) |
| Trigram hash (on SOTA) | Redundant with BigramHash 3072×112 |
