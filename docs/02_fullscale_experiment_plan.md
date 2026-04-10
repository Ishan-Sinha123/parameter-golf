# Full-Scale Experiment Plan (v3)

**Goal:** Beat current SOTA of 1.11473 val_bpb
**Script:** Unmodified SOTA `train_gpt.py` from `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/`
**Runner:** `./scripts/run_ablations.sh`
**Approach:** Every experiment runs the exact same SOTA code with different env var overrides. No code forks.

---

## Quick Start

### 1. Setup (Vast.ai / RunPod — 4×H100 SXM or 8×H100 SXM)

```bash
# As root on your GPU instance:
useradd -m -s /bin/bash dev
chmod -R 777 /root /root/parameter-golf  # or wherever the repo lives
ln -s /opt/nvm/versions/node/v24.14.1/bin/claude /usr/local/bin/claude
ln -s /opt/nvm/versions/node/v24.14.1/bin/node /usr/local/bin/node
su - dev

# As dev user:
cd /root/parameter-golf  # or clone fresh: git clone git@github.com:Ishan-Sinha123/parameter-golf.git
pip install -r requirements.txt
export HF_HOME=/home/dev/.cache/huggingface
python3 data/cached_challenge_fineweb.py --variant sp1024
```

### 2. Run SOTA baseline first — full eval (validate your setup)

```bash
# On 4 GPUs: double wallclock to match 8-GPU step count
NGPUS=4 MAX_WALLCLOCK_SECONDS=1200 ./scripts/run_ablations.sh 0

# On 8 GPUs: use default 600s
./scripts/run_ablations.sh 0
```

This reproduces PR #1019 exactly. Expected: ~1.1147 BPB on 8×H100 SXM (~6900 steps in 600s).
On 4×H100 SXM with 1200s: ~same step count, comparable BPB.
Includes full sliding window eval (~5 min on 4 GPUs).

### 3. Fast ablations (180s, no sliding window eval)

```bash
# Architecture experiments — most promising, run these first
NGPUS=4 MAX_WALLCLOCK_SECONDS=180 ./scripts/run_ablations.sh A

# Training dynamics
NGPUS=4 MAX_WALLCLOCK_SECONDS=180 ./scripts/run_ablations.sh B

# Eval stride changes — these DO run sliding window (that's what they test)
# Reuses training from experiment 0, so only eval cost
NGPUS=4 MAX_WALLCLOCK_SECONDS=1200 ./scripts/run_ablations.sh C
```

Sliding window eval is **automatically skipped** for A, B, and D experiments (except D4).
This saves ~5-10 min per run. Compare using the `Int6 BPB` column in the summary.

### 4. Combinations (after A/B/C show signal)

```bash
# Full budget for the best combos
NGPUS=4 MAX_WALLCLOCK_SECONDS=1200 ./scripts/run_ablations.sh D1 D2
```

### 5. Other usage

```bash
./scripts/run_ablations.sh A1 A3 B1           # cherry-pick specific experiments
FULL_EVAL=1 ./scripts/run_ablations.sh A1     # force sliding window eval on any experiment
DRY_RUN=1 ./scripts/run_ablations.sh          # print all commands without executing
```

Logs go to `experiment_logs/ablations/`. Summary table printed at end of each run.

### 6. Scaling: 4 GPUs vs 8 GPUs

The SOTA script handles this automatically via `grad_accum_steps = 8 // world_size`:
- **4 GPUs:** 2× gradient accumulation, same effective batch size, ~2× slower per step
- **8 GPUs:** 1× gradient accumulation, native speed

For ablations (relative ranking), use any GPU count — just keep it consistent across experiments.
For final submission numbers, use 8×H100 SXM with 600s.

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
| **B7** | `MLP_ACTIVATION=swiglu` | SwiGLU activation (strongest finding across all tracks) |
| **B8** | `LOGIT_SOFTCAP=15` | Softcap 15 (autoresearch optimal) |
| **B9** | `MUON_WD=0.2 ADAM_WD=0.2` | WD 0.2 (autoresearch optimal) |

**Rationale:** Autoresearch sweep found Muon LR 0.03 and higher head LR are slightly better. The SOTA submission.json says 3072×112 bigram but the script defaults to 2048×128 — B4 checks if the submission values actually matter. B7 tests SwiGLU — the strongest finding from all 92 prior experiments. Note: SwiGLU doubles the MLP up-projection parameters (gate+up concatenated). B8 tests softcap 15 (autoresearch optimal vs default 30). B9 tests WD 0.2 (autoresearch found 0.2 >> 0.04 default).

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
| **D5** | `NUM_LAYERS=7 MLP_MULT=4.0 MLP_ACTIVATION=swiglu XSA_LAST_N=7 VE_LAYERS=5,6 MATRIX_LR=0.03 SCALAR_LR=0.03` | 7L/4x + SwiGLU + Muon 0.03 |

---

## How the Pipeline Works

Each experiment runs the SOTA pipeline:

1. **Train** — Parallel Muon + Adam optimizers, EMA + SWA weight averaging
2. **GPTQ int6** — Full Hessian quantization with AR self-generated calibration data
3. **LZMA compress** — Artifact must be ≤16MB
4. **Int6 roundtrip eval** — Quick single-pass eval (~23s on 8 GPUs)
5. **Sliding window eval** — Full stride-64 eval (~105s on 8 GPUs) — **skipped for fast ablations**

No TTT — PR #1019 dropped it after 25 failed attempts. The GPTQ improvement more than compensated.

### Sliding Window Eval Policy

| Experiments | Sliding window? | Why |
|---|---|---|
| **0 (SOTA repro)** | Yes | Need full score for reference |
| **A1-A4 (architecture)** | No | Int6 roundtrip BPB is sufficient for ranking |
| **B1-B6 (training)** | No | Same — relative comparison only |
| **C1-C2 (stride)** | Yes | That's what they're testing |
| **D1-D3 (combos)** | No | Run with `FULL_EVAL=1` once you have a winner |
| **D4 (combo + stride)** | Yes | Stride experiment |

Override with `FULL_EVAL=1` to force sliding window on any experiment.

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
