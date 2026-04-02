# Parameter Golf: Scaled Ablation Study & Full-Scale Experiment Plan

> **Ishan Sinha** — April 2026  
> Current SOTA: 1.11473 val_bpb | Budget: 90s training, 16MB model, 8×H100

---

## Motivation

The Parameter Golf leaderboard is dominated by 11-layer, 3× MLP expansion models. But is that the right shape? MLPs are where the model does its "thinking" — pattern matching across the residual stream. With a 512-dim model and 3× expansion, each MLP has 1536 intermediate neurons. Pushing to 4× gives 2048 — roughly 4 "lenses" per input dimension. Given the entropy of internet text, there are far more patterns than 2048 neurons can capture, so wider should help.

The catch: wider MLPs mean slower steps. At a fixed 90-second wall clock, every millisecond per step matters. The question isn't just "is 4× better per step?" but "is 4× better per second?"

We ran 30 scaled-down experiments to find out.

---

## Scaled Ablation Setup

- **Model:** 6 layers, 384 dim, MLP 2×, seq 512 (scaled from 11L/512d/MLP3x)
- **Training:** 90 seconds per run, ~160-200 steps
- **Hardware:** 1× GPU with `TORCHDYNAMO_DISABLE=1`
- **Metrics:** val_bpb every 10 steps

This small model trains fast enough to see real curve separation in 90 seconds — the full-size model only manages ~50 steps in the same window, too few to differentiate ablations.

---

## Key Result 1: Width Beats Depth

![MLP Expansion Comparison](experiment_logs_scaled/plots/mlp_expansion_comparison.png)

The left panel shows per-step efficiency — all configs learn at similar rates per gradient update. The right panel shows what actually matters: **val_bpb vs wall time**. Shallow wide (4L/MLP4x, red) crushes everything else.

| Config | Final Val BPB | ms/step | Steps in 90s |
|--------|--------------|---------|--------------|
| **4L / MLP 4× (shallow wide)** | **2.0058** | **420** | **~150** |
| 6L / MLP 2× (baseline) | 2.2729 | 548 | ~110 |
| 6L / MLP 3× | 2.2820 | 570 | ~105 |
| 6L / MLP 4× | 2.3393 | 610 | ~100 |
| 8L / MLP 2× (deep narrow) | 2.5372 | 720 | ~80 |

**The takeaway:** MLP 4× at the same depth *hurts* (slower steps offset capacity). But reducing depth to compensate gives a massive win — 36% more training steps, and the wider MLP more than makes up for lost depth. This aligns with the literature: wider models outperform deeper ones at fixed compute.

---

## Key Result 2: Architecture Feature Comparison

![Architecture Winners](experiment_logs_scaled/plots/architecture_winners.png)

**Trigram hash embedding** was the single best feature addition (+0.063 bpb). It adds a parallel embedding channel that hashes 3-token windows, giving the model n-gram context essentially for free.

**SwiGLU activation** (replacing ReLU²) also helped, especially paired with MLP 3× expansion (2.2455 vs 2.2729 baseline).

---

## Key Result 3: TTT Variants Are Training-Identical

![TTT LoRA vs FFT](experiment_logs_scaled/plots/ttt_lora_vs_fft.png)

All Test-Time Training variants (LoRA rank 8/16/32, FFT 2L/4L/all, bias-only) produce **identical training curves**. This is expected — TTT only modifies the post-training evaluation pass, not the training itself. The 9 TTT experiments in Phase 2-3 all trained the exact same model.

From a separate eval-only comparison on the trained checkpoint:
- **LoRA r16: 3.3848** (best)
- LoRA r8: 3.3921
- LoRA r32: 3.3881

r16 is the sweet spot — r32 likely overfits per-document.

---

## Full Results

![All Experiments Bar Chart](experiment_logs_scaled/plots/final_bpb_bar_chart.png)

**Red** = MLP/depth variants | **Blue** = TTT variants (same training) | **Green** = embedding variants

---

## What Failed

| Experiment | Result | Why |
|-----------|--------|-----|
| Residual gating | 2.4063 | Extra parameters destabilize training |
| LN inverse sqrt schedule | 2.3174 | Too aggressive gain decay |
| Gram Newton-Schulz optimizer | 2.8827 | Wrong optimizer for this regime entirely |
| Residual sqrt scaling | NaN | Numerically unstable from step 1 |
| Causal conv (k=3) | 2.2861 | Marginal cost, no benefit after fixing a causal masking bug |
| Deep narrow (8L/MLP2x) | 2.5372 | Slower steps → fewer total steps → worse |

---

## Full-Scale Experiment Plan

Based on these findings, here's the plan for 8×H100 (2-hour rental).

### Quick Start

```bash
source .venv/bin/activate
./scripts/run_fullscale.sh          # all phases (~2 hours)
./scripts/run_fullscale.sh 1 2 3    # architecture only (~25 min)
./scripts/run_fullscale.sh 4        # TTT comparison only (~90 min)
```

### Phase 1: Isolate Features (~10 min)

Test each winning feature at full scale (11L/512d/MLP3x). No TTT eval.

| Run | Change | Scaled Signal |
|-----|--------|---------------|
| `p1_baseline` | Control | — |
| `p1_swiglu` | SwiGLU activation | -0.027 bpb |
| `p1_trigram_hash` | Trigram hash embedding | -0.063 bpb |
| `p1_swiglu_trigram` | Both combined | stackable? |
| `p1_leaky_relu2` | LeakyReLU² | -0.008 bpb |

### Phase 2: Depth/Width Tradeoff (~8 min)

The big question: does the shallow-wide advantage survive at full scale with 8-GPU parallelism?

| Run | Config | Rationale |
|-----|--------|-----------|
| `p2_9L_mlp4x` | 9 layers, MLP 4× | Conservative: 2 fewer layers, wider MLP |
| `p2_8L_mlp4x` | 8 layers, MLP 4× | Moderate: 3 fewer layers |
| `p2_9L_mlp3x` | 9 layers, MLP 3× | Test depth alone (no width increase) |
| `p2_7L_mlp4x` | 7 layers, MLP 4× | Aggressive: near the scaled-test optimum |

**What matters:** Compare val_bpb AND ms/step. The win only works if faster steps translate to enough extra training.

### Phase 3: Combined Winners (~8 min)

Stack best features from Phase 1 + best architecture from Phase 2.

| Run | Config |
|-----|--------|
| `p3_9L_mlp4x_swiglu_trigram` | Best arch + SwiGLU + trigram |
| `p3_8L_mlp4x_swiglu_trigram` | Alt arch + SwiGLU + trigram |
| `p3_11L_swiglu_trigram` | Safe: SOTA depth + features |
| `p3_9L_mlp4x_leaky_trigram` | Fallback: LeakyReLU² if SwiGLU fails |

### Phase 4: TTT — LoRA vs FFT (~90 min)

All train the same 11L/MLP3x model. Difference is post-training eval only. Each run = 90s training + 10-15 min TTT eval.

**LoRA Rank Sweep:**
| Run | Rank | From Scaled Tests |
|-----|------|-------------------|
| `p4_lora_r4` | 4 | — |
| `p4_lora_r8` | 8 | 3.3921 (baseline) |
| `p4_lora_r16` | 16 | 3.3848 (best) |
| `p4_lora_r32` | 32 | 3.3881 (overfits) |

**FFT (Full Fine-Tuning):**
| Run | Layers | Notes |
|-----|--------|-------|
| `p4_fft_last2` | Last 2 | Lightweight fine-tune |
| `p4_fft_last4` | Last 4 | More capacity |
| `p4_fft_all` | All 11 | Maximum adaptation |

**LoRA Variations (on r16):**
| Run | Variation | Why |
|-----|-----------|-----|
| `p4_lora_r16_3step` | 3 gradient steps/chunk | More adaptation per window |
| `p4_lora_r16_chunk128` | Chunk size 128 (vs 256) | Finer-grained windows |
| `p4_lora_r16_qvk` | Adapt Q+V+K (vs Q+V) | More projection surfaces |
| `p4_bias_ttt` | Bias-only | Minimal overhead baseline |

---

## Decision Tree

```
Phase 1 → Which features help at full scale?
  ├─ SwiGLU helps    → include in combos
  ├─ Trigram helps   → include in combos
  └─ Neither helps   → stick with baseline activation/embedding

Phase 2 → Does shallow+wide survive?
  ├─ YES (any <11L beats 11L) → use that depth going forward
  └─ NO                        → keep 11L, only use feature wins

Phase 3 → Do gains stack?
  ├─ Combined > parts → new best recipe found
  └─ Combined ≤ parts → use single best feature on 11L

Phase 4 → Best TTT strategy?
  └─ Apply winner to best architecture → final recipe
```

---

## If Short on Time

| Priority | What | Time | Why |
|----------|------|------|-----|
| 1 | `./scripts/run_fullscale.sh 1 2 3` | 25 min | Full architecture answer |
| 2 | Add Phase 4 LoRA sweep (r4/r8/r16/r32) | +40 min | TTT rank answer |
| 3 | Add Phase 4 FFT runs | +30 min | LoRA vs FFT answer |
| 4 | Phase 4 variations | +40 min | Diminishing returns |

---

## Files

| File | Purpose |
|------|---------|
| `scripts/run_fullscale.sh` | Full-scale runner (8×H100, 90s + TTT eval) |
| `train_alpha.py` | Unified experiment script |
| `scripts/plot_experiments.py` | Comparison plot generator |
| `experiment_logs_scaled/` | Scaled ablation results + plots |
| `experiment_logs_fullscale/` | Full-scale results (created on run) |
