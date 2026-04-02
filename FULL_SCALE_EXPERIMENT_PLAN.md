# Full-Scale Experiment Plan

**Goal:** Beat current SOTA of 1.11473 val_bpb  
**Setup:** 90s training + full TTT eval, 8×H100, `train_alpha.py`  
**Time budget:** ~2 hours GPU rental  
**Script:** `./scripts/run_fullscale.sh`  

See [RESEARCH_LOG.md](RESEARCH_LOG.md) for scaled ablation results that justify these experiments.

---

## Quick Start

```bash
source .venv/bin/activate
./scripts/run_fullscale.sh          # all 4 phases (~2 hours)
./scripts/run_fullscale.sh 1 2 3    # architecture only (~25 min)
./scripts/run_fullscale.sh 4        # TTT comparison only (~90 min)
```

---

## Phase 1: Isolate Features (~10 min, 5 runs)

Test each winning feature independently at full model scale. All use `TTT_MODE=none` (no post-training eval) for fast iteration.

| Run | Change | Justification |
|-----|--------|---------------|
| `p1_baseline` | 11L/MLP3x, defaults | Control. Establishes full-scale reference on train_alpha.py. |
| `p1_swiglu` | `MLP_ACTIVATION=swiglu` | SwiGLU's gating mechanism gave -0.027 bpb at scaled size. Widely used in modern LLMs (LLaMA, Mistral). Zero-cost swap — same parameter count, better gradient flow. |
| `p1_trigram_hash` | `HASH_MODE=trigram` | Biggest single-feature win in scaled tests (-0.063 bpb). Adds n-gram context via parallel hash embedding channel. Minimal speed overhead (~2% slower). |
| `p1_swiglu_trigram` | Both above | Tests whether gains stack. If additive, expect ~-0.09 bpb combined. |
| `p1_leaky_relu2` | `MLP_ACTIVATION=leaky_relu2` | Prevents dead neurons at zero speed cost. Small but free win (-0.008 bpb). Fallback if SwiGLU doesn't transfer to full scale. |

---

## Phase 2: Depth/Width Tradeoff (~8 min, 4 runs)

MLPs are where the model does its "thinking" — pattern matching across the residual stream. With 512-dim and 3× expansion, each MLP has 1536 intermediate neurons. Pushing to 4× gives 2048 — roughly 4 "lenses" per input dimension. Given the entropy of internet text, there are far more learnable patterns than 2048 neurons can capture, so wider should help.

The catch: wider MLPs mean slower steps. The scaled ablations showed this is solved by trading depth for width — fewer layers means faster steps, and the wider MLP more than compensates.

**The big unknown:** Does this hold at full scale with 8-GPU parallelism? The step-time calculus changes with data-parallel training.

| Run | Config | Justification |
|-----|--------|---------------|
| `p2_9L_mlp4x` | 9 layers, MLP 4× | Conservative: drop 2 layers, widen MLP. At scaled size, reducing from 6L→4L with MLP 4× gave a 0.27 bpb win. This is the cautious equivalent. |
| `p2_8L_mlp4x` | 8 layers, MLP 4× | Moderate reduction. 8L gives 2048 MLP neurons per layer × 8 layers. Should be ~15-20% faster per step than 11L. |
| `p2_9L_mlp3x` | 9 layers, MLP 3× | Isolates the depth effect. If 9L/MLP3x beats 11L/MLP3x, we know depth reduction alone helps (independent of width). |
| `p2_7L_mlp4x` | 7 layers, MLP 4× | Aggressive. Closest to the scaled-test optimum (4L at 6L scale ≈ 7L at 11L scale). High risk, high reward. |

---

## Phase 3: Combined Winners (~8 min, 4 runs)

Stack best features from Phase 1 onto best architecture from Phase 2.

| Run | Config | Justification |
|-----|--------|---------------|
| `p3_9L_mlp4x_swiglu_trigram` | 9L/MLP4x + SwiGLU + trigram | Full combo on conservative reduced-depth arch. If all three gains are independent, this should meaningfully beat baseline. |
| `p3_8L_mlp4x_swiglu_trigram` | 8L/MLP4x + SwiGLU + trigram | Same combo, more aggressive depth. Tests whether the extra speed from fewer layers outweighs capacity loss. |
| `p3_11L_swiglu_trigram` | 11L/MLP3x + SwiGLU + trigram | Safe fallback. Keeps SOTA depth, adds only feature wins. Run this if Phase 2 shows no depth/width gain at full scale. |
| `p3_9L_mlp4x_leaky_trigram` | 9L/MLP4x + LeakyReLU² + trigram | Fallback if SwiGLU fails at full scale. LeakyReLU² was a smaller but more robust win. |

---

## Phase 4: TTT — LoRA vs FFT + Rank Sweep (~90 min, 11 runs)

All runs train the **same** 11L/MLP3x base model. They differ only in post-training TTT evaluation. Each run = 90s training + 10-15 min eval.

### LoRA Rank Sweep

**Justification:** Scaled eval showed r16 > r8 > r32. At full scale with a larger model, the optimal rank may shift — more model capacity could benefit from higher-rank adapters. Need to re-verify.

| Run | Config | Notes |
|-----|--------|-------|
| `p4_lora_r4` | `TTT_LORA_RANK=4` | Minimal adapter. Tests if even small adapters help. |
| `p4_lora_r8` | `TTT_LORA_RANK=8` | Current competition default. |
| `p4_lora_r16` | `TTT_LORA_RANK=16` | Best at scaled size. Expected winner. |
| `p4_lora_r32` | `TTT_LORA_RANK=32`, batch=32 | Higher capacity but risks per-document overfitting. Reduced batch to avoid OOM. |

### FFT (Full Fine-Tuning)

**Justification:** FFT adapts the actual model weights per document rather than low-rank projections. With a full-scale 11L model, FFT on the last few layers has much more capacity than LoRA — it may outperform LoRA at this scale even if it didn't at scaled size.

| Run | Config | Notes |
|-----|--------|-------|
| `p4_fft_last2` | `TTT_MODE=fft2` | Lightweight: fine-tune only last 2 layers. |
| `p4_fft_last4` | `TTT_MODE=fft4` | More adaptation surface. |
| `p4_fft_all` | `TTT_MODE=fft_all` | Maximum: fine-tune entire model per doc. Slowest eval. |

### LoRA Variations

**Justification:** Once we know the best rank (likely r16), these test whether more aggressive adaptation strategies extract additional signal.

| Run | Config | Rationale |
|-----|--------|-----------|
| `p4_lora_r16_3step` | r16 + `TTT_STEPS=3` | Multiple gradient steps per chunk. More compute at eval time → better per-document fit? |
| `p4_lora_r16_chunk128` | r16 + `TTT_CHUNK_SIZE=128` | Smaller adaptation windows. Trades global coherence for finer-grained local fit. |
| `p4_lora_r16_qvk` | r16 + `TTT_LORA_TARGETS=qvk` | Adapt K projections too (default is Q+V only). More projection surfaces for the adapter. |
| `p4_bias_ttt` | `TTT_MODE=bias` | Minimal overhead baseline. If bias-only matches LoRA, the adapter capacity doesn't matter — it's the adaptation signal itself. |

---

## Decision Tree

```
Phase 1 → Which features transfer to full scale?
  ├─ SwiGLU helps       → include in Phase 3 combos
  ├─ Trigram helps      → include in Phase 3 combos
  └─ Neither helps      → use baseline activation/embedding

Phase 2 → Does shallow+wide survive at 8×H100?
  ├─ YES (any <11L beats 11L) → use that depth in Phase 3
  └─ NO                        → keep 11L, use only feature wins

Phase 3 → Do gains stack?
  ├─ Combined > parts   → new best architecture recipe
  └─ Combined ≤ parts   → use single best feature on 11L

Phase 4 → Best TTT strategy?
  └─ Apply to best architecture → final competition recipe
```

---

## Priority If Short on Time

| Priority | Command | Time | What You Get |
|----------|---------|------|--------------|
| 1 | `./scripts/run_fullscale.sh 1 2` | ~18 min | Feature + depth/width answer |
| 2 | `./scripts/run_fullscale.sh 3` | +8 min | Combined winner |
| 3 | `./scripts/run_fullscale.sh 4` (LoRA r4/r8/r16/r32 only) | +40 min | TTT rank answer |
| 4 | Remaining Phase 4 | +50 min | FFT comparison + variations |

**Minimum viable:** Phases 1+2+3 in 25 min gives the architecture answer.

---

## Do NOT Run (Failed in Scaled Tests)

| Experiment | Why Skip |
|-----------|----------|
| Residual gating | Destabilizes training (+0.13 bpb) |
| LN inverse sqrt | Aggressive decay hurts (+0.04 bpb) |
| Gram Newton-Schulz | Wrong optimizer entirely (+0.61 bpb) |
| Causal conv (k=3) | No signal after bug fix (+0.01 bpb) |
| Deeper models (>11L) | More depth = slower steps = fewer total steps |
| MLP 4× at same depth | Slower steps offset capacity (+0.07 bpb) |
| Residual sqrt | Numerically unstable (NaN) |
