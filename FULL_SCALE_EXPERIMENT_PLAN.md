# Full-Scale Experiment Plan

**Goal:** Beat current SOTA of 1.11473 val_bpb  
**Budget:** 10 min training, 16MB model, 8×H100  
**Base:** `train_gpt.py` (current best record) + modifications from `train_alpha.py`  

---

## Phase 1: Isolate Individual Gains (4 runs, ~40 min)

Run each winner independently against the current best to measure isolated impact.

### 1A. Baseline (Control)
Current best config, unmodified. Establishes the reference for this round.
```
# Default train_gpt.py settings
NUM_LAYERS=11  MLP_MULT=3  NUM_HEADS=8  NUM_KV_HEADS=4
```
**Expected:** ~1.1147 val_bpb (reproduces current SOTA)

### 1B. SwiGLU Activation
Replace ReLU² with SwiGLU in the MLP. Zero-cost activation swap.
```
NUM_LAYERS=11  MLP_MULT=3  MLP_ACTIVATION=swiglu
```
**Rationale:** SwiGLU scored 2.2455 vs 2.2729 baseline at scaled size. Better gradient flow from gating mechanism.

### 1C. Trigram Hash Embedding
Add parallel trigram hash embedding channel alongside the standard token embedding.
```
NUM_LAYERS=11  MLP_MULT=3  HASH_MODE=trigram
```
**Rationale:** Scored 2.2104 vs 2.2729 baseline — biggest single-feature improvement at scaled size. Adds n-gram context at minimal compute cost.

### 1D. LoRA r16 TTT Eval
Same training as baseline, but use rank-16 LoRA for test-time training eval instead of rank-8.
```
NUM_LAYERS=11  MLP_MULT=3  TTT_LORA_RANK=16
```
**Rationale:** Eval-only comparison showed r16 (3.3848) > r8 (3.3921) > r32 (3.3881). r16 is the sweet spot.

---

## Phase 2: Depth/Width Tradeoff (3 runs, ~30 min)

The scaled experiments showed massive gains from reducing depth and increasing width. Test whether this holds at full scale.

### 2A. Conservative Shallow (9L/MLP4x)
Modest depth reduction, significant width increase.
```
NUM_LAYERS=9  MLP_MULT=4  NUM_HEADS=8  NUM_KV_HEADS=4
XSA_LAST_N=9  VE_LAYERS="7,8"
```
**Rationale:** Small depth reduction to test if the depth/width tradeoff direction holds at full scale.

### 2B. Moderate Shallow (8L/MLP4x)
Larger depth reduction with maximum width.
```
NUM_LAYERS=8  MLP_MULT=4  NUM_HEADS=8  NUM_KV_HEADS=4
XSA_LAST_N=8  VE_LAYERS="6,7"
```
**Rationale:** At scaled size, 4L/MLP4x crushed 6L/MLP2x (2.0058 vs 2.2729). This is the full-scale equivalent.

### 2C. Moderate with MLP3x (9L/MLP3x)
Test whether 3x expansion at reduced depth is better than 4x.
```
NUM_LAYERS=9  MLP_MULT=3  NUM_HEADS=8  NUM_KV_HEADS=4
XSA_LAST_N=9  VE_LAYERS="7,8"
```
**Rationale:** MLP4x at same depth hurt in scaled tests (slower steps). 9L/MLP3x balances speed and capacity.

---

## Phase 3: Combined Winners (3 runs, ~30 min)

Stack the best individual features from Phase 1 onto the best architecture from Phase 2.

### 3A. Best Arch + SwiGLU + Trigram Hash
```
NUM_LAYERS=<best from Phase 2>  MLP_MULT=<best from Phase 2>
MLP_ACTIVATION=swiglu  HASH_MODE=trigram  TTT_LORA_RANK=16
XSA_LAST_N=<match layers>  VE_LAYERS="<last 2>"
```
**Rationale:** Stack all three independent wins. If gains are additive, this should beat SOTA.

### 3B. Best Arch + SwiGLU + Trigram Hash + LeakyReLU² fallback
If SwiGLU doesn't work at full scale, try LeakyReLU² instead:
```
NUM_LAYERS=<best from Phase 2>  MLP_MULT=<best from Phase 2>
MLP_ACTIVATION=leaky_relu2  HASH_MODE=trigram  TTT_LORA_RANK=16
```

### 3C. Current SOTA Arch + All Features (no depth change)
Conservative combo — keep 11L but add SwiGLU + trigram hash + r16.
```
NUM_LAYERS=11  MLP_MULT=3
MLP_ACTIVATION=swiglu  HASH_MODE=trigram  TTT_LORA_RANK=16
```
**Rationale:** If depth/width gains don't transfer to full scale, this captures the activation + embedding wins on the proven architecture.

---

## Phase 4: Fine-Tuning the Winner (2-3 runs, ~30 min)

Take the best config from Phase 3 and tune hyperparameters.

### 4A. Learning Rate Sweep
Run the best config at 0.8× and 1.2× the default learning rate.

### 4B. Warmdown Schedule
Adjust warmdown iterations proportional to the new step count (shorter warmdown if more steps from shallower model).

### 4C. Batch Size Tuning
If the shallower model is significantly faster per step, try increasing batch size to use more tokens per step instead of just doing more steps.

---

## Decision Tree

```
Phase 1 results:
├─ SwiGLU helps?     → use in Phase 3
├─ Trigram helps?    → use in Phase 3
├─ r16 helps?        → use in Phase 3
│
Phase 2 results:
├─ 9L/MLP4x best?   → use as base in Phase 3
├─ 8L/MLP4x best?   → use as base in Phase 3
├─ 9L/MLP3x best?   → use as base in Phase 3
├─ None beat 11L?    → use 11L in Phase 3 (skip 2-series)
│
Phase 3 results:
├─ Combined > SOTA?  → proceed to Phase 4 tuning
├─ Combined ≤ SOTA?  → try individual features on 11L only
```

---

## Run Order (Priority)

| Order | Run | Est. Time | Why First |
|-------|-----|-----------|-----------|
| 1 | 1A Baseline | 10 min | Establish reference |
| 2 | 1C Trigram Hash | 10 min | Biggest isolated signal |
| 3 | 1B SwiGLU | 10 min | Second biggest signal |
| 4 | 2B 8L/MLP4x | 10 min | Biggest potential upside |
| 5 | 2A 9L/MLP4x | 10 min | Backup if 8L too aggressive |
| 6 | 3A Combined | 10 min | Stack winners |
| 7 | 3C Safe Combined (11L) | 10 min | Fallback if depth change fails |
| 8 | 1D LoRA r16 | 10 min | Eval-only, run on best checkpoint |
| 9-11 | Phase 4 tuning | 30 min | Polish the winner |

**Total estimated wall time:** ~2 hours sequential, ~40 min if parallelized across GPUs.

---

## Notes

- All TTT variants (LoRA, FFT, bias) produce identical training curves. Only test TTT config changes on the final checkpoint, not as separate training runs.
- The depth/width tradeoff is the biggest unknown. At scaled size the effect was enormous (2.00 vs 2.27), but full-scale models with 8-GPU parallelism may behave differently.
- Trigram hash and SwiGLU are low-risk additions — they helped at scaled size with minimal speed overhead.
- Do NOT run: residual gating, LN schedules, Gram Newton-Schulz, causal conv, deep narrow. All hurt or showed no signal in scaled tests.
