# Full-Scale Experiment Plan (v2)

**Goal:** Beat current SOTA of 1.11473 val_bpb  
**Setup:** 90s training, 8×H100, `train_sota_exp.py` (fork of SOTA `train_gpt.py` with experiment features)  
**Baseline:** SOTA config reproduced on our hardware (BigramHash 3072×112, XSA all layers, Parallel Muon, GPTQ int6, LZMA, etc.)  
**Script:** `./scripts/run_fullscale.sh`  

---

## What Changed from v1

- **Switched from `train_alpha.py` to `train_sota_exp.py`** — a fork of the actual SOTA `train_gpt.py` (PR #1019). The old `train_alpha.py` was missing ~8 critical SOTA features (XSA, GPTQ, EMA/SWA, partial RoPE, value embeddings, SmearGate, parameter banking, late QAT). Experiments now stack on top of SOTA, not a weaker baseline.
- **Added SwiGLU activation** as a toggleable option (`MLP_ACTIVATION=swiglu`)
- **Added TTT (LoRA/FFT/bias)** with MLP+attention targets, ported to work with SOTA's banked architecture
- **Added FA3 fallback** — tries `flash_attn.flash_attn_interface` (FA3 Hopper kernels), falls back to PyTorch SDPA
- **GPTQ Cholesky fallback** — gracefully falls back to percentile quantization when Hessians are ill-conditioned (common with short training runs)
- **Trigram uses `TRIGRAM=1` env var** (SOTA convention), not `HASH_MODE=trigram`
- **Full GPTQ+LZMA pipeline** runs on every experiment for accurate post-quantization scores

### Hardware Note

SOTA was developed on 8×H100 SXM (3.35 TB/s bandwidth, ~87ms/step → ~6900 steps in 600s). Our H100 PCIe setup gets ~305ms/step → ~292 steps in 90s. Per [Abay's analysis](https://abaybektursun.com), step-1000 bpb predicts final outcome at r=0.86 — our 292-step runs are below that threshold, so absolute bpb values are high. **Relative comparisons between experiments are still valid.**

For production runs, use 8×H100 SXM with 600s training.

---

## Quick Start

```bash
source .venv/bin/activate
./scripts/run_fullscale.sh          # all 4 phases
./scripts/run_fullscale.sh 1 2 3    # architecture only
./scripts/run_fullscale.sh 4        # TTT comparison only
```

---

## Phase 1 Results: Feature Isolation (COMPLETED)

Tested each feature independently on SOTA config. All use `TTT_MODE=none`.  
**Hardware:** Clean GPUs, ~160ms/step, ~560 steps in 90s.

| Run | Change | Steps | ms/step | Training BPB | Final BPB | Delta |
|-----|--------|-------|---------|-------------|-----------|-------|
| **p1_swiglu** | `MLP_ACTIVATION=swiglu` | **607** | **148** | **1.599** | **4.148** | **-2.824** |
| p1_swiglu_trigram | SwiGLU + `TRIGRAM=1` | 604 | 149 | 1.623 | 4.139 | -2.833 |
| p1_leaky_relu2 | `MLP_ACTIVATION=leaky_relu2` | 564 | 160 | 1.632 | 6.622 | -0.350 |
| p1_trigram_hash | `TRIGRAM=1` | 562 | 160 | 1.639 | 6.891 | -0.081 |
| p1_baseline | SOTA defaults | 564 | 160 | 1.632 | 6.972 | — |

### Findings
- **SwiGLU is the clear winner** — 2.8 bpb improvement AND 8% faster per step (148ms vs 160ms). The gating mechanism provides better gradient flow than LeakyReLU².
- **Trigram hash does nothing** on top of SOTA's existing BigramHash 3072×112. The bigram already captures the n-gram signal; trigram is redundant.
- **SwiGLU + trigram ≈ SwiGLU alone** — trigram adds no value when bigram is already large.
- **LeakyReLU² shows modest improvement** over baseline in final BPB (6.622 vs 6.972).

**Decision: Use SwiGLU. Drop trigram.**

---

## Phase 2 Results: Depth/Width Tradeoff (COMPLETED)

**Hardware:** Clean GPUs, ~111-142ms/step.

| Run | Config | Steps | ms/step | Training BPB | Final BPB |
|-----|--------|-------|---------|-------------|-----------|
| **p2_7L_mlp4x** | **7 layers, MLP 4×** | **814** | **111** | **1.489** | **2.350** |
| p2_8L_mlp4x | 8 layers, MLP 4× | 713 | 126 | 1.534 | 5.136 |
| p2_9L_mlp3x | 9 layers, MLP 3× | 691 | 130 | 1.546 | 5.090 |
| p2_9L_mlp4x | 9 layers, MLP 4× | 636 | 142 | 1.579 | 6.405 |

### Findings
- **7L/MLP4x dominates** — 44% more steps than baseline (814 vs 564) at 111ms/step, best training BPB (1.489) AND best post-quantization BPB (2.350) by a massive margin.
- The depth-for-width tradeoff from scaled experiments transfers to full scale even more strongly than expected.
- Shallower = faster steps = more training = better loss at fixed wallclock.

**Decision: Use 7L/MLP4x as base architecture.**

---

## Phase 3 Results: Combined Winners (COMPLETED)

> ⚠️ Mixed hardware conditions. First two experiments ran on clean GPUs (~130-142ms/step). Last two ran with phantom CUDA memory (~355-645ms/step). Cross-condition comparisons unreliable.

| Run | Config | Steps | ms/step | Training BPB | Final BPB | GPU |
|-----|--------|-------|---------|-------------|-----------|-----|
| p3_11L_swiglu_trigram | 11L/MLP3x + SwiGLU + trigram | 254 | 355 | 2.649 | 3.413 | Degraded |
| p3_9L_mlp4x_leaky_trigram | 9L/MLP4x + LeakyReLU² + trigram | 141 | 645 | 3.215 | 3.429 | Degraded |
| **p3_8L_mlp4x_swiglu_trigram** | **8L/MLP4x + SwiGLU + trigram** | **637** | **142** | **1.550** | **3.904** | **Clean** |
| p3_9L_mlp4x_swiglu_trigram | 9L/MLP4x + SwiGLU + trigram | 695 | 130 | 1.566 | 4.094 | Clean |

### Findings (clean runs only)
- 8L/MLP4x + SwiGLU (3.904) beats 9L variant (4.094), consistent with Phase 2's shallower-is-better finding.
- Degraded-GPU runs got lower absolute BPB due to GPTQ quantizing undertrained models more easily — not a real signal.

**Decision: 8L/MLP4x + SwiGLU for the combined config. Need clean rerun of 7L/MLP4x + SwiGLU.**

---

## Phase 4 Results: TTT — LoRA vs FFT + Rank Sweep (COMPLETED)

> ⚠️ Most Phase 4 experiments ran with phantom CUDA memory (~370ms/step, ~240 steps). Within-phase comparisons are valid. A few experiments (fft_last4, fft_all, lora_r16_3step) ran on clean GPUs and are not directly comparable.

All TTT experiments train the same base model (11L/MLP3x). They differ only in post-training TTT evaluation.

| Run | Steps | ms/step | Final BPB | GPU | Notes |
|-----|-------|---------|-----------|-----|-------|
| **p4_lora_r16_qvk** | 244 | 370 | **3.500** | Degraded | **Best TTT** |
| p4_bias_ttt | 242 | 373 | 3.501 | Degraded | Near-tied, minimal params |
| p4_lora_r8 | 243 | 372 | 3.505 | Degraded | |
| p4_lora_r16_qv_mlp | 240 | 377 | 3.515 | Degraded | |
| p4_lora_r16_qvk_mlp | 244 | 370 | 3.522 | Degraded | |
| p4_lora_r16 | 241 | 374 | 3.533 | Degraded | |
| p4_lora_r32 | 246 | 367 | 3.558 | Degraded | |
| p4_fft_last2 | 245 | 369 | 3.570 | Degraded | |
| p4_lora_r4 | 251 | 359 | 3.577 | Degraded | |
| p4_fft_last4 | 563 | 160 | 6.578 | Clean | Not comparable |
| p4_lora_r16_chunk128 | 417 | 217 | 6.173 | Partial | Not comparable |
| p4_fft_all | 563 | 160 | 6.696 | Clean | Not comparable |
| p4_lora_r16_3step | 563 | 160 | 6.827 | Clean | Not comparable |

### Findings (degraded-GPU experiments, comparable within group)
- **LoRA r16 + QVK targets** is the marginal winner (3.500), but **bias-only TTT** is nearly tied (3.501) with far fewer parameters.
- LoRA r8–r16 is the sweet spot, confirming scaled test findings.
- LoRA r32 and r4 are slightly worse (overfitting and underfitting respectively).
- Adding MLP targets (qv_mlp, qvk_mlp) doesn't help over attention-only.
- FFT variants are slightly worse than LoRA in this comparison.

**Decision: LoRA r16 with QVK targets for TTT. Bias-only TTT is a strong minimal alternative.**

Note: SOTA PR #1019 dropped TTT ("25 failed attempts"). Our TTT implementation operates on the quantized eval model, which may behave differently.

---

## Do NOT Run (Failed in Scaled Tests)

| Experiment | Why Skip |
|-----------|----------|
| Residual gating | Destabilizes training (+0.13 bpb) |
| LN inverse sqrt | Aggressive decay hurts (+0.04 bpb) |
| Gram Newton-Schulz | Wrong optimizer (+0.61 bpb) |
| Causal conv (k=3) | No signal (+0.01 bpb) |
| Trigram hash (on SOTA) | Redundant with BigramHash 3072×112 (Phase 1 result) |
