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

| Run | Change | Training BPB | Final BPB | Steps | Delta |
|-----|--------|-------------|-----------|-------|-------|
| **p1_swiglu** | `MLP_ACTIVATION=swiglu` | **1.958** | **3.694** | 306 | **-0.560** |
| p1_swiglu_trigram | SwiGLU + `TRIGRAM=1` | 1.967 | 3.723 | 305 | -0.531 |
| p1_leaky_relu2 | `MLP_ACTIVATION=leaky_relu2` | 2.029 | 4.247 | 291 | -0.007 |
| p1_trigram_hash | `TRIGRAM=1` | 2.035 | 4.253 | -0.001 |
| p1_baseline | SOTA defaults | 2.028 | 4.254 | 292 | — |

### Findings
- **SwiGLU is the clear winner** — 0.56 bpb improvement AND 5% faster per step (294ms vs 305ms). The gating mechanism provides better gradient flow than LeakyReLU².
- **Trigram hash does nothing** on top of SOTA's existing BigramHash 3072×112. The bigram already captures the n-gram signal; trigram is redundant.
- **SwiGLU + trigram is slightly worse than SwiGLU alone** — trigram adds noise when bigram is already large.
- **LeakyReLU² ≈ baseline** — SOTA's default activation is already near-optimal within the ReLU² family.

**Decision: Use SwiGLU. Drop trigram.**

---

## Phase 2: Depth/Width Tradeoff (IN PROGRESS)

| Run | Config | Justification |
|-----|--------|---------------|
| `p2_9L_mlp4x` | 9 layers, MLP 4× | Conservative: drop 2 layers, widen MLP |
| `p2_8L_mlp4x` | 8 layers, MLP 4× | Moderate reduction |
| `p2_9L_mlp3x` | 9 layers, MLP 3× | Isolate depth effect |
| `p2_7L_mlp4x` | 7 layers, MLP 4× | Aggressive shallow |

**Early signal:** p2_9L_mlp4x gets 330 steps (vs 292 baseline) with training bpb 1.942 (best so far), but post-quantization bpb is poor (4.61) due to GPTQ Cholesky fallback on undertrained model. The training signal is strong — this config may shine with 600s training.

---

## Phase 3: Combined Winners

Stack best features from Phase 1 onto best architecture from Phase 2.

| Run | Config | Justification |
|-----|--------|---------------|
| `p3_9L_mlp4x_swiglu_trigram` | 9L/MLP4x + SwiGLU + trigram | Full combo (trigram kept for completeness) |
| `p3_8L_mlp4x_swiglu_trigram` | 8L/MLP4x + SwiGLU + trigram | More aggressive depth |
| `p3_11L_swiglu_trigram` | 11L/MLP3x + SwiGLU + trigram | Safe: SOTA depth + SwiGLU |
| `p3_9L_mlp4x_leaky_trigram` | 9L/MLP4x + LeakyReLU² + trigram | Fallback |

**Given Phase 1 results, the most promising combo is SwiGLU + reduced depth (9L/MLP4x).** If Phase 2 confirms depth/width helps, Phase 3 should produce the best architecture.

---

## Phase 4: TTT — LoRA vs FFT + Rank Sweep

All runs train the same base model. They differ only in post-training TTT evaluation.

### LoRA Rank Sweep
| Run | Config |
|-----|--------|
| `p4_lora_r4` | `TTT_LORA_RANK=4` |
| `p4_lora_r8` | `TTT_LORA_RANK=8` |
| `p4_lora_r16` | `TTT_LORA_RANK=16` |
| `p4_lora_r32` | `TTT_LORA_RANK=32`, batch=32 |

### FFT (Full Fine-Tuning)
| Run | Config |
|-----|--------|
| `p4_fft_last2` | `TTT_MODE=fft2` |
| `p4_fft_last4` | `TTT_MODE=fft4` |
| `p4_fft_all` | `TTT_MODE=fft_all` |

### LoRA Variations (including MLP targets)
| Run | Config | Rationale |
|-----|--------|-----------|
| `p4_lora_r16_3step` | r16, `TTT_STEPS=3` | Multiple gradient steps per chunk |
| `p4_lora_r16_chunk128` | r16, `TTT_CHUNK_SIZE=128` | Finer-grained adaptation |
| `p4_lora_r16_qvk` | r16, `TTT_LORA_TARGETS=qvk` | Adapt K projections too |
| `p4_lora_r16_qv_mlp` | r16, `TTT_LORA_TARGETS=qv_mlp` | **LoRA on attention + MLP** |
| `p4_lora_r16_qvk_mlp` | r16, `TTT_LORA_TARGETS=qvk_mlp` | **Full coverage: Q/V/K + MLP** |
| `p4_bias_ttt` | `TTT_MODE=bias` | Minimal adapter baseline |

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
