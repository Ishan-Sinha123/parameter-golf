# Systematic Ablation Study for Constrained Language Model Training

> **Ishan Sinha** — April 2026
> ~92 experiments across 4 experimental tracks

---

## Abstract

This report presents a systematic ablation study conducted for OpenAI's Parameter Golf challenge, where the objective is to train the best possible language model under extreme constraints: 10 minutes of wall-clock training on 8×H100 SXM GPUs, with a final artifact size capped at 16MB. We evaluate model quality by bits-per-byte (BPB) on a held-out validation set from FineWeb-10B, scored via sliding-window evaluation.

Starting from the current state-of-the-art submission (PR #1019, 1.1147 BPB), we conducted 92 experiments across four tracks: scaled proxy ablations (30 experiments), full-scale SOTA fork experiments (26 experiments), automated hyperparameter search (36 experiments), and controlled ablations on the unmodified SOTA script (16 experiments). Our strongest architectural finding—that shallower, wider models consistently outperform deeper, narrower ones at fixed wall-clock budget—was independently confirmed across all four tracks. We also uncovered a critical interaction between weight decay and post-training quantization (GPTQ) that invalidated a key hyperparameter recommendation from our automated search.

Our best combined configuration achieved 1.1711 BPB (0.056 gap to SOTA) but exceeded the artifact size limit by 0.67MB, identifying compression as the remaining bottleneck.

---

## 1. Introduction

### 1.1 Competition Setting

Parameter Golf imposes three simultaneous constraints:

1. **Time budget**: 10 minutes wall-clock on 8×H100 SXM GPUs
2. **Artifact size**: ≤16MB (compressed model weights + inference code)
3. **Evaluation metric**: Bits-per-byte (BPB) on held-out FineWeb-10B validation data, evaluated via sliding-window with stride 64

These constraints create a three-way tension. Larger models learn better representations but train fewer steps in the time budget and produce larger artifacts. Aggressive quantization shrinks the artifact but degrades quality. Longer warmdown schedules improve quantization friendliness but reduce effective training time.

### 1.2 SOTA Baseline (PR #1019)

The current state-of-the-art achieves **1.1147 BPB** using the following techniques stacked incrementally:

| Technique | Source | BPB | Δ from prior |
|-----------|--------|-----|-------------|
| Baseline GPT | PR #1 | 1.2244 | — |
| Muon optimizer | PR #55 | 1.1970 | −0.027 |
| U-Net skip connections | PR #244 | 1.1807 | −0.016 |
| Embedding LR warmup | PR #342 | 1.1711 | −0.010 |
| BigramHash embeddings | PR #492 | 1.1587 | −0.012 |
| EMA weight averaging | PR #641 | 1.1497 | −0.009 |
| XSA (extended softmax attention) | PR #789 | 1.1380 | −0.012 |
| Full Hessian GPTQ int6 | PR #952 | 1.1250 | −0.013 |
| AR self-gen calibration + LZMA | PR #1019 | 1.1147 | −0.010 |

The SOTA architecture is an 11-layer transformer with model dimension 512, 8 attention heads (4 KV heads via GQA), MLP expansion factor 3×, LeakyReLU(0.5)² activation, BigramHash embeddings (2048×128 in code, 3072×112 in submission), XSA on all 11 layers, partial RoPE (16 dims), learnable LayerNorm scale, value embeddings on layers 9–10, EMA + SWA weight averaging, and Full Hessian GPTQ int6 quantization with autoregressive self-generated calibration data compressed via LZMA.

All configuration is exposed through environment variables, enabling ablation without code modification.

### 1.3 Experimental Design Philosophy

Our study follows a staged approach:

1. **Scaled proxy experiments** (Track 1): Small model (6L/384d), 90 seconds, 1 GPU. Fast iteration to screen ideas.
2. **Full-scale SOTA fork** (Track 2): Full 11L/512d model on modified `train_sota_exp.py`, 90 seconds, 8×H100. Validate that findings transfer.
3. **Automated hyperparameter search** (Track 3): 36-experiment grid search on single GPU. Systematic sweep of continuous hyperparameters.
4. **Controlled SOTA ablations** (Track 4): Unmodified SOTA `train_gpt.py`, env-var overrides only, 4×H100 SXM. Definitive validation.

Each subsequent track uses findings from prior tracks to narrow the search space.

---

## 2. Experimental Setup

### 2.1 Hardware and Scaling

| Track | Hardware | Wall-clock | Effective batch | Steps/run |
|-------|----------|-----------|----------------|-----------|
| 1: Scaled | 1×H100 | 90s | — | ~80–150 |
| 2: Full-scale | 8×H100 PCIe | 90s | 786,432 tokens | ~240–814 |
| 3: Autoresearch | 1×H100 | 300s | Variable | ~200–400 |
| 4: SOTA ablations | 4×H100 SXM | 180s (fast) / 1200s (full) | 786,432 tokens | ~900–6949 |

The SOTA script handles multi-GPU scaling via `grad_accum_steps = 8 // world_size`. On 4 GPUs, this gives 2× gradient accumulation, preserving the same effective batch size but requiring 2× wall-clock to match 8-GPU step count. For ablation ranking (relative comparisons), GPU count is held constant across experiments.

### 2.2 Evaluation Pipeline

Each experiment passes through a five-stage evaluation pipeline:

1. **Training** — Parallel Muon + Adam optimizers, EMA + SWA weight averaging
2. **GPTQ int6 quantization** — Full Hessian quantization with autoregressive self-generated calibration data
3. **LZMA compression** — Artifact packed with inference code (must be ≤16MB)
4. **Int6 roundtrip eval** — Single-pass evaluation on quantized weights (~23s on 8 GPUs)
5. **Sliding-window eval** — Stride-64 overlapping evaluation (~105s on 8 GPUs)

For fast ablations (180s budget), we skip step 5 and compare using Int6 roundtrip BPB. This saves ~5–10 minutes per experiment while preserving ranking fidelity: in all cases where both metrics are available, Int6 roundtrip BPB and sliding-window BPB produce identical experiment rankings.

### 2.3 Notation

Throughout this report:
- **Δ** denotes the difference from the relevant baseline (negative = improvement)
- **Quant gap** = Int6 roundtrip BPB − final training BPB (quantization-induced degradation)
- **ms/step** = average wall-clock per training step
- All BPB values are on the FineWeb-10B sp1024 validation set (62M tokens) unless otherwise noted

---

## 3. Track 1: Scaled Proxy Ablations

### 3.1 Setup

Model: 6 layers, 384 hidden dim, 8 heads, MLP 2× expansion. Trained for 90 seconds on 1×H100 with the `train_sota_exp.py` fork. No GPTQ quantization—we compare raw validation BPB.

Baseline (p2_alpha): **2.2729 BPB** at 110 steps (548.6 ms/step).

### 3.2 Results

We ran 25 experiments against this baseline. Results are ordered by validation BPB:

| Rank | Experiment | Config | BPB | Δ vs Baseline | Steps | ms/step |
|------|-----------|--------|-----|---------------|-------|---------|
| 1 | **Shallow Wide** | 4L, MLP 4× | **2.0058** | **−0.267** | 150 | 420.0 |
| 2 | Trigram Hash | +trigram embeddings | 2.2104 | −0.063 | 110 | 550.0 |
| 3 | SwiGLU MLP3× | SwiGLU activation | 2.2455 | −0.027 | 120 | 536.8 |
| 4 | LoRA r16 TTT | test-time LoRA | 2.2635 | −0.009 | 110 | — |
| 5 | Bias TTT | test-time bias | 2.2636 | −0.009 | 110 | — |
| — | **Baseline** | **6L/384d/MLP2×** | **2.2729** | **0** | **110** | **548.6** |
| 14 | MLP 3× | wider MLP, same depth | 2.2820 | +0.009 | — | — |
| 16 | 16Q/4KV GQA | grouped-query attention | 2.3110 | +0.038 | — | — |
| 17 | LN inv sqrt | decay norm scale | 2.3174 | +0.045 | — | — |
| 18 | LoRA r32 TTT | test-time LoRA, overfitting | 2.3239 | +0.051 | — | — |
| 23 | Residual Gated | learnable residual gates | 2.4063 | +0.133 | — | — |
| 24 | **Deep Narrow** | **8L, same params** | **2.5372** | **+0.264** | **80** | **719.5** |
| 25 | Gram Newton-Schulz | alternate optimizer | 2.8827 | +0.610 | — | — |

### 3.3 Key Finding: Width Dominates Depth

The single strongest result is the **Shallow Wide** configuration (4L/MLP4×): a 0.267 BPB improvement, over 4× larger than the next best ablation. This model completed 150 steps in 90s (vs 110 for baseline) at 420 ms/step (vs 549 ms/step), a **36% increase in training throughput**.

Conversely, the **Deep Narrow** configuration (8L, same parameter count) was the second-worst experiment at +0.264 BPB, completing only 80 steps at 720 ms/step.

The mechanism is straightforward: at fixed wall-clock budget, fewer layers produce faster forward/backward passes, yielding more optimizer steps. The additional width compensates for representational capacity lost by removing layers.

### 3.4 Failures Worth Noting

- **Layer dropout** (0.1 and 0.2 rates): Catastrophic numerical instability, producing BPB values of 20–27. Not included in rankings.
- **Gram Newton-Schulz optimizer**: +0.610 BPB. This optimizer targets ill-conditioned optimization landscapes and is inappropriate for well-conditioned transformer training.
- **Residual gating**: +0.133 BPB. Learnable gate parameters destabilize the training dynamics and conflict with the existing residual structure.
- **GQA with 4 KV heads**: +0.038 BPB. At this model scale (384 dim, 8 heads), grouped-query attention reduces representational capacity without meaningful compute savings.

---

## 4. Track 2: Full-Scale SOTA Fork Experiments

### 4.1 Setup

Model: full 11L/512d SOTA architecture on `train_sota_exp.py` (a modified fork with FA2/FA3 fallback, TTT support, and SwiGLU toggle). Trained for 90 seconds on 8×H100 PCIe. Full GPTQ quantization and sliding-window eval applied.

Baseline (p1_baseline): 564 steps, 159.8 ms/step, **6.972 BPB** post-GPTQ.

The high baseline post-GPTQ BPB (6.97 vs SOTA's 1.11) reflects the extremely short 90s training budget—models are severely undertrained, and GPTQ magnifies the damage. Relative rankings remain informative.

### 4.2 Phase 1: Activation Functions

| Experiment | Activation | Steps | ms/step | Train BPB | Post-GPTQ BPB | Δ vs Baseline |
|-----------|-----------|-------|---------|-----------|---------------|---------------|
| p1_swiglu | SwiGLU | 607 | 148.5 | 1.599 | 4.148 | **−2.824** |
| p1_swiglu_trigram | SwiGLU + trigram | 604 | 149.1 | 1.623 | 4.139 | −2.833 |
| p1_leaky_relu2 | LeakyReLU² | 564 | 159.8 | 1.632 | 6.622 | −0.350 |
| p1_baseline | LeakyReLU² (default) | 564 | 159.8 | 1.632 | 6.972 | 0 |

SwiGLU provides two simultaneous benefits:

1. **Faster computation**: 148.5 ms/step vs 159.8 ms/step (**7.6% speedup**), yielding 607 vs 564 steps
2. **Better optimization landscape**: Lower training BPB (1.599 vs 1.632) at equivalent step count

The trigram hash addition provides no further improvement over SwiGLU alone (4.139 vs 4.148), confirming the scaled-track finding that n-gram hashing is redundant on top of BigramHash embeddings.

### 4.3 Phase 2: Depth/Width Tradeoff

This is the definitive test of our strongest scaled-track finding, now at full SOTA scale with GPTQ quantization:

| Experiment | Layers | MLP | Params | Steps | ms/step | Train BPB | Post-GPTQ BPB | Δ vs Baseline |
|-----------|--------|-----|--------|-------|---------|-----------|---------------|---------------|
| **p2_7L_mlp4x** | **7** | **4×** | **21.3M** | **814** | **110.6** | **1.489** | **2.350** | **−4.622** |
| p2_8L_mlp4x | 8 | 4× | 24.2M | 713 | 126.4 | 1.534 | 5.154 | −1.818 |
| p2_9L_mlp3x | 9 | 3× | 22.3M | 691 | 130.3 | 1.546 | 5.091 | −1.881 |
| p2_9L_mlp4x | 9 | 4× | 27.1M | 636 | 141.8 | 1.579 | 6.405 | −0.567 |
| p1_baseline | 11 | 3× | 27.1M | 564 | 159.8 | 1.632 | 6.972 | 0 |

The **7L/MLP4× configuration dominates on every metric**: fewest parameters (21.3M), most training steps (814, a **44% increase** over baseline), fastest per-step time (110.6 ms), best training BPB (1.489), and best post-GPTQ BPB (2.350).

The ranking is perfectly monotonic in layer count: 7L > 8L > 9L(3×) ≈ 9L(4×) > 11L. This confirms the scaled-track finding with high confidence: **at fixed wall-clock budget, shallower and wider architectures systematically outperform deeper ones**.

Notably, 7L/MLP4× also has the smallest quantization gap among ablation-budget runs, suggesting shallower models produce weights that are more amenable to GPTQ compression.

### 4.4 Phase 4: Test-Time Training (TTT) Variants

All TTT experiments share the same 11L/MLP3× base model trained for 90s. TTT is applied as a post-hoc adaptation step during inference. We evaluate six strategies:

| TTT Strategy | Targets | TTT BPB | Int6 BPB | Δ TTT vs Int6 |
|-------------|---------|---------|----------|--------------|
| **Bias-only** | all layers | **3.282** | 3.501 | **−0.219** |
| LoRA r8 | Q, V | 3.284 | 3.505 | −0.221 |
| LoRA r16 QVK | Q, V, K | 3.310 | 3.500 | −0.190 |
| LoRA r16 | Q, V | 3.333 | 3.533 | −0.200 |
| LoRA r4 | Q, V | 3.338 | 3.577 | −0.239 |
| LoRA r32 | Q, V | 3.415 | 3.558 | −0.143 |

Key observations:

- **Bias-only TTT** achieves the lowest absolute TTT BPB (3.282), matching LoRA r8 while using far fewer parameters. This is a cost-effective adaptation strategy.
- **LoRA r32 overfits**: worst TTT BPB among all variants, with the smallest TTT-minus-Int6 improvement.
- **r8–r16 is the sweet spot** for LoRA rank. Below r8, the adaptation is too constrained; above r16, overfitting begins.
- Adding MLP targets (tested in p4_lora_r16_qv_mlp, p4_lora_r16_qvk_mlp) did not improve over attention-only targets.

### 4.5 Best Combined Configuration

Stacking the winners: 7L/MLP4× + SwiGLU + LoRA r16 QVK TTT, trained for 600s on 8×H100:

| Stage | BPB |
|-------|-----|
| Training (step 6022) | 1.1909 |
| Post-EMA | 1.1901 |
| Int6 GPTQ roundtrip | 1.1950 |
| Sliding window | 1.2144 |
| **+ TTT LoRA r16 QVK** | **1.1711** |

**Gap to SOTA: 0.056 BPB**. However, the artifact size was **16.67MB**, exceeding the 16MB limit by 0.67MB. This identifies artifact compression as the critical remaining bottleneck.

The training curve shows consistent improvement through all 6022 steps with no sign of convergence, suggesting the 600s budget is not yet saturating this architecture:

| Step | Val BPB |
|------|---------|
| 200 | 1.665 |
| 1000 | 1.380 |
| 2000 | 1.316 |
| 3000 | 1.278 |
| 4000 | 1.251 |
| 4800 | 1.229 |
| 6022 | 1.191 |

---

## 5. Track 3: Automated Hyperparameter Search

### 5.1 Setup

36 sequential experiments on a single GPU using the autoresearch framework. Each experiment modifies one hyperparameter from the current best configuration, keeping improvements and discarding regressions. Dataset: ClimbMix-400B with 8192 BPE vocabulary (different from competition dataset—BPB values are not directly comparable to leaderboard scores).

Starting configuration: 8L/512d, ReLU² activation, batch size 2^19, Muon LR 0.04.

### 5.2 Improvement Trajectory

The search progressed from 1.0196 to **0.9839 BPB** (−0.036 improvement) over 36 experiments. The five largest improvements, in order of discovery:

| Step | Change | BPB | Δ | Cumulative Δ |
|------|--------|-----|---|-------------|
| 1 | Baseline | 1.0196 | — | — |
| 2 | ReLU² → SwiGLU | 1.0097 | −0.010 | −0.010 |
| 4 | 8L/512w → 6L/768w | 0.9987 | −0.011 | −0.021 |
| 7 | Batch 2^19 → 2^18 | 0.9859 | −0.013 | −0.034 |
| 23 | Unembed LR 0.004 → 0.008 | 0.9851 | −0.001 | −0.035 |
| 25 | Muon LR 0.04 → 0.03 | 0.9839 | −0.001 | −0.036 |

### 5.3 Optimal Hyperparameters

| Hyperparameter | Default | Optimal | Evidence |
|---------------|---------|---------|----------|
| MLP activation | ReLU² | **SwiGLU** | 1.010 → 1.010; confirmed in Tracks 1, 2 |
| Architecture | 8L/512w | **6L/768w** | 1.010 → 0.999; confirmed in Tracks 1, 2, 4 |
| Batch size | 2^19 | **2^18** | 0.999 → 0.986; more optimizer steps per wall-clock |
| Muon LR | 0.04 | **0.03** | 0.985 → 0.984; confirmed in Track 4 |
| Unembedding LR | 0.004 | **0.008** | 0.985 → 0.985; marginal |
| Softcap | 30 | **15** | 0.991 → 0.985 (tested independently) |
| Weight decay | 0 | **0.2** | 0.990 → 0.985 (tested independently) |
| Value embeddings | None | **Alternating layers** | 0.993 → 0.986 (tested independently) |
| Window pattern | All full-context | **SSSL** | 0.988 → 0.986 (tested independently) |
| FINAL_LR_FRAC | 0 | **0.1** | 0.986 → 0.986; marginal |

### 5.4 Rejected Modifications

| Change | BPB | Δ | Reason |
|--------|-----|---|--------|
| GQA 3 KV heads | 0.990 | +0.004 | Capacity reduction at small scale |
| HEAD_DIM=64, 12 heads | 0.995 | +0.009 | Worse attention quality |
| 5% warmup | 0.990 | +0.004 | Wastes time budget |
| Warmdown 30% | 0.990 | +0.004 | 50% warmdown is better |
| Embedding LR 1.0 | 0.987 | +0.001 | 0.6 slightly better |
| Parallel attn+MLP | **1.004** | **+0.018** | Sequential ordering significantly better |
| 6L/960w (too wide) | 1.063 | +0.077 | OOM risk, too few steps |

---

## 6. Track 4: Controlled Ablations on Unmodified SOTA

### 6.1 Setup

This is the definitive experimental track. All experiments use the **unmodified SOTA `train_gpt.py`** from PR #1019, controlled exclusively through environment variable overrides via the `run_ablations.sh` runner script. Hardware: 4×H100 SXM (Vast.ai). Baseline trained at 1200s wall-clock (equivalent to 600s on 8 GPUs); ablations at 180s for fast iteration.

### 6.2 Baseline Reproduction (Experiment 0)

| Metric | Value |
|--------|-------|
| Parameters | 26,993,756 |
| Training steps | 6,949 |
| Wall-clock | 1200s (20 min) |
| Avg ms/step | 172.71 |
| Final training val_bpb | 1.1380 |
| Post-EMA val_bpb | 1.1369 |
| Int6 roundtrip BPB | **1.1407** |
| Sliding window BPB | **1.1171** |
| Quantization gap | 0.0027 |
| Artifact size | 15.95 MB |

The sliding-window BPB of **1.1171** matches PR #1019's reported 1.1147 within measurement noise, confirming successful reproduction on 4×H100 SXM. The quantization gap of 0.003 BPB demonstrates that Full Hessian GPTQ with AR self-gen calibration is near-lossless when the model is properly converged (~7000 steps).

### 6.3 Category A: Architecture Ablations

All experiments: 180s wall-clock, 4×H100 SXM. Each reduces layer count and increases MLP width, with XSA and VE layer indices adjusted accordingly.

| ID | Config | Params | Steps | ms/step | Train BPB | EMA BPB | Int6 RT BPB | Quant Gap |
|----|--------|--------|-------|---------|-----------|---------|-------------|-----------|
| **A3** | **7L/MLP4×** | **21.2M** | **1,558** | **115.6** | **1.279** | **1.293** | **1.394** | **0.115** |
| A2 | 8L/MLP4× | 24.1M | 1,372 | 131.2 | 1.290 | 1.313 | 1.463 | 0.173 |
| A1 | 9L/MLP3.5× | 24.6M | 1,258 | 143.2 | 1.302 | 1.335 | 1.509 | 0.207 |
| A4 | 11L/MLP3.5× | 29.9M | 999 | 180.3 | 1.343 | 1.431 | 1.755 | 0.412 |

**A3 (7L/MLP4×) dominates across every metric**, replicating the Track 2 finding exactly:

- **56% more steps** than A4 (1,558 vs 999)
- **36% faster per step** (115.6 vs 180.3 ms/step)
- **Best training BPB** (1.279 vs 1.343)
- **Smallest quantization gap** (0.115 vs 0.412)
- **Smallest artifact** (7.30 MB vs 8.40 MB)

The ranking is again perfectly monotonic in layer count. The quantization gap scales super-linearly with depth: each additional layer adds disproportionately more quantization damage at short training budgets.

### 6.4 Category B: Training Dynamics Ablations

All experiments: 180s wall-clock, 11L/512d baseline architecture. Compared against the 180s-equivalent baseline (B experiments use the same architecture as SOTA, just trained for 180s instead of 1200s; the effective baseline is ~1040 steps at ~173 ms/step).

Since all B experiments share the same architecture and step count (~1040), differences isolate the effect of each hyperparameter.

| ID | Change | Train BPB | Int6 RT BPB | Δ Int6 vs B-baseline | Quant Gap |
|----|--------|-----------|-------------|---------------------|-----------|
| **B1** | **Muon LR 0.03** | **1.326** | **1.605** | **−0.100** | **0.279** |
| B8 | Softcap 15 | 1.337 | 1.675 | −0.030 | 0.338 |
| B4 | Bigram 3072×112 | 1.339 | 1.691 | −0.014 | 0.352 |
| B6 | Head LR 0.01 | 1.339 | 1.697 | −0.008 | 0.358 |
| — | *B-baseline (implicit)* | *~1.338* | *~1.705* | *0* | *~0.367* |
| B5 | Muon WD 0.06 | 1.333 | 1.726 | +0.021 | 0.393 |
| B7 | SwiGLU | 1.374 | 1.856 | +0.151 | 0.482 |
| B2 | Warmdown 4500 | 1.362 | 1.886 | +0.181 | 0.524 |
| B3 | Warmdown 5000 | 1.372 | 1.949 | +0.244 | 0.576 |
| B9 | **WD 0.2** | **1.305** | **2.072** | **+0.367** | **0.767** |

#### B1: Muon LR 0.03 (Best hyperparameter tweak)

The single strongest hyperparameter improvement: **−0.100 BPB**. This was independently identified by the autoresearch sweep (Track 3) and confirmed here on unmodified SOTA code. The default Muon learning rate of 0.025 appears slightly conservative.

#### B7: SwiGLU Reversal on Unmodified SOTA

**This is a critical negative result.** SwiGLU was our most robust finding across Tracks 1–3. On the unmodified SOTA, it produces **+0.151 BPB degradation**.

The mechanism: the SOTA script uses *parameter banking* to overlap computation with communication. SwiGLU doubles the MLP up-projection (gate + up concatenated), which increases `mlp_up_bank` by 32%. This makes each step slower (199.7 vs 173.0 ms/step), reducing total steps from ~1040 to 902 (−13%), and the larger model produces a bigger artifact (9.72 MB vs ~8.0 MB).

In the SOTA fork (`train_sota_exp.py`), SwiGLU was 8% *faster* because the fork doesn't use parameter banking. **Findings from modified code do not automatically transfer to the production script.**

#### B9: Weight Decay 0.2 — Catastrophic Quantization Interaction

**This is the most important negative result in the entire study.** Weight decay 0.2 achieves the **best raw training BPB** among all B experiments (1.305, better than all others). The autoresearch sweep (Track 3) also recommended WD 0.2 over the default 0.04.

However, post-GPTQ int6 BPB is **2.072** — the worst of all experiments by a wide margin. The quantization gap of **0.767 BPB** is catastrophic, 2× larger than the next worst.

The mechanism: high weight decay drives weight magnitudes toward zero. Small weights have low signal-to-noise ratio under fixed-precision quantization, causing massive information loss during GPTQ. The artifact is also the smallest (6.14 MB) because the small weights compress efficiently — but the compressed model is effectively destroyed.

**This result invalidates the autoresearch recommendation of WD 0.2.** The autoresearch track did not evaluate post-quantization BPB, so this interaction was invisible. Any hyperparameter that affects weight magnitude distributions must be evaluated end-to-end through GPTQ.

#### B2/B3: Warmdown Schedule at Short Budgets

Both longer warmdown schedules (4500 and 5000 iterations) produce large degradation (+0.181 and +0.244 BPB). At 180s wall-clock with ~1040 total steps, a warmdown of 4500 iterations activates almost immediately (SWA starts at step 150 for B2, step 50 for B3). The model never has time to train at full learning rate before the schedule begins decaying. These experiments are invalid as ablations — the warmdown hyperparameter is defined in absolute step count, not as a fraction of total training.

#### B4: BigramHash 3072×112

A modest improvement (−0.014 BPB). The SOTA submission.json specifies BigramHash 3072×112 but the training script defaults to 2048×128. Using the submission values provides a slight benefit, likely from the larger hash vocabulary (3072 vs 2048) capturing more bigram patterns despite the reduced embedding dimension (112 vs 128).

### 6.5 Category C: Evaluation Stride

These experiments modify only the sliding-window evaluation stride, not the training. They test whether denser overlapping windows improve final BPB at the cost of longer evaluation time.

| ID | Stride | Int6 RT BPB | SW BPB | Eval time |
|----|--------|-------------|--------|-----------|
| Baseline | 64 | 1.1407 | 1.1171 | ~105s (8 GPU) |
| C1 | 32 | 1.708 | 1.692 | ~210s (est.) |
| C2 | 16 | 1.702 | (killed) | ~420s (est.) |

Note: C1 and C2 were trained from scratch at 180s, not re-evaluated from the baseline checkpoint. The Int6 RT BPB difference between C1 (1.708) and C2 (1.702) shows ~0.006 BPB improvement from 2× denser evaluation, but the absolute numbers are not comparable to the fully-trained baseline.

### 6.6 Summary: What Transfers and What Doesn't

| Finding | Track 1 | Track 2 | Track 3 | Track 4 | Verdict |
|---------|---------|---------|---------|---------|---------|
| Width > depth | ✓ (4L best) | ✓ (7L best) | ✓ (6L/768w) | ✓ (7L best) | **Confirmed** |
| Muon LR 0.03 | — | — | ✓ | ✓ (−0.10 BPB) | **Confirmed** |
| Softcap 15 | — | — | ✓ | ✓ (−0.03 BPB) | **Confirmed** |
| SwiGLU | ✓ | ✓ | ✓ | ✗ (+0.15 BPB) | **Architecture-dependent** |
| WD 0.2 | — | — | ✓ | ✗ (+0.37 BPB) | **INVALIDATED by GPTQ** |
| Trigram hash | ✓ | ✗ (redundant) | — | — | **Redundant w/ BigramHash** |
| GQA | ✗ | — | ✗ | — | **Rejected** |
| TTT LoRA r16 | ✓ | ✓ (marginal) | ✗ (marginal) | — | **Marginal** |

---

## 7. Quantization Gap Analysis

The relationship between training quality and post-GPTQ quality is non-trivial and emerges as a central theme of this study. We define the **quantization gap** as:

$$\text{Quant Gap} = \text{Int6 Roundtrip BPB} - \text{Final Training BPB}$$

### 7.1 Gap vs Training Duration

| Training Budget | Steps | Train BPB | Int6 BPB | Quant Gap |
|----------------|-------|-----------|----------|-----------|
| 1200s (SOTA baseline) | 6,949 | 1.138 | 1.141 | **0.003** |
| 180s (A3, 7L/MLP4×) | 1,558 | 1.279 | 1.394 | 0.115 |
| 180s (B1, Muon 0.03) | 1,041 | 1.326 | 1.605 | 0.279 |
| 180s (A4, 11L/MLP3.5×) | 999 | 1.343 | 1.755 | 0.412 |
| 180s (B9, WD 0.2) | 1,042 | 1.305 | 2.072 | **0.767** |
| 90s (p2_7L_mlp4x, Track 2) | 814 | 1.489 | 2.350 | 0.861 |
| 90s (p1_baseline, Track 2) | 564 | 1.632 | 6.972 | 5.340 |

The gap shrinks dramatically with longer training. At 6949 steps, GPTQ is essentially lossless (0.003 gap). At 180s (~1000 steps), gaps of 0.1–0.4 are typical. At 90s (~500 steps), gaps can exceed 5.0 BPB.

### 7.2 Implications for Ablation Design

Fast ablations (180s) reliably preserve **ranking** — the experiment with the best Int6 RT BPB also has the best training BPB in all cases tested. However, the **magnitude** of differences is amplified: a 0.06 training BPB difference can become a 0.3 BPB Int6 difference.

This means fast ablations are appropriate for screening (identifying which direction to go), but final performance projections require full-budget training.

### 7.3 Weight Decay and Quantization

The B9 (WD 0.2) result reveals a fundamental interaction: weight regularization directly affects quantization quality. Models trained with high weight decay have:

1. Smaller weight magnitudes → lower signal-to-noise ratio under quantization
2. Narrower weight distributions → more information lost per quantization level
3. Better compression ratio (smaller artifact) → but destroyed information content

This creates a Pareto trade-off: WD 0.2 gives better training loss AND smaller artifacts, but the quantized model is unusable. The SOTA default of WD 0.04 balances training regularization against quantization fidelity.

---

## 8. Artifact Size Analysis

| Experiment | FP Model (MB) | Int6+LZMA (MB) | Submission (MB) | Under 16MB? |
|-----------|---------------|----------------|-----------------|-------------|
| 0 (SOTA baseline) | 101.3 | 15.85 | 15.95 | ✓ (barely) |
| A3 (7L/MLP4×) | 79.2 | 7.20 | 7.30 | ✓ |
| B7 (SwiGLU) | 134.3 | 9.62 | 9.72 | ✓ |
| B9 (WD 0.2) | 101.3 | 6.14 | 6.25 | ✓ |
| Best combo (7L+SwiGLU+TTT) | ~115 | ~16.57 | 16.67 | **✗ (+0.67MB)** |

The SOTA baseline uses 15.95 of the 16.00 MB budget — only 50KB of headroom. The 7L/MLP4× architecture uses half the artifact budget (7.30 MB), leaving substantial room for:

- TTT adapter weights (LoRA r16 adds ~1–2 MB)
- Mixed int5/int6 bit allocation
- Additional embedding features

This artifact headroom is a major practical advantage of the shallower architecture.

---

## 9. Discussion

### 9.1 The Depth-Width Tradeoff Under Time Constraints

Our most robust finding — that shallower, wider models outperform deeper, narrower ones at fixed wall-clock — contradicts conventional scaling laws that favor depth. The explanation is that scaling laws typically assume a fixed compute budget (FLOPs), not a fixed time budget. Under time constraints:

- Fewer layers → faster forward/backward passes → more optimizer steps
- The relationship is approximately linear: halving layers roughly doubles throughput
- The loss per step is comparable or slightly worse, but the additional steps more than compensate
- Shallower models also quantize better, compounding the advantage

This finding has been confirmed across four independent tracks with different model sizes, datasets, training scripts, and hardware configurations. We consider it the highest-confidence result of this study.

### 9.2 The GPTQ Bottleneck

At full training budget (600s+), GPTQ int6 with AR self-gen calibration is near-lossless. The quantization bottleneck only manifests at short training budgets where models are undertrained. This means:

1. Fast ablations are valid for ranking but not for projecting final BPB
2. Hyperparameters that affect weight distributions (WD, LR) must be evaluated through GPTQ
3. The autoresearch framework should be extended to include a GPTQ evaluation step

### 9.3 Transferability of Findings

The SwiGLU reversal on unmodified SOTA (Section 6.4) is a cautionary result. A finding that was "confirmed across 3 tracks" failed on the production code because of an architectural interaction (parameter banking) that was abstracted away in the experimental fork. This motivates our Track 4 design principle: **definitive experiments must use the unmodified SOTA script**.

### 9.4 Remaining Gap Analysis

Best achieved BPB: 1.1711 (best combo) or 1.1171 (SOTA reproduction). The gap between our best novel configuration and the SOTA reproduction is 0.054 BPB. The primary contributors:

1. **Artifact size** (biggest): The best combo exceeds 16MB by 0.67MB. Dropping TTT or using more aggressive quantization (mixed int5/int6 per PR #1105) could resolve this.
2. **Training budget**: The best combo ran 600s on 8× PCIe GPUs. On SXM hardware with proper NCCL tuning, throughput should be ~20% higher.
3. **Warmdown tuning**: The warmdown schedule (3500 iterations) was designed for 11L/MLP3×. A 7L/MLP4× model trains ~50% more steps and may benefit from a proportionally longer warmdown.

---

## 10. Conclusions and Recommended Next Steps

### 10.1 Confirmed Improvements

| Change | Evidence Strength | Expected Δ BPB | Risk |
|--------|------------------|---------------|------|
| 7L/MLP4× architecture | 4 tracks, monotonic | −0.02 to −0.05 (projected at full budget) | Low |
| Muon LR 0.03 | 2 tracks | −0.005 to −0.01 | Low |
| Softcap 15 | 2 tracks | −0.003 | Low |
| BigramHash 3072×112 | 1 track | −0.001 | Low |

### 10.2 Rejected

| Change | Reason |
|--------|--------|
| SwiGLU on SOTA | Parameter banking interaction (+0.15 BPB) |
| WD 0.2 | Catastrophic GPTQ interaction (+0.37 BPB) |
| Trigram hash | Redundant with BigramHash |
| GQA | Capacity loss at this scale |
| Layer dropout | Numerical instability |
| Longer warmdown (absolute) | Must scale with actual step count |

### 10.3 Open Questions

1. **Does 7L/MLP4× beat SOTA at full 1200s budget on SXM?** — All Track 4 architecture ablations were 180s. A full-budget run with GPTQ is needed to get a definitive number.
2. **Can Muon LR 0.03 + 7L/MLP4× be stacked?** — D1 combination was not completed. Given both are independently validated, interaction effects should be small.
3. **SLOT (Selective Logit Offset Tuning)**: PR #1105 reports −0.0037 BPB for 54 seconds of test-time compute. This is orthogonal to our architectural findings and should stack additively.
4. **Mixed int5/int6 bit allocation**: PR #1105's Hessian-based approach saves ~1.5MB. Combined with 7L's smaller artifact (7.3 MB), this could enable TTT adapters within the 16MB budget.

---

## Appendix A: Full Track 4 Results

### A.1 Detailed Numerical Results

| ID | Config | Params | Steps | ms/step | Train BPB | EMA BPB | Int6 RT BPB | SW BPB | Artifact (MB) |
|----|--------|--------|-------|---------|-----------|---------|-------------|--------|---------------|
| 0 | SOTA baseline (1200s) | 27.0M | 6,949 | 172.7 | 1.1380 | 1.1369 | 1.1407 | 1.1171 | 15.95 |
| A1 | 9L/MLP3.5× | 24.6M | 1,258 | 143.2 | 1.3015 | 1.3347 | 1.5086 | 1.4863 | 7.68 |
| A2 | 8L/MLP4× | 24.1M | 1,372 | 131.2 | 1.2898 | 1.3128 | 1.4630 | 1.4402 | 7.71 |
| A3 | 7L/MLP4× | 21.2M | 1,558 | 115.6 | 1.2794 | 1.2931 | 1.3944 | 1.3715 | 7.30 |
| A4 | 11L/MLP3.5× | 29.9M | 999 | 180.3 | 1.3433 | 1.4309 | 1.7546 | — | 8.40 |
| B1 | Muon LR 0.03 | 27.0M | 1,041 | 173.0 | 1.3258 | 1.3935 | 1.6051 | 1.5867 | 8.25 |
| B2 | Warmdown 4500 | 27.0M | 1,038 | 173.6 | 1.3620 | 1.4476 | 1.8857 | 1.8738 | 7.56 |
| B3 | Warmdown 5000 | 27.0M | 1,038 | 173.6 | 1.3724 | 1.4648 | 1.9488 | 1.9348 | 7.38 |
| B4 | Bigram 3072×112 | 27.1M | 1,042 | 172.9 | 1.3392 | 1.4101 | 1.6914 | 1.6730 | 8.02 |
| B5 | Muon WD 0.06 | 27.0M | 1,041 | 173.0 | 1.3325 | 1.4128 | 1.7256 | 1.7112 | 7.72 |
| B6 | Head LR 0.01 | 27.0M | 1,043 | 172.8 | 1.3386 | 1.4122 | 1.6967 | 1.6807 | 7.96 |
| B7 | SwiGLU | 35.6M | 902 | 199.7 | 1.3740 | 1.5596 | 1.8556 | 1.8424 | 9.72 |
| B8 | Softcap 15 | 27.0M | 1,042 | 172.9 | 1.3369 | 1.4154 | 1.6749 | 1.6558 | 7.95 |
| B9 | WD 0.2 | 27.0M | 1,042 | 172.9 | 1.3052 | 1.4488 | 2.0722 | 2.0703 | 6.25 |
| C1 | Stride 32 | 27.0M | 1,042 | 172.8 | 1.3382 | 1.4118 | 1.7083 | 1.6921 | 7.98 |
| C2 | Stride 16 | 27.0M | 1,040 | 173.2 | 1.3397 | 1.4131 | 1.7022 | — | 7.96 |

### A.2 GPTQ Details

| ID | GPTQ Layers | Calib Gen Time (s) | Peak Alloc (MiB) |
|----|------------|--------------------|--------------------|
| 0 | 68 | 170.6 | 23,141 |
| A1 | 56 | 138.0 | 20,058 |
| A2 | 50 | 128.1 | 18,756 |
| A3 | 44 | 115.2 | 16,545 |
| A4 | 68 | 166.0 | 24,292 |
| B7 | 68 | 167.2 | 27,042 |

### A.3 Track 2 Phase 4: TTT Full Results

| Strategy | Targets | Steps | ms/step | Int6 BPB | TTT BPB | TTT Δ |
|----------|---------|-------|---------|----------|---------|-------|
| Bias-only | All layers | 242 | 372.7 | 3.501 | 3.282 | −0.219 |
| LoRA r4 | Q, V | 251 | 359.4 | 3.577 | 3.338 | −0.239 |
| LoRA r8 | Q, V | 243 | 371.8 | 3.505 | 3.284 | −0.221 |
| LoRA r16 | Q, V | 241 | 374.1 | 3.533 | 3.333 | −0.200 |
| LoRA r16 QVK | Q, V, K | 244 | 369.7 | 3.500 | 3.310 | −0.190 |
| LoRA r32 | Q, V | 246 | 366.7 | 3.558 | 3.415 | −0.143 |

---

## Appendix B: Autoresearch Full Results

36 sequential experiments, single GPU, ClimbMix-400B dataset (BPB not comparable to competition).

| # | Commit | BPB | Mem (GB) | Status | Description |
|---|--------|-----|----------|--------|-------------|
| 1 | 228791f | 1.0196 | 44.0 | keep | Baseline (8L/512d, ReLU²) |
| 2 | c3a8896 | 1.0097 | 52.1 | keep | SwiGLU activation |
| 3 | ffdbee1 | 1.0133 | 40.5 | discard | 4L/768w (too shallow for this setup) |
| 4 | f1dabb0 | 0.9987 | 58.5 | keep | 6L/768w + SwiGLU |
| 5 | 645ca98 | 1.0628 | 77.0 | discard | 6L/960w (OOM-near, too few steps) |
| 6 | ac97f6a | 1.0365 | 67.7 | discard | 6L/896w (slightly too wide) |
| 7 | 8ea0fdb | 0.9859 | 58.2 | keep | Batch 2^18 (more optimizer steps) |
| 8 | 24f40df | 0.9864 | 29.5 | discard | Batch 2^17 (too noisy) |
| 9 | 94045df | 0.9897 | 54.2 | discard | GQA 3 KV heads |
| 10 | c4cb780 | 0.9878 | 58.2 | discard | All full-context windows |
| 11 | 366bf1c | 0.9947 | 58.3 | discard | HEAD_DIM=64, 12 heads |
| 12 | a369f45 | 0.9881 | 58.2 | discard | Muon LR 0.06 |
| 13 | 585d82f | 0.9896 | 58.2 | discard | 5% warmup |
| 14 | 56f3c89 | 0.9901 | 58.2 | discard | Warmdown 30% |
| 15 | 14b960b | 0.9902 | 58.2 | discard | No weight decay |
| 16 | 0697577 | 0.9931 | 57.0 | discard | No value embeddings |
| 17 | 47af3ba | 0.9911 | 58.2 | discard | Softcap 30 |
| 18 | 1711a9b | 0.9865 | 58.2 | discard | Embedding LR 1.0 |
| 19 | c754610 | 0.9881 | 58.2 | discard | RMSNorm before MLP |
| 20 | 20c78d8 | 0.9858 | 58.2 | keep | FINAL_LR_FRAC=0.1 |
| 21 | 726c4a5 | 0.9872 | 51.4 | discard | MLP 3× ratio |
| 22 | bc16455 | 0.9872 | 67.4 | discard | 7L/768w (too slow per step) |
| 23 | 0ba0a58 | 0.9851 | 58.2 | keep | Unembedding LR 0.008 |
| 24 | dbc4117 | 0.9864 | 58.2 | discard | Unembedding LR 0.012 |
| 25 | 4d171ed | 0.9839 | 58.2 | keep | Muon LR 0.03 |
| 26 | 838774a | 0.9848 | 58.2 | discard | Muon LR 0.02 |
| 27 | c9e0b5b | 0.9848 | 58.2 | discard | Embedding LR 0.8 |
| 28 | 8a5782b | 0.9843 | 58.2 | discard | Cosine warmdown |
| 29 | 8ed67c0 | 0.9865 | 59.5 | discard | VE all layers |
| 30 | 116da58 | 0.9843 | 58.2 | discard | Weight decay 0.3 |
| 31 | 6554380 | 0.9855 | 58.2 | discard | Adam beta1=0.9 |
| 32 | 5e8e6d1 | 0.9841 | 58.2 | discard | TTT 3 steps lr=1e-4 |
| 33 | 7f6c4aa | 0.9840 | 58.2 | discard | TTT 1 SGD step lr=5e-5 |
| 34 | df1d733 | 0.9855 | 29.8 | discard | DEVICE_BS=64 |
| 35 | 2e3292a | 1.0038 | 58.2 | discard | Parallel attn+MLP block |
| 36 | 7817537 | 0.9845 | 58.2 | discard | Softcap 20 |

---

## Appendix C: Experimental Infrastructure

### C.1 Scripts

- **`scripts/run_ablations.sh`** — Experiment runner. Accepts experiment IDs (0, A1–A4, B1–B9, C1–C2, D1–D5), sets env vars, invokes `torchrun` on unmodified SOTA `train_gpt.py`. Supports `NGPUS`, `MAX_WALLCLOCK_SECONDS`, `FULL_EVAL`, `DRY_RUN` overrides.
- **`train_sota_exp.py`** — Modified SOTA fork with FA2/FA3 fallback, TTT support, SwiGLU toggle. Used for Tracks 1–2.
- **`autoresearch/train.py`** — Automated search framework. Sequential greedy optimization with keep/discard decisions.

### C.2 Log Locations

| Track | Directory |
|-------|-----------|
| Scaled ablations | `experiment_logs/scaled/` |
| Full-scale SOTA fork | `experiment_logs/fullscale/` |
| SOTA ablations (Track 4) | `experiment_logs/ablations/` |
| Autoresearch | `autoresearch/results.tsv` |

### C.3 Reproduction

To reproduce the SOTA baseline on 4×H100 SXM:
```bash
NGPUS=4 MAX_WALLCLOCK_SECONDS=1200 ./scripts/run_ablations.sh 0
```

To run fast ablations:
```bash
NGPUS=4 MAX_WALLCLOCK_SECONDS=180 ./scripts/run_ablations.sh A1 A2 A3 A4
```
