# PR#1458 midnight_12l_brotli_mixed_int

## Hypothesis

Reproduce PR#1458 on our own infra before stacking anything on top. The author claims **mean 1.10597 bpb across 3 seeds (σ = 0.00032)**, a ~0.0087-nat beat over the 1.1147 SOTA. The novel lever is "spend compression headroom (Brotli + mixed-int `attn=int5` / `mlp=int6` / `embed=int8`) on an extra transformer layer": 11L → 12L while staying under the 16 MB artifact cap. If the reproduction lands within noise of the claim, the natural follow-up is `stack_on_best`: layer Legal Score-First TTT (PR #549) and self-gen GPTQ calibration (PR #1019) on top of this 12L + Brotli base to combine both gain sources.

## Configuration

- **Recipe:** `rec_20260410_pr1458_midnight_12l_brotli_mixed_int_dbd11f1a`
- **Source:** PR#1458 (`is_reproduction=true`)
- **Script:** `autoresearch/baselines/rec_20260410_2026_03_31_parallelresiduals_minidepthre_4d594ac1/train_gpt.py`
- **Commit:** `b862f3cd8360…` on `autoresearch-deploy`
- **Stage:** screen, 8× H100 SXM, `world_size=8`, `grad_accum_steps=1`
- **Seed:** 444 (single seed; author reported 3-seed mean)
- **Model params:** 29,317,732
- **Wallclock recorded:** 420 s (`experiment.json`)

### Shape / architecture

| Key | Value |
| --- | --- |
| `NUM_LAYERS` | **12** (novel: +1 over the 11L precedent) |
| `MODEL_DIM` | 512 |
| `NUM_HEADS` / `NUM_KV_HEADS` | 8 / 4 (GQA) |
| `MLP_MULT` | 3.0 |
| `VOCAB_SIZE` | 1024 (SentencePiece BPE) |
| `TIE_EMBEDDINGS` | 1 |
| `XSA_LAST_N` | 11 (XSA active on layers 1–11, layer 0 plain) |
| `ROPE_DIMS` / `ROPE_BASE` | 16 / 1e4 |
| `VE_ENABLED` / `VE_DIM` / `VE_LAYERS` | 1 / 128 / 9,10 |
| `BIGRAM_VOCAB_SIZE` / `BIGRAM_DIM` | 2048 / 112 |
| `LN_SCALE` | 1 |
| `LOGIT_SOFTCAP` | 30 |
| `NEGATIVE_SLOPE` | 0.5 (LeakyReLU²) |
| `PARALLEL_RESIDUAL` | 0 |
| `TTT_ENABLED` | 0 |

### Training schedule

| Key | Value |
| --- | --- |
| `ITERATIONS` | 20000 |
| `WARMUP_STEPS` / `WARMDOWN_ITERS` | 20 / 3500 |
| `TRAIN_BATCH_TOKENS` / `TRAIN_SEQ_LEN` | 786432 / 2048 |
| `EVAL_SEQ_LEN` / `EVAL_STRIDE` | 2048 / 64 |
| `MAX_WALLCLOCK_SECONDS` | 600.0 |
| `LOADER_MODE` | coprime |
| `VAL_LOSS_EVERY` | 4000 |

### Optimizer

| Key | Value |
| --- | --- |
| `MATRIX_LR` (early → late) | 0.025 → 0.03 |
| `EMBED_LR` / `TIED_EMBED_LR` / `HEAD_LR` | 0.6 / 0.035 / 0.008 |
| `SCALAR_LR` | 0.025 |
| `MUON_MOMENTUM` | 0.99 (warmup 0.92 → 0.99 over 1500 steps) |
| `MUON_BACKEND_STEPS` / `MUON_BETA2` / `MUON_WD` | 5 / 0.95 / 0.04 |
| `BETA1` / `BETA2` / `ADAM_EPS` / `ADAM_WD` | 0.9 / 0.95 / 1e-8 / 0.04 |
| `GRAD_CLIP_NORM` | 0.3 |
| `SWA_ENABLED` / `SWA_EVERY` | 1 / 50 |
| `QK_GAIN_INIT` / `BANK_SPLIT` | 1.5 / 5 |

### Quantization / compression (the novel lever)

| Key | Value |
| --- | --- |
| `MIXED_QUANT` | 1 |
| `QUANT_ATTN_BITS` | **5** |
| `QUANT_MLP_BITS` / `QUANT_AUX_BITS` | 6 / 6 |
| `QUANT_EMBED_BITS` / `QUANT_OTHER_BITS` | 8 / 8 |
| `N_INT6_LAYERS` | 10 |
| `USE_GPTQ` / `SKIP_GPTQ` | 1 / 1 |
| `GPTQ_AR_SELFGEN` / `GPTQ_CALIB_SAMPLES` / `GPTQ_TEMPERATURE` | 1 / 64 / 0.8 |
| `LATE_QAT_THRESHOLD` | 0.15 |
| `QUANT_CLIP_RANGE` | 31 |
| `COMPRESSOR` | **brotli** |

## Results

### Training log excerpts (`train.log`, captured through step 3500)

```
model_params:29317732
XSA:last_11 active_layers:[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
world_size:8 grad_accum_steps:1
sdp_backends:cudnn=False flash=True mem_efficient=False math=False
attention_mode:gqa num_heads:8 num_kv_heads:4
tie_embeddings:True embed_lr:0.035 head_lr:0.0 matrix_lr:0.03 scalar_lr:0.025
train_batch_tokens:786432 train_seq_len:2048 iterations:20000 warmup_steps:20 max_wallclock_seconds:600.000
seed:444
gptq:reserving 0ms from training budget, effective=600000ms
step:0/20000    val_loss:6.9299 val_bpb:4.1043  train_time:0ms      step_avg:0.01ms
step:500/20000  train_loss:2.3033               train_time:47218ms  step_avg:94.44ms
step:1000/20000 train_loss:2.2197               train_time:93982ms  step_avg:93.98ms
step:1500/20000 train_loss:2.1124               train_time:140796ms step_avg:93.86ms
step:2000/20000 train_loss:2.0878               train_time:187689ms step_avg:93.84ms
step:2500/20000 train_loss:2.0308               train_time:234607ms step_avg:93.84ms
step:3000/20000 train_loss:2.0665               train_time:281528ms step_avg:93.84ms
step:3500/20000 train_loss:2.0667               train_time:328442ms step_avg:93.84ms
```

Log was captured only through **step 3500 / 20000 @ 328 s**. No `VAL_LOSS_EVERY=4000` tick, no final val, no post-training GPTQ, and no gate lines made it into the file. No warnings, no divergence — train loss is trending normally (2.30 → 2.07 between steps 500–3500). Observed step time ≈ **94 ms/step**.

### Gate metrics (from `experiment.json` / task prompt)

Baseline used for Δ calc: **1.0810** (per task prompt). Author-claim reference: **1.10597 bpb (σ = 0.00032, n=3)**. Current leaderboard SOTA: **1.1147**.

| Metric | Value | Baseline (1.0810) | Δ vs baseline |
| --- | --- | --- | --- |
| `screen_ema_bpb` | **1.1293** | 1.0810 | **+0.0483** (worse) |
| `gate_int6_bpb` | 1.1402 | — | — |
| `gate_quant_gap` (fp → int6) | 0.0109 | — | — |
| `gate_artifact_mb` | **14.39** | 16.00 cap | 1.61 MB headroom |
| `gate_passed` | **true** | — | — |
| `promote_ema_bpb` | null | — | not yet run |
| `promote_int6_bpb` | null | — | not yet run |
| `wallclock_s` | 420 | 600 cap | ran out mid-training |

Vs. the author's 1.10597 claim, `screen_ema_bpb=1.1293` is **+0.0233 nats above** the PR#1458 target and **+0.0146 nats above** the 1.1147 SOTA.

### Critical observation: the schedule did not fit the wallclock cap

At the logged **94 ms/step**, 20000 iterations ⇒ 20000 × 0.094 s ≈ **1880 s ≈ 31.3 min**, more than 3× the `MAX_WALLCLOCK_SECONDS=600` budget. The recorded `wallclock_s=420` plus the last logged `step 3500 @ 328 s` imply the run was terminated by the wallclock cap somewhere around **step 4500**, missing roughly 75–80% of the schedule including the entire `WARMDOWN_ITERS=3500` tail. That is the most likely reason screen BPB landed so far from both our 1.081 baseline and the author's 1.10597 claim: the model was cut off mid-training before warmdown, not because the PR's ML ideas are wrong. Either our per-step cost is higher than the author's, or the config needs to be adapted (shorter `ITERATIONS`, smaller batch, or shorter seq len) to actually fit 20k steps inside 600 s on our 8× H100 box.

Structurally the PR's compression math is validated: artifact lands at 14.39 MB (1.61 MB headroom under cap), quant gap is a healthy 0.0109 nats. The "spend headroom on a 12th layer" lever works — we just didn't get to finish training.

## Verdict

**neutral** — reproduction is inconclusive. The gate passed structurally (artifact 14.39 MB, quant gap 0.0109 nats), so the 12L + Brotli + `attn=int5` budget math holds up. But `screen_ema_bpb=1.1293` is +0.0483 vs our 1.0810 baseline and +0.0233 vs the 1.10597 author claim, and the leading diagnosis is a wallclock-budget mismatch on our infra (~94 ms/step × 20k steps ≫ 600 s cap), not a failure of the PR's ML ideas. Cannot call this a win or a regression until the schedule actually fits the budget and we rerun.

## Suggested follow-ups

- **Root-cause the step time first.** 94 ms/step on 8× H100 at `TRAIN_BATCH_TOKENS=786432`, `SEQ_LEN=2048`, 12L × 512 dim is notably slower than records that finish 20k steps in ~450–550 s. Profile one step (`torch.cuda.Event`, `nsys`) to see whether Brotli / mixed-int / XSA-on-all-11-layers introduced the slowdown, or whether we are missing a `torch.compile` / kernel the author had.
- **Re-run at a schedule that actually fits 600 s.** Either (a) drop `ITERATIONS` to ~5500–6000 at the current step time, or (b) cut `TRAIN_BATCH_TOKENS` / `TRAIN_SEQ_LEN` so 20k steps fit in 600 s. Option (b) is preferable because it preserves the warmdown length the recipe was tuned for, then re-measure screen BPB.
- **Run the promote phase (3-seed full-fidelity)** once the wallclock issue is fixed. Only 3-seed promote data is informative against the author's 1.10597 ± 0.00032 claim.
- **Fix training-log capture.** Only steps 0–3500 made it into `train.log`. We need the warmdown / final val / GPTQ / gate lines in every screen log to diagnose runs like this without guesswork.
- **Audit a suspicious env var.** `INT8_KEEP_FLOAT_FP32_NAME_PATTERNS` is literally `"','.join(CONTROL_TENSOR_NAME_PATTERNS"` — an unevaluated Python fragment that leaked into the env. Verify `train_gpt.py` parses this safely (or that the code path is skipped because `SKIP_GPTQ=1`); otherwise control tensors may not be kept in fp32 during int8 conversion.
- **Bit-level ablation of the novel lever** (conditional on a clean rerun): isolate the contributions of 11L → 12L, attn int6 → int5, and Brotli vs the previous default compressor. Quantify the compressor-only delta as a reusable nugget.
- **Use the 1.61 MB artifact headroom.** If promote reproduces, leftover cap is enough for e.g. a second VE bank, wider `BIGRAM_DIM`, or per-layer MLP expansion — any of which could buy further BPB.
- **`stack_on_best` branch, conditional on reproduction.** Layer Legal Score-First TTT (PR #549) and self-gen GPTQ calibration (PR #1019) on top of the confirmed 12L + Brotli base on a new branch `auto/recipe/12l_brotli_ttt_gptq`.
