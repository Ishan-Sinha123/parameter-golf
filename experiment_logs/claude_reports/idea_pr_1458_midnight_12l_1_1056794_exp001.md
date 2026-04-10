# PR#1458 midnight_12l_brotli_mixed_int

## Hypothesis

Reproduce PR#1458 (midnight_12l_brotli_mixed_int) on our own infra. The author claims a
mean of **1.10597 bpb** across 3 seeds (σ = 0.00032), which would be a ~0.0087-nat beat
over the 1.1147 SOTA — strong enough to be worth verifying before stacking anything on
top. The core novel lever is "spend compression headroom (Brotli + mixed-int attn=int5 /
mlp=int6 / embed=int8) on an extra transformer layer": 11L → 12L while staying under the
16 MB artifact cap. If reproduction lands within noise of the claim, the natural
follow-up is `stack_on_best`: layer Legal Score-First TTT (PR #549) and self-generated
GPTQ calibration (PR #1019) on top of this 12 L + Brotli base.

## Configuration

**Recipe:** `rec_20260410_pr1458_midnight_12l_brotli_mixed_int_dbd11f1a`
**Source:** PR#1458 (reproduction, `is_reproduction=true`)
**Script:** `autoresearch/baselines/rec_20260410_2026_03_31_parallelresiduals_minidepthre_4d594ac1/train_gpt.py`
**Commit:** `b862f3cd83601023495386128c3e87bac87ab41b` on `autoresearch-deploy`
**Stage:** screen (8× H100, `world_size=8`, `grad_accum_steps=1`)
**Seed:** 444
**Model params:** 29,317,732 (from `train.log`)
**Wallclock:** 420 s recorded in `experiment.json`

### Shape / architecture

| Key | Value |
| --- | --- |
| `NUM_LAYERS` | **12** (novel: +1 over 11 L precedent) |
| `MODEL_DIM` | 512 |
| `NUM_HEADS` / `NUM_KV_HEADS` | 8 / 4 (GQA) |
| `MLP_MULT` | 3.0 |
| `VOCAB_SIZE` | 1024 (SentencePiece BPE) |
| `TIE_EMBEDDINGS` | 1 |
| `XSA_LAST_N` | 11 (XSA on layers 1–11, layer 0 standard) |
| `ROPE_DIMS` / `ROPE_BASE` | 16 / 1e4 |
| `VE_ENABLED` / `VE_DIM` / `VE_LAYERS` | 1 / 128 / 9,10 |
| `BIGRAM_VOCAB_SIZE` / `BIGRAM_DIM` | 2048 / 112 |
| `LN_SCALE` | 1 |
| `LOGIT_SOFTCAP` | 30 |
| `NEGATIVE_SLOPE` | 0.5 (LeakyReLU²) |

### Training schedule

| Key | Value |
| --- | --- |
| `ITERATIONS` | 20000 |
| `WARMUP_STEPS` / `WARMDOWN_ITERS` | 20 / 3500 |
| `TRAIN_BATCH_TOKENS` / `TRAIN_SEQ_LEN` | 786432 / 2048 |
| `EVAL_SEQ_LEN` / `EVAL_STRIDE` | 2048 / 64 |
| `MAX_WALLCLOCK_SECONDS` | 600.0 |
| `LOADER_MODE` | coprime |

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
| `GPTQ_AR_SELFGEN` | 1 |
| `GPTQ_CALIB_SAMPLES` / `BATCH_SIZE` / `TEMPERATURE` | 64 / 8 / 0.8 |
| `LATE_QAT_THRESHOLD` | 0.15 |
| `QUANT_CLIP_RANGE` | 31 |
| `COMPRESSOR` | **brotli** |
| `TTT_ENABLED` | 0 |

## Results

### Training log excerpts (`train.log`, 59 lines — partial capture)

```
model_params:29317732
XSA:last_11 active_layers:[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
world_size:8 grad_accum_steps:1
attention_mode:gqa num_heads:8 num_kv_heads:4
tie_embeddings:True embed_lr:0.035 head_lr:0.0 matrix_lr:0.03 scalar_lr:0.025
train_batch_tokens:786432 train_seq_len:2048 iterations:20000 warmup_steps:20 max_wallclock_seconds:600.000
seed:444
gptq:reserving 0ms from training budget, effective=600000ms
step:0/20000  val_loss:6.9299 val_bpb:4.1043  train_time:0ms       step_avg:0.01ms
step:500/20000  train_loss:2.3033  train_time:47218ms  step_avg:94.44ms
step:1000/20000 train_loss:2.2197  train_time:93982ms  step_avg:93.98ms
step:1500/20000 train_loss:2.1124  train_time:140796ms step_avg:93.86ms
step:2000/20000 train_loss:2.0878  train_time:187689ms step_avg:93.84ms
step:2500/20000 train_loss:2.0308  train_time:234607ms step_avg:93.84ms
step:3000/20000 train_loss:2.0665  train_time:281528ms step_avg:93.84ms
step:3500/20000 train_loss:2.0667  train_time:328442ms step_avg:93.84ms
```

The log was captured only through step 3500 / 20000; no `VAL_LOSS_EVERY=4000` tick or
post-training GPTQ / gate lines made it into the slice. No warnings, no divergence, loss
trending normally. Step time ≈ **94 ms/step**.

### Gate metrics (from `experiment.json`)

| Metric | Value | Baseline (PR#1458 claim mean) | Δ vs baseline |
| --- | --- | --- | --- |
| `screen_ema_bpb` | **1.1293** | 1.10625 | **+0.02305** |
| `gate_int6_bpb` | 1.1402 | — | +0.03395 |
| `gate_quant_gap` (ema → int6) | 0.0109 | — | — |
| `gate_artifact_mb` | **14.39** | 16.00 cap | 1.61 MB headroom |
| `gate_passed` | ✅ **true** | — | — |
| `promote_ema_bpb` | null | — | not yet run |
| `promote_int6_bpb` | null | — | not yet run |

Gate passed structurally: the Brotli + mixed-int5-attn compression math works and the
artifact lands at 14.39 MB with 1.61 MB headroom, validating the "spend it on a 12th
layer" premise of PR#1458. The quant gap of 0.0109 nats (fp → int6) is healthy. But the
screen `ema_bpb` of 1.1293 is ~0.023 nats *above* the author's 1.10597 claim and ~0.018
nats above the 1.1147 SOTA.

### Critical observation: the run cannot finish within the 10-min cap on our infra

At the logged **94 ms/step**, 20000 iterations ⇒ 20000 × 0.094 s ≈ **1880 s ≈ 31.3
min**, which is more than 3× the `MAX_WALLCLOCK_SECONDS=600` budget. The recorded
`wallclock_s=420` plus the final logged step 3500 @ 328 s imply the run was killed by
the wallclock cap somewhere around step 4000–4500, missing **~80 % of the schedule**,
including the entire `WARMDOWN_ITERS=3500` tail. That is almost certainly why the screen
BPB is so far from the claim: the model was cut off mid-training before warmdown, not
because the PR recipe itself is wrong. Either our per-step cost is higher than the
author's, or the config needs to be adapted (shorter `ITERATIONS` / smaller batch / seq
len) to actually fit 20000 steps inside 600 s on our 8× H100 box.

## Verdict

**neutral** — The reproduction is inconclusive because the run never reached its warmdown.
The gate passed on a structural level (artifact 14.39 MB, quant gap 0.0109) so the
12 L + Brotli + attn-int5 budget math holds up. But the screen `ema_bpb` of 1.1293 did
not land near 1.10597, and the leading diagnosis is a wallclock-budget mismatch on our
infra (~94 ms/step × 20 k steps > 600 s cap), not a failure of the PR's ML ideas.
Cannot call reproduction either a win or a regression until we fit the schedule into the
actual budget and rerun.

## Suggested follow-ups

- **Root-cause the step time first.** 94 ms/step on 8× H100 at `TRAIN_BATCH_TOKENS=786432`, `SEQ_LEN=2048`, 12 L × 512 dim is noticeably slower than comparable records that finish 20 k steps in ~450–550 s. Profile one step (kernel timeline, `torch.cuda.Event`, `nsys`) to see whether Brotli / mixed-int / XSA-on-all-layers introduced a slowdown, or whether we are missing a compile / kernel that the author had.
- **Re-run at a schedule that actually fits 600 s.** Either (a) drop `ITERATIONS` to ~5500–6000 at the current step-time, or (b) cut `TRAIN_BATCH_TOKENS` / `TRAIN_SEQ_LEN` so 20 k steps fit in 600 s. Option (b) is preferable because it preserves the warmdown length the recipe was tuned for. Then re-measure screen BPB.
- **Run the promote phase (3-seed full-fidelity)** once the wallclock issue is fixed. Only promote data against the author's 1.10597 ± 0.00032 claim is informative for the reproduction verdict.
- **Fix training-log capture.** Only steps 0–3500 made it into `train.log`. We need the warmdown / final val / GPTQ / gate lines in every screen log to diagnose runs like this without guesswork.
- **Audit a suspicious env var.** `INT8_KEEP_FLOAT_FP32_NAME_PATTERNS` is literally `"','.join(CONTROL_TENSOR_NAME_PATTERNS"` — an unevaluated Python fragment that leaked into the env. Verify `train_gpt.py` parses this safely (or that the code path is skipped because `SKIP_GPTQ=1`); otherwise the control tensors might not be kept in fp32 during int8 conversion.
- **Bit-level ablation of the novel lever** (conditional on a clean rerun): isolate the contributions of 11 L → 12 L, attn int6 → int5, and Brotli vs the previous default compressor. The headline is "they only work as a bundle" — confirm with a small sweep and quantify the compressor-only delta as a reusable nugget.
- **Use the 1.61 MB artifact headroom.** If promote reproduces, the leftover cap is enough for e.g. a second VE bank, a wider `BIGRAM_DIM`, or per-layer MLP expansion, any of which could buy further BPB.
- **`stack_on_best` branch, conditional on reproduction.** Layer Legal Score-First TTT (PR #549) and self-gen GPTQ calibration (PR #1019) on top of the confirmed 12 L + Brotli base, on a new branch `auto/recipe/12l_brotli_ttt_gptq`.
