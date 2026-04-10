# matrix_lr=0.03

## Hypothesis

A lower matrix learning rate of 0.03 may better fit the compressed training budget. With fewer optimization steps available under the tight wallclock cap, a smaller matrix LR could avoid overshooting and yield tighter final loss, especially after int6 quantization.

## Configuration

| Parameter | Value |
|---|---|
| `MATRIX_LR` | `0.03` |
| `embed_lr` | 0.05 |
| `head_lr` | 0.0 |
| `scalar_lr` | 0.04 |
| GPUs | 1 (`world_size:1`, `grad_accum_steps:8`) |
| Wallclock cap | 540 s |
| Scheduled iterations | 20000 |
| Steps completed | **1608 / 20000** (wallclock-capped) |
| Model params | 17,059,912 |
| Attention | GQA (8 heads, 4 KV heads) |
| Tokenizer | `fineweb_1024_bpe.model` (sentencepiece, 1024 vocab) |
| Seed | 1337 |
| Recipe | None (single env override on default `train_gpt.py`) |

## Results

Baseline for delta calculation: **val_bpb = 1.10625353**.

| Metric | Value | Δ vs baseline (1.10625) |
|---|---|---|
| Final raw val_bpb (step 1608) | 1.3155 | +0.2092 |
| Screen EMA BPB | 1.28256 | +0.17630 |
| Int8+zlib roundtrip BPB | 1.31700 | +0.21074 |
| Int6 gate BPB | 1.317 | +0.21075 |
| Quant gap (gate) | 4.75e-06 | — |
| Artifact size (int8+zlib) | 13,522,894 B ≈ 13.52 MB (log) / 0.0 (metadata) | under 16 MB cap |
| Gate passed | true | — |
| Promoted | no (`promote_ema_bpb` / `promote_int6_bpb` both `null`) | — |

**Key log lines:**

```
model_params:17059912
world_size:1 grad_accum_steps:8
attention_mode:gqa num_heads:8 num_kv_heads:4
tie_embeddings:True embed_lr:0.05 head_lr:0.0 matrix_lr:0.03 scalar_lr:0.04
train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:20 max_wallclock_seconds:540.000
step:1000/20000 val_loss:2.3094 val_bpb:1.3678 train_time:336788ms
step:1608/20000 val_loss:2.2212 val_bpb:1.3155 train_time:540148ms step_avg:335.91ms
stopping_early: wallclock_cap train_time:540148ms step:1608/20000
Serialized model int8+zlib: 13475201 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
Total submission size int8+zlib: 13522894 bytes
final_int8_zlib_roundtrip val_loss:2.2237 val_bpb:1.3170 eval_time:10851ms
final_int8_zlib_roundtrip_exact val_loss:2.22369131 val_bpb:1.31699525
```

**Critical caveat:** run hit `stopping_early: wallclock_cap` at step **1608 / 20000** (~8% of scheduled iterations) on a single GPU at ~336 ms/step. This screen run never reached the final warmdown phase, so the absolute BPB cannot fairly be compared against the 1.10625 baseline (which presumably trained on 8×H100 SXM to completion). The quant gap (~5e-6) and artifact footprint (13.52 MB under the 16 MB cap) are clean.

## Verdict

**neutral** — the gate technically passed (clean quant gap, under artifact cap), but the screen EMA BPB (1.28256) is +0.176 nats above the 1.10625 baseline and the run was not promoted. Because the single-GPU screen only reached 1608 / 20000 scheduled steps (wallclock-capped), the absolute delta is not a fair head-to-head comparison against full-schedule baselines — it only tells us that `MATRIX_LR=0.03` did not break out under screening conditions. The result should be interpreted as a sweep data point within its sibling runs (exp002 / exp003), not as an independent regression call.

## Suggested follow-ups

- Compare exp001 directly against sibling sweep points (exp002, exp003) under identical single-GPU screen conditions to establish relative ordering of `MATRIX_LR` values before promoting any candidate.
- Re-run the leading screen candidate on the full 8×H100 SXM configuration so the 20000-iteration schedule actually completes; single-GPU truncated screens are not a fair test of LR sweet spots under the real training budget.
- Layer the `MATRIX_LR` sweep on top of the current SOTA recipe (e.g. `rec_20260410_..._3layerrecur_parresid_q`) rather than the bare default, since the hypothesis is specifically about tight-budget compressed recipes.
- Bracket with coupled LRs: explore joint sweeps of `matrix_lr` × `scalar_lr` × `embed_lr`, since they may trade off against each other at this scale.
- Investigate whether the optimal matrix LR shifts as step count scales (8-GPU budget → ~8× more completed steps), which would confirm or refute the `compressed budget` interaction hypothesis.
- Audit the metrics-logging pipeline: `gate_artifact_mb` is reported as 0.0 in the experiment metadata while the training log clearly shows a 13.52 MB artifact. This is a reporting bug worth tracking down before other sweeps rely on that field.
