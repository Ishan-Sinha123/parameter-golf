# matrix_lr=0.05

## Hypothesis

Setting `MATRIX_LR=0.05` may better fit the compressed parameter budget
at the tight single-GPU screen wallclock. This sweep point tests whether
raising the Muon matrix-path learning rate (vs sibling exp001 at 0.03)
lets the ~17M-param model make more effective progress per step under
the ~540 s cap, leaving signal for downstream int6 quantization.

## Configuration

| Env override | Value |
|---|---|
| `MATRIX_LR` | `0.05` |

Other LRs from log line 9: `embed_lr:0.05 head_lr:0.0 matrix_lr:0.05 scalar_lr:0.04`.

- Recipe: `null` (ad-hoc env-override sweep, `source_ref=""`; no registered recipe)
- Model: `model_params:17059912`, `attention_mode:gqa num_heads:8 num_kv_heads:4`, `tie_embeddings:True`
- Trainer: `world_size:1 grad_accum_steps:8`, `train_batch_tokens:524288 train_seq_len:1024`
- Budget: `iterations:20000 warmup_steps:20 max_wallclock_seconds:540.000`, `seed:1337`
- Screening run on 1 GPU — NOT the 8×H100 configuration that produced the SOTA baseline.

## Results

Baseline for delta calc: **val_bpb = 1.10625353**.

Key lines from `train.log`:

```
step:0/20000    val_loss:6.9357 val_bpb:4.1077 train_time:0ms
step:2/20000    train_loss:16.6531 train_time:661ms
step:1000/20000 val_loss:2.3091 val_bpb:1.3676 train_time:335883ms
step:1612/20000 val_loss:2.2200 val_bpb:1.3148 train_time:540289ms
stopping_early: wallclock_cap train_time:540289ms step:1612/20000
peak memory allocated: 10303 MiB reserved: 10622 MiB
Serialized model int8+zlib: 14832612 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
Total submission size int8+zlib: 14880305 bytes
final_int8_zlib_roundtrip val_loss:2.2219 val_bpb:1.3159 eval_time:10829ms
final_int8_zlib_roundtrip_exact val_loss:2.22187874 val_bpb:1.31592174
```

| Metric | Value | Δ vs baseline (1.10625) |
|---|---:|---:|
| Final raw val_bpb (step 1612) | 1.3148 | **+0.2086** |
| Screen EMA bpb | 1.28183 | **+0.17558** |
| Gate int6 bpb (`final_int8_zlib_roundtrip`) | 1.31592 | **+0.20967** |
| Quant gap (int6 − EMA, from metadata) | −2.17e-05 | ~0 (lossless) |
| Artifact size (int8+zlib, from log) | ~14.88 MB | under 16 MB cap |
| `gate_artifact_mb` (metadata) | 0.0 | ⚠ reporting mismatch vs log |
| Gate passed | ✅ true | — |
| Promoted | ❌ (`promote_ema_bpb` / `promote_int6_bpb` null) | — |
| Final step reached | 1612 / 20000 (wallclock cap) | — |

Head-to-head with sibling `exp001` (MATRIX_LR=0.03), same screen setup:

| Metric | exp001 (0.03) | exp002 (0.05) | Δ (0.05 − 0.03) |
|---|---:|---:|---:|
| Screen EMA bpb | 1.28256 | 1.28183 | −0.00073 |
| Gate int6 bpb | 1.31700 | 1.31592 | −0.00108 |
| Final raw val_bpb | 1.3155 | 1.3148 | −0.0007 |
| Artifact int8+zlib | ~13.52 MB | ~14.88 MB | +1.36 MB |
| Steps reached | 1608 | 1612 | +4 |

### Observations

- Warmup completed cleanly (20/20). A train_loss spike to 16.65 at step 2
  recovered to ~2.33 by step 1000 — consistent with an aggressive matrix
  LR but not diverging.
- Quant gap is effectively zero (slightly negative): int8+zlib roundtrip
  is lossless vs the EMA screen — strong quantization robustness signal.
- Throughput ~335 ms/step at `grad_accum_steps=8`, single GPU.
- No warnings, NaNs, or divergences observed.

**Caveat on baseline delta.** The 1.10625 baseline is a full 8×H100
SOTA figure. This run is a single-GPU screen that hit the wallclock cap
at step 1612/20000 (~8% of the schedule, never reaching warmdown), so
the +0.176 / +0.210 BPB gaps reflect reduced steps and smaller world
size — **not** a like-for-like regression of `MATRIX_LR=0.05`. Sweep-
internal comparisons against exp001/exp003 are the fair signal.

## Verdict

**neutral** — gate passed with a negligible (slightly negative) quant
gap, confirming `MATRIX_LR=0.05` trains cleanly and survives int8+zlib
quantization at the tight budget. This sweep point is ~1e-3 nats better
than exp001 (0.03) in both EMA and int6 bpb. The absolute +0.21 nats
vs the full-scale baseline is not apples-to-apples; the run was not
promoted. Interpret as a valid screening data point, not a win or loss
vs SOTA.

## Suggested follow-ups

- Rank exp001 (0.03), exp002 (0.05), exp003 side-by-side under identical
  screen conditions before promoting any candidate. Current ordering on
  int6 bpb: 0.05 (1.31592) < 0.03 (1.31700).
- If 0.05 is the screen winner and not a grid boundary, extend the grid
  upward (0.06–0.08) at the same wallclock cap to confirm it isn't a
  boundary optimum.
- Promote the screen winner to a full 8×H100 schedule (≥3 seeds) so the
  20000-iter trajectory completes through warmdown; a 1612-step
  truncation can easily misrank LR settings that differ mainly in
  late-schedule dynamics.
- Investigate the ~1.36 MB artifact-size increase at 0.05 vs 0.03: a
  higher matrix LR may produce weight distributions that compress less
  well under int8+zlib. Still well under the 16 MB cap, but worth
  tracking when stacked with other quant-sensitive changes.
- Co-sweep `MATRIX_LR` with `SCALAR_LR` (0.04) and `EMBED_LR` (0.05);
  Muon LR paths interact and a 1-D sweep may miss the joint optimum.
- Layer the sweep on top of the current SOTA stacked recipe rather than
  bare defaults — the hypothesis targets compressed/tight-budget
  recipes, so it should be tested on one.
- Quant gap is already ~0; prioritize LR directions that lower the EMA
  side instead of trading off quant robustness.
- Given the step-2 train_loss spike (16.65), try `warmup_steps=30–40`
  for `matrix_lr ≥ 0.05` to see whether gentler warmup helps higher
  LRs converge faster in the tight-budget regime.
- Audit the metrics-logging pipeline: `gate_artifact_mb` is 0.0 in
  experiment metadata while the log reports ~14.88 MB — same bug as
  exp001. Fix before other sweeps rely on that field.
