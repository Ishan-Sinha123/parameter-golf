# matrix_lr=0.03

## Hypothesis

A matrix LR of 0.03 (vs the default used in recent recipes) may better
fit the compressed parameter budget at tight wallclock. The sweep
screens whether lowering the Muon matrix-path learning rate helps a
~17M-param model converge more efficiently under the screening budget,
leaving more signal for downstream int6 quantization.

## Configuration

| Env override | Value |
|---|---|
| `MATRIX_LR` | `0.03` |

Other LRs from log line 9: `embed_lr:0.05 head_lr:0.0 matrix_lr:0.03 scalar_lr:0.04`.

- Recipe: `null` (ad-hoc env-override sweep, not attached to a registered recipe)
- Model: `model_params:17059912`, `attention_mode:gqa num_heads:8 num_kv_heads:4`, `tie_embeddings:True`
- Trainer: `world_size:1 grad_accum_steps:8`, `train_batch_tokens:524288 train_seq_len:1024`
- Budget: `iterations:20000 warmup_steps:20 max_wallclock_seconds:540.000`, `seed:1337`
- Screening run on 1 GPU — not the 8×H100 configuration used for the SOTA baseline.

## Results

Baseline for delta calc: **val_bpb = 1.10625353**.

Key lines from `train.log`:

```
step:0/20000    val_loss:6.9357 val_bpb:4.1077
step:1000/20000 val_loss:2.3094 val_bpb:1.3678 train_time:336788ms
step:1608/20000 val_loss:2.2212 val_bpb:1.3155 train_time:540148ms
stopping_early: wallclock_cap train_time:540148ms step:1608/20000
Serialized model int8+zlib: 13475201 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
Total submission size int8+zlib: 13522894 bytes
final_int8_zlib_roundtrip_exact val_loss:2.22369131 val_bpb:1.31699525
```

| Metric | Value | Δ vs baseline (1.10625) |
|---|---|---|
| Final raw val_bpb (step 1608) | 1.3155 | +0.2093 |
| Screen EMA bpb | 1.28256 | +0.17630 |
| Gate int6 bpb | 1.31700 | +0.21075 |
| Quant gap (int6 − EMA) | ~4.75e-06 | — |
| Artifact size (int8+zlib, log) | ~13.52 MB | under 16 MB cap |
| `gate_artifact_mb` (metadata) | 0.0 | ⚠ metadata vs log mismatch |
| Gate passed | ✅ true | — |
| Promoted | ❌ (`promote_ema_bpb` / `promote_int6_bpb` null) | — |
| Final step reached | 1608 / 20000 (wallclock cap) | — |

**Caveat:** single-GPU screen, wallclock-capped at 1608/20000 steps
(~8% of the scheduled schedule, never reached warmdown). The absolute
bpb is NOT directly comparable to the 1.10625 baseline that assumes a
full 8×H100 schedule. The useful signals from this run are: (a) the
screen gate passed, (b) quant gap is essentially zero, (c) artifact
fits comfortably under the 16 MB cap.

## Verdict

**neutral** — gate passed with a negligible quant gap, but the screen
EMA bpb is +0.176 nats above baseline and the run did not promote.
Because this is a truncated single-GPU sweep point, the absolute delta
is not a fair head-to-head vs the 1.10625 baseline. Interpret as one
point in an LR sweep; promotion should come from ranking against
sibling runs at matched conditions, not from this number alone.

## Suggested follow-ups

- Compare against sibling sweep points (e.g. `MATRIX_LR` 0.02 / 0.04 /
  0.05) under identical single-GPU screen conditions to rank values
  before promoting any candidate.
- If 0.03 is not an interior minimum of the sweep, widen the grid
  rather than promote on a boundary point.
- Re-run the screen winner on the full 8×H100 schedule so the
  20000-iter trajectory completes; a 1608-step truncation can easily
  misrank LR settings that differ mainly in warmdown behavior.
- Layer the sweep on top of the current SOTA stacked recipe rather
  than bare defaults — the hypothesis specifically targets compressed
  / tight-budget recipes, so it should be tested on one.
- Since the quant gap is already ~0, prioritize LR directions that
  lower the EMA side of the gap instead of trading off quant
  robustness.
- Co-sweep `matrix_lr` with `scalar_lr` (and possibly `embed_lr`);
  these Muon LR paths often interact and a 1-D sweep can miss the
  joint optimum.
- Audit the metrics pipeline: `gate_artifact_mb=0.0` in experiment
  metadata while the log reports a 13.52 MB int8+zlib artifact. Fix
  before other sweeps rely on that field.
