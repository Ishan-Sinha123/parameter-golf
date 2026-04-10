# muon_mom=0.98

## Hypothesis

Momentum 0.98 may be optimal for the Muon optimizer at this screening
wallclock. This is the fourth run in a fine sweep (`exp001`–`exp004`)
walking momentum values upward from the 0.95 default; prior runs suggested
a monotonic trend favoring higher momentum with diminishing returns, and
this run tests whether 0.98 continues that trend or overshoots.

## Configuration

| Env var | Value |
|---|---|
| `MUON_MOMENTUM` | `0.98` (default: `0.95`) |

Other relevant config (from `train.log`):

- `model_params`: 17,059,912
- `attention_mode`: gqa (8 heads, 4 kv heads), tied embeddings
- `train_batch_tokens`: 524,288; `train_seq_len`: 1024; `iterations`: 20,000
- `warmup_steps`: 20; `max_wallclock_seconds`: **540** (note: 60 s longer
  than `exp001`–`exp003`, which used 480 s)
- `embed_lr`: 0.05, `matrix_lr`: 0.04, `scalar_lr`: 0.04
- `world_size`: 1; `grad_accum_steps`: 8
- `seed`: 1337
- `recipe_id`: *(none — single env-var override on baseline screening recipe)*
- `commit`: 739fb23ec1d1237c00d7075f9616a9c62ced1f2a (autoresearch-deploy)
- Host / GPU: 206.125.32.60 / GPU 2

## Results

Training stopped early at step 1617 / 20000 when the 540 s wallclock cap
hit. Step avg ≈ 334 ms.

Key log lines:

```
step:1000/20000 val_loss:2.3082 val_bpb:1.3671 train_time:333929ms step_avg:333.93ms
step:1617/20000 val_loss:2.2151 val_bpb:1.3119 train_time:540065ms step_avg:333.99ms
stopping_early: wallclock_cap train_time:540065ms step:1617/20000
peak memory allocated: 10303 MiB reserved: 10622 MiB
Serialized model int8+zlib: 15259302 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
final_int8_zlib_roundtrip val_loss:2.2168 val_bpb:1.3129 eval_time:10853ms
final_int8_zlib_roundtrip_exact val_loss:2.21677284 val_bpb:1.31289774
```

| Metric | Value | Δ vs baseline (1.0810) |
|---|---|---|
| val_bpb @ step 1000 (fp) | 1.3671 | +0.2861 |
| val_bpb @ step 1617 (fp) | 1.3119 | +0.2309 |
| `screen_ema_bpb` | 1.29012 | +0.20912 |
| `gate_int6_bpb` (int8+zlib roundtrip) | **1.31290** | **+0.23190** |
| `gate_quant_gap` | ~2.26e-6 (negligible) | — |
| Artifact size (int8+zlib) | 15.26 MB | under 16 MB cap |
| Peak memory | 10,303 MiB | — |
| Gate passed | ✅ true | — |

**Apples-to-apples caveat:** the 1.0810 baseline is the multi-GPU SOTA
from the recipe chain; this run is a single-GPU 540 s screening gate on
the tiny-vocab baseline recipe, so the +0.23 delta is expected and not a
real regression. The meaningful comparison is against sibling sweep runs
`exp001`–`exp003` on the same gate (and `exp004` additionally benefits
from the extra 60 s of wallclock, reaching ~180 more steps than its
siblings).

## Verdict

**neutral** — the gate passed cleanly and the int6 quant gap is
effectively zero (~2e-6 nats), but the absolute gate_int6_bpb cannot be
fairly ranked against `exp001`–`exp003` because `exp004` ran for 540 s
instead of 480 s. No promotion metrics were recorded
(`promote_ema_bpb: null`), so 0.98 was not selected for a larger-scale
rerun. The `screen_ema_bpb` (1.29012) is essentially tied with the
0.96 / 0.97 siblings, suggesting momentum has already saturated.

## Suggested follow-ups

- Re-run `MUON_MOMENTUM=0.98` at an iso-wallclock 480 s budget to get a
  fair head-to-head against `exp001`–`exp003`.
- Probe one step further (`0.99`) to confirm saturation vs. reversal at
  very high momentum.
- Multi-seed (≥3) the best momentum on the screening gate to estimate
  variance before committing a full 8×H100 promotion slot.
- Port the winning momentum into the current SOTA chain
  (`rec_20260410_*_sp8192_3layerrecur_parresid_q_*`, val_bpb 1.0810) in
  a proper stacking experiment; single-GPU 540 s gates cannot tell us
  whether the momentum optimum transfers to the real 10-minute 8×H100
  target.
- Couple with a Muon weight-decay sweep — momentum and WD interact, and
  a higher momentum often wants slightly different effective WD.
