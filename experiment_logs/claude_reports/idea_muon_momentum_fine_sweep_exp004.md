# muon_mom=0.98

## Hypothesis

Momentum 0.98 may be optimal for the Muon optimizer at this screening
wallclock. This is the fourth run in a fine sweep (`exp001`–`exp004`) walking
momentum upward from the 0.95 default; prior sweep legs showed diminishing
returns as momentum rose, and this run tests whether 0.98 continues the trend
or overshoots and starts hurting short-run optimization dynamics.

## Configuration

| Env var         | Value                     |
|-----------------|---------------------------|
| `MUON_MOMENTUM` | `0.98` (default: `0.95`)  |

- Recipe: *(none — single env-var override on baseline screening recipe)*
- Source ref: *(none)*  ·  Reproduction: no

Other relevant config from `train.log`:

- `model_params`: 17,059,912 (~17.06M)
- `attention_mode`: gqa (8 heads, 4 kv heads), tied embeddings
- `train_batch_tokens`: 524,288; `train_seq_len`: 1024; `iterations`: 20,000
- `warmup_steps`: 20; `max_wallclock_seconds`: **540**
- `embed_lr`: 0.05, `matrix_lr`: 0.04, `scalar_lr`: 0.04, `head_lr`: 0.0
- `world_size`: 1; `grad_accum_steps`: 8; `seed`: 1337

## Results

Training stopped early at step 1617 / 20000 when the 540 s wallclock cap
hit. Step avg ≈ 334 ms. No NaN / divergence / stability warnings.

Key log lines (`experiment_logs/idea_muon_momentum_fine_sweep/idea_muon_momentum_fine_sweep_exp004/train.log`):

```
step:0/20000     val_loss:6.9357 val_bpb:4.1077
step:1000/20000  val_loss:2.3082 val_bpb:1.3671  train_time:333929ms step_avg:333.93ms
step:1617/20000  val_loss:2.2151 val_bpb:1.3119  train_time:540065ms step_avg:333.99ms
stopping_early: wallclock_cap train_time:540065ms step:1617/20000
peak memory allocated: 10303 MiB reserved: 10622 MiB
Serialized model int8+zlib: 15259302 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
final_int8_zlib_roundtrip val_loss:2.2168 val_bpb:1.3129 eval_time:10853ms
final_int8_zlib_roundtrip_exact val_loss:2.21677284 val_bpb:1.31289774
```

| Metric                                 | Value          | Δ vs baseline (1.10625353) |
|----------------------------------------|----------------|----------------------------|
| val_bpb @ step 1000 (fp)               | 1.36710        | +0.26085                   |
| val_bpb @ step 1617 (fp)               | 1.31190        | +0.20565                   |
| `screen_ema_bpb`                       | **1.29012**    | **+0.18386**               |
| `gate_int6_bpb` (int8+zlib roundtrip)  | 1.31290        | +0.20664                   |
| `gate_quant_gap`                       | ~2.26e-6 (≈ 0) | —                          |
| Artifact size (int8+zlib)              | 15.26 MB       | under 16 MB cap            |
| Peak GPU memory                        | 10,303 MiB     | —                          |
| `gate_passed`                          | ✅ true        | —                          |
| `promote_ema_bpb`                      | *(not run)*    | —                          |
| `promote_int6_bpb`                     | *(not run)*    | —                          |

**Apples-to-apples caveat:** the 1.10625 baseline is a longer-budget multi-GPU
SOTA-track number; this run is a single-GPU 540 s screening gate on the
baseline recipe, so the +0.18 nat gap vs baseline is expected and is **not** a
real regression of the momentum knob. The meaningful comparison is
horizontally against `exp001`–`exp003` on the same gate.

## Verdict

**neutral**

The gate passed cleanly, the int8+zlib quant gap is effectively zero (~2e-6
nats), and the artifact is comfortably under the 16 MB cap. However, no
promotion metrics were recorded (`promote_ema_bpb: null`), so we have no
full-budget signal that 0.98 is actually the sweep winner. Against the
full-budget baseline, this screen is 0.18 nats behind — expected for a
1-GPU / 540 s run, but means no record claim is possible from this leg alone.
Neither a win nor a regression; awaits within-sweep comparison.

## Suggested follow-ups

- Compare `screen_ema_bpb` across all four legs (`exp001`–`exp004`) to pick
  the within-sweep winner before allocating any full-budget promote runs.
- Multi-seed (≥3) the top 1–2 momentum values at the same screening wallclock
  to measure the seed-level noise floor and confirm whether differences
  exceed the 5e-3 nat PR bar.
- Probe one step further (`MUON_MOMENTUM=0.99`) to confirm saturation vs.
  reversal at very high momentum.
- Try a momentum warmup schedule (e.g. `0.85 → 0.98` over the first ~100
  steps) — late-stage high momentum sometimes beats fixed high momentum on
  short runs.
- Port the sweep winner onto the current SOTA chain in a stacking experiment
  (XSA + self-gen GPTQ + LeakyReLU²); single-GPU screening cannot tell us
  whether the momentum optimum transfers to the 8×H100 / 10-minute target.
- Couple momentum with a `MUON_WD` sweep — momentum and weight decay
  interact, and a higher momentum typically prefers a slightly different
  effective WD.
