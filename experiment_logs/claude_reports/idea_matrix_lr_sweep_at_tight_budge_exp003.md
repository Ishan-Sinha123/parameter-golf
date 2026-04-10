# matrix_lr=0.06

## Hypothesis

Raising `MATRIX_LR` to `0.06` was expected to better fit the compressed
tight-budget sweep (540s single-GPU wallclock cap, ~1614/20000 steps
completed before `stopping_early: wallclock_cap`). Under truncated
training a hotter matrix learning rate could push the model further
down the loss curve before the cap fires, making it the top arm of the
`matrix_lr_sweep_at_tight_budge` series.

## Configuration

Single env override on the `autoresearch-deploy` baseline (no recipe
forked — `recipe_id: null`).

| Env var | Value |
|---|---|
| `MATRIX_LR` | `0.06` |

Effective training config observed in the log header:

| Parameter | Value |
|---|---|
| `embed_lr` | 0.05 |
| `head_lr` | 0.0 |
| `matrix_lr` | **0.06** |
| `scalar_lr` | 0.04 |
| `model_params` | 17,059,912 |
| `world_size` | 1 |
| `grad_accum_steps` | 8 |
| attention | GQA, 8 heads, 4 KV |
| `tie_embeddings` | True |
| `train_seq_len` | 1024 |
| `train_batch_tokens` | 524,288 |
| `iterations` (cap) | 20,000 |
| `warmup_steps` | 20 |
| `max_wallclock_seconds` | 540 |
| `seed` | 1337 |

## Results

### Key log lines

```
step:0/20000    val_loss:6.9357 val_bpb:4.1077
step:1000/20000 val_loss:2.3094 val_bpb:1.3677
step:1614/20000 val_loss:2.2195 val_bpb:1.3145 train_time:540029ms
stopping_early: wallclock_cap train_time:540029ms step:1614/20000
peak memory allocated: 10303 MiB reserved: 10622 MiB
Serialized model int8+zlib: 15269231 bytes (payload_ratio:3.91x)
Total submission size int8+zlib: 15316924 bytes
final_int8_zlib_roundtrip val_loss:2.2211 val_bpb:1.3155
final_int8_zlib_roundtrip_exact val_loss:2.22109921 val_bpb:1.31546006
```

No warnings in log. Run terminated on the expected wallclock cap, not
on divergence.

### Metrics

Baseline for delta: `1.10625353`.

| Metric | Value | Δ vs baseline |
|---|---|---|
| screen_ema_bpb | 1.28204 | **+0.17579** |
| raw val_bpb @ step 1614 | 1.31460 | +0.20835 |
| gate_int6_bpb (int8+zlib roundtrip) | 1.31546 | +0.20921 |
| quant_gap | 3.99e-05 | ~0 |
| artifact size (int8+zlib) | 15,316,924 B | under 16 MB cap |
| gate_passed | **true** | — |
| peak memory | 10,303 MiB | — |

**Caveat on the baseline delta.** The baseline `1.10625353` is the
full-budget SOTA-class reference. This arm trained single-GPU with a
540s wallclock cap and hit the cap at step 1614/20000. Absolute BPB is
therefore not comparable with a full 10-minute 8×H100 baseline — this
run is only meaningful as one arm of the tight-budget sweep, ranked
against exp001 and exp002.

## Verdict

**neutral.** Gate passed with a near-zero quant gap (3.99e-05) and
artifact comfortably under the 16 MB cap. screen_ema_bpb is ~0.176 nats
above the `1.10625353` reference, which is expected for a truncated
single-GPU sweep point and does not imply a regression against the
actual sweep arms. A single point in isolation cannot declare a win —
it must be compared against exp001/exp002 before acting.

## Suggested follow-ups

- Compare `screen_ema_bpb` across all `matrix_lr_sweep_at_tight_budge`
  arms to identify the tight-budget optimum and whether 0.06 is a
  monotone top or an interior point.
- If 0.06 wins the sweep, extend upward (0.07, 0.08) to locate the
  turnover and bracket the maximum.
- Promote the winning `MATRIX_LR` into a full-budget run on the current
  SOTA recipe
  `rec_20260410_2026_04_09_sp8192_3layerrecur_parresid_q_3710821c` to
  test whether the tight-budget ranking transfers to the 10-min regime.
- Joint sweep `MATRIX_LR` × `EMBED_LR` and `MATRIX_LR` × warmdown
  schedule — these parameters plausibly interact in the truncated
  regime.
- Multi-seed (≥3) re-run at the winning `MATRIX_LR` to confirm the
  signal exceeds single-seed noise before treating it as actionable.
