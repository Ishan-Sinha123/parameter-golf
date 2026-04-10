# matrix_lr=0.06

## Hypothesis

Raising `MATRIX_LR` to `0.06` was expected to better fit the compressed
single-GPU tight-budget sweep (540s wallclock cap, ~1614/20000 steps
completed). Under such truncated training, a hotter matrix learning
rate could push the model further down the loss curve before the cap
terminates the run — this is the top arm of the tight-budget sweep.

## Configuration

| Env var | Value |
|---|---|
| `MATRIX_LR` | `0.06` |

Recipe: none (`recipe_id: null`, single env override on the
autoresearch-deploy baseline). Config observed in the log:

| Parameter | Value |
|---|---|
| `embed_lr` | 0.05 |
| `head_lr` | 0.0 |
| `matrix_lr` | 0.06 |
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

### Training trajectory (key log lines)

```
step:0/20000    val_loss:6.9357 val_bpb:4.1077
step:1000/20000 val_loss:2.3094 val_bpb:1.3677
step:1614/20000 val_loss:2.2195 val_bpb:1.3145 train_time:540029ms
stopping_early: wallclock_cap train_time:540029ms step:1614/20000
peak memory allocated: 10303 MiB reserved: 10622 MiB
Serialized model int8+zlib: 15269231 bytes (payload_ratio:3.91x)
Total submission size int8+zlib: 15316924 bytes
final_int8_zlib_roundtrip val_loss:2.2211 val_bpb:1.3155 eval_time:10872ms
final_int8_zlib_roundtrip_exact val_loss:2.22109921 val_bpb:1.31546006
```

### Metrics

| Metric | Value | Delta vs baseline (1.081) |
|---|---|---|
| screen_ema_bpb | 1.28204 | +0.20104 |
| raw val_bpb @ step 1614 | 1.3145 | +0.2335 |
| gate_int6_bpb (int8+zlib roundtrip) | 1.3155 | +0.2345 |
| quant_gap | 3.99e-05 | negligible |
| artifact size (int8+zlib) | 15,316,924 B | under 16 MB cap |
| gate_passed | **true** | — |
| peak memory | 10,303 MiB | — |

**Important caveat on the baseline delta.** The baseline `1.081` refers
to the current SOTA recipe trained under its full 8×H100 10-minute
budget. This sweep ran single-GPU with a 540s wallclock cap and
terminated at step 1614/20000 via `wallclock_cap`. Absolute BPB is
therefore not directly comparable with the SOTA; the sweep's purpose is
to rank `MATRIX_LR` values against its other arms under identical
truncated conditions.

## Verdict

**neutral.** Gate passed with a near-zero quant gap (3.99e-05) and
artifact comfortably under the 16 MB cap. screen_ema_bpb of 1.28204 is
~0.201 nats above the 1.081 SOTA baseline, which is expected for a
truncated single-GPU sweep point and does not constitute a regression
against the actual tight-budget arms. A single point in isolation
cannot declare a win — it must be compared against exp001/exp002 and
any other sweep arms before acting.

## Suggested follow-ups

- Compare screen_ema_bpb across all `matrix_lr_sweep_at_tight_budge`
  arms to pick the tight-budget winner and check whether 0.06 is the
  monotone top or there is an interior optimum.
- If 0.06 is the sweep winner, extend the sweep upward (0.07, 0.08) to
  locate the turnover point.
- Promote the best `MATRIX_LR` from the sweep into a full-budget run on
  the current SOTA recipe
  `rec_20260410_2026_04_09_sp8192_3layerrecur_parresid_q_3710821c`
  to test whether the tight-budget ranking transfers.
- Joint sweep of `MATRIX_LR` × `EMBED_LR` or `MATRIX_LR` × warmdown
  schedule, since these parameters plausibly interact in the truncated
  regime.
- Multi-seed (≥3) re-run at the winning `MATRIX_LR` to confirm the
  signal exceeds single-seed noise before treating it as actionable.
