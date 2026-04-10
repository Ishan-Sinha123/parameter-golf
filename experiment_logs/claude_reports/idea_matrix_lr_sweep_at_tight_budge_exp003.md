# matrix_lr=0.06

## Hypothesis

Under the tight-budget screening harness (540 s wallclock cap, single-GPU,
stopped at step 1614/20000), a matrix learning rate of **0.06** may better
fit the compressed step budget than the default. `MATRIX_LR` governs the
hidden-to-hidden linear weights updated by Muon; with far fewer usable
updates than the nominal 20k iteration plan, a slightly hotter LR could
move those weights further along their useful direction before the cap
fires, lowering the screen EMA BPB relative to sibling arms of the
`idea_matrix_lr_sweep_at_tight_budge` sweep.

## Configuration

Single env override on the active baseline recipe. No recipe fork
(`recipe_id: null`).

| Env var | Value |
|---|---|
| `MATRIX_LR` | `0.06` |

Effective config observed in the log header:

| Parameter | Value |
|---|---|
| `model_params` | 17,059,912 |
| `attention_mode` | `gqa` (8 heads, 4 KV) |
| `tie_embeddings` | `True` |
| `embed_lr` | 0.05 |
| `head_lr` | 0.0 |
| `matrix_lr` | **0.06** |
| `scalar_lr` | 0.04 |
| `train_seq_len` | 1024 |
| `train_batch_tokens` | 524,288 |
| `grad_accum_steps` | 8 |
| `world_size` | 1 |
| `iterations` (cap) | 20,000 |
| `warmup_steps` | 20 |
| `max_wallclock_seconds` | 540 |
| `seed` | 1337 |
| tokenizer | sentencepiece `fineweb_1024_bpe.model` |

## Results

Key lines from
`experiment_logs/idea_matrix_lr_sweep_at_tight_budge/idea_matrix_lr_sweep_at_tight_budge_exp003/train.log`:

```
step:0/20000    val_loss:6.9357 val_bpb:4.1077
step:1000/20000 val_loss:2.3094 val_bpb:1.3677 train_time:334788ms step_avg:334.79ms
step:1614/20000 val_loss:2.2195 val_bpb:1.3145 train_time:540029ms step_avg:334.59ms
stopping_early: wallclock_cap train_time:540029ms step:1614/20000
peak memory allocated: 10303 MiB reserved: 10622 MiB
Serialized model int8+zlib: 15269231 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
Total submission size int8+zlib: 15316924 bytes
final_int8_zlib_roundtrip       val_loss:2.2211      val_bpb:1.3155
final_int8_zlib_roundtrip_exact val_loss:2.22109921  val_bpb:1.31546006
```

No NaNs, divergences, or warnings. Early warmup spike at step 2
(`train_loss:16.56`) resolved by step 10 (`train_loss:5.93`), consistent
with normal Muon warmup. Step time stable at ~334.6 ms/step. Run
terminated on the expected wallclock cap, not on instability.

### Metrics

Baseline for delta: `1.10625353` (full-budget reference).

| Metric | Value | Δ vs baseline |
|---|---|---|
| `screen_ema_bpb` | 1.28204035 | **+0.17579** |
| raw `val_bpb` @ step 1614 | 1.31460 | +0.20835 |
| `gate_int6_bpb` (int8+zlib roundtrip) | 1.31546006 | +0.20921 |
| `gate_quant_gap` | 3.99e-05 | ~0 |
| `gate_artifact_mb` (reported) | 0.0 | — |
| int8+zlib payload bytes | 15,269,231 | under 16 MB cap |
| total submission bytes | 15,316,924 | under 16 MB cap |
| peak memory | 10,303 MiB | — |
| steps completed | 1,614 / 20,000 | capped by wallclock |
| `gate_passed` | **true** | — |
| `promote_ema_bpb` | null | not promoted |

**Caveat on the baseline delta.** The `1.10625353` baseline is a
full-budget 8×H100 reference. This arm trained single-GPU
(`world_size:1`) with a 540 s wallclock cap and only completed 1614 of
20000 planned iterations (~8 %). The absolute ~0.18-nat gap is
essentially all schedule truncation — it is *not* a regression signal
against the real 10-minute 8×H100 regime. The load-bearing comparison is
against the other arms of the same tight-budget sweep (`exp001`,
`exp002`, …) under the identical harness.

## Verdict

**neutral**

Mechanically clean: gate passed, quant gap near zero (3.99e-05, effectively
free), artifact comfortably under the 16 MB cap (~15.32 MB), no divergence
or warnings, step time stable. But `promote_ema_bpb` is null, meaning this
arm was not promoted out of the sweep — so at least one sibling `MATRIX_LR`
beat it on `screen_ema_bpb`. In isolation this is neither a win nor a
regression; its value is as one point on the sweep curve, and the verdict
should be revisited once the full `matrix_lr_sweep_at_tight_budge` series
is aggregated.

## Suggested follow-ups

- Aggregate `screen_ema_bpb` across all `idea_matrix_lr_sweep_at_tight_budge`
  arms (`exp001`, `exp002`, `exp003`, …) into a single LR-vs-BPB curve to
  locate the tight-budget optimum and confirm whether 0.06 is monotone or
  an interior point.
- If the sweep optimum lands near 0.06, extend upward (`0.07`, `0.08`) to
  bracket the turnover; otherwise sweep toward the winning neighbor.
- Promote the winning `MATRIX_LR` into a full-budget 8×H100 10-minute run
  on the current SOTA recipe chain to test whether the tight-budget
  ranking transfers to the real regime — tight-budget winners often do
  not.
- Co-sweep `MATRIX_LR` with the warmdown schedule (`WARMDOWN_STEPS` /
  `warmdown3500`). At ~8 % of the nominal step count, the schedule is
  wildly mismatched and may be the dominant factor rather than LR itself.
- Joint sweep `MATRIX_LR × EMBED_LR` and `MATRIX_LR × MUON_MOMENTUM` at
  the winning point — Muon LR and momentum are known to interact.
- Multi-seed (≥3) re-run at the winning `MATRIX_LR` before acting on it,
  per the ≥0.005-nat / p<0.01 record bar, to make sure the signal
  exceeds single-seed noise.
- No action needed on the quantization axis: the 3.99e-05 int8+zlib gap
  is already effectively free.
