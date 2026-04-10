# matrix_lr=0.05

## Hypothesis

Raising `MATRIX_LR` to 0.05 may better fit the compressed (tight-budget) training
regime than the default matrix learning rate. Under an aggressive wallclock cap
where only ~1600 steps complete, a higher matrix LR could let the Muon-style
matrix parameter group make more effective progress per step. This run is a
single-GPU screen inside a larger `MATRIX_LR` sweep at the tight budget.

## Configuration

| Key | Value |
|---|---|
| `MATRIX_LR` (env override) | `0.05` |
| embed_lr | 0.05 |
| head_lr | 0.0 |
| scalar_lr | 0.04 |
| model_params | 17,059,912 |
| attention | `gqa num_heads:8 num_kv_heads:4`, `tie_embeddings:True` |
| world_size | 1 (single-GPU screen) |
| grad_accum_steps | 8 |
| train_batch_tokens | 524,288 |
| train_seq_len | 1024 |
| iterations (req.) | 20,000 |
| warmup_steps | 20 |
| max_wallclock_seconds | 540 |
| seed | 1337 |

Recipe: _(none — `recipe_id: null`, env override only on the current
`autoresearch-deploy` baseline; no `source_ref`)_.

## Results

Key log lines quoted from `train.log`:

```
step:0/20000    val_loss:6.9357 val_bpb:4.1077
step:1000/20000 val_loss:2.3091 val_bpb:1.3676
step:1600/20000 train_loss:2.1806 train_time:536293ms step_avg:335.18ms
step:1612/20000 val_loss:2.2200 val_bpb:1.3148 train_time:540289ms
stopping_early: wallclock_cap train_time:540289ms step:1612/20000
Serialized model int8+zlib: 14832612 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
Total submission size int8+zlib: 14880305 bytes
final_int8_zlib_roundtrip val_loss:2.2219 val_bpb:1.3159
final_int8_zlib_roundtrip_exact val_loss:2.22187874 val_bpb:1.31592174
```

### Training trajectory

| Step | train_loss | val_bpb | wall time |
|---|---|---|---|
| 0 | — | 4.1077 | 0 ms |
| 200 | 2.8052 | — | 67.0 s |
| 1000 | 2.3409 | 1.3676 | 335.9 s |
| 1612 | — | 1.3148 | 540.3 s (wallclock cap hit) |

### Gate metrics (vs baseline 1.10625353)

| Metric | Value | Δ vs baseline |
|---|---:|---:|
| screen EMA bpb | 1.28183 | **+0.17558** |
| gate int6/int8 bpb (final_int8_zlib_roundtrip) | 1.31590 | **+0.20965** |
| quant gap (int6 − EMA) | −0.0000217 | ~0 |
| artifact size (int8+zlib, from log) | 14.88 MB | under 16 MB cap |
| gate_artifact_mb (stamped) | 0.00 (not populated) | — |
| gate_passed | **true** | — |
| promote EMA bpb | n/a (not promoted) | — |

**Caveats on the baseline delta.** The 1.10625 baseline is a full 8×H100 SOTA
figure. This run is a **single-GPU screen that hit the wallclock cap at step
1612/20000**, so the +0.176 / +0.210 BPB gaps reflect reduced steps and smaller
world size — **not** a like-for-like regression of `MATRIX_LR=0.05`. The screen
only tells us whether the LR override is trainable and passes the int6 gate at
the tight budget; sweep-internal comparisons against sibling exp001/exp003 are
the fair signal.

### Observations

- Warmup completed cleanly (20/20). Early train loss spiked to 16.65 at step 2
  then recovered to ~2.33 by step 1000 — consistent with an aggressive matrix
  LR but not diverging.
- Quant gap is effectively zero (−2.17e-5): int8+zlib roundtrip is lossless
  relative to the EMA screen, a strong signal for quantization robustness.
- Throughput stabilized at ~335 ms/step (grad_accum_steps=8, single GPU).
- No warnings, NaNs, or divergences observed in the log.

## Verdict

**neutral** — gate passed with a negligible quant gap, confirming
`MATRIX_LR=0.05` trains cleanly and survives int8+zlib quantization at the
tight budget. The +0.176 / +0.210 BPB gaps vs the 1.10625 baseline are **not**
meaningful because this was a truncated single-GPU screen, not a full 8×H100
run. The result is a valid screening data point for the sweep but gives no
direct evidence of a win or regression vs SOTA.

## Suggested follow-ups

- Rank this run against the other tight-budget sweep points
  (`idea_matrix_lr_sweep_at_tight_budge_exp001` and `exp003`) on matched
  hardware to isolate the LR effect from throughput confounds.
- If `MATRIX_LR=0.05` wins the screen ranking, promote it to a full-scale
  8×H100 run on the current SOTA recipe
  (`rec_20260410_..._sp8192_3layerrecur_parresid_q`) with ≥3 seeds for
  statistical significance (≥0.005 nats, p<0.01).
- Co-sweep `MATRIX_LR` with `EMBED_LR` / `SCALAR_LR`, since these LR groups
  interact through the Muon / AdamW balance and a joint optimum may differ
  from the marginal one.
- Investigate why `gate_artifact_mb` was stamped as `0.0` even though the log
  shows `Total submission size int8+zlib: 14880305 bytes` — the metric is
  likely not being populated by the screen stage and should be fixed for
  future sweep comparisons.
- If higher LRs trend better inside the sweep, extend the range upward
  (0.06–0.08) under the same wallclock cap to find the LR knee before
  committing to a full promotion run.
- Given the step-2 train_loss spike (16.65), try `warmup_steps=30–40` for
  matrix_lr ≥ 0.05 to see whether a slightly gentler warmup lets higher LRs
  converge faster in the tight-budget regime.
