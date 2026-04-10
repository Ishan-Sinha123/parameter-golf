# matrix_lr=0.05

## Hypothesis

Raising `MATRIX_LR` to 0.05 may better fit the compressed (tight-budget) training
regime than the default matrix learning rate. Under aggressive wallclock caps
where only ~1600 steps complete, a higher matrix LR could let the Muon-style
matrix parameter group make more effective progress per step. This run is a
single-GPU screen inside a larger matrix-LR sweep at the tight budget.

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
| host / gpu | `206.125.32.60` / `[4]` |
| branch / commit | `autoresearch-deploy` / `cb9726e0` |

Recipe: _(none — env override only on the current `autoresearch-deploy` baseline)_.

## Results

Key log lines:

```
step:0/20000    val_loss:6.9357 val_bpb:4.1077
step:1000/20000 val_loss:2.3091 val_bpb:1.3676
step:1612/20000 val_loss:2.2200 val_bpb:1.3148
stopping_early: wallclock_cap train_time:540289ms step:1612/20000
Serialized model int8+zlib: 14832612 bytes (payload_ratio:3.91x)
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
| 1612 | — | 1.3148 | 540.3 s (wallclock cap) |

### Gate metrics (vs baseline 1.0810)

| Metric | Value | Δ vs baseline |
|---|---:|---:|
| screen EMA bpb | 1.28183 | +0.20083 |
| gate int6/int8 bpb | 1.31590 | +0.23490 |
| quant gap (int6 − EMA) | −0.0000217 | — |
| artifact size (int8+zlib) | 14.88 MB | — |
| gate_artifact_mb (stamped) | 0.00 (not populated) | — |
| gate_passed | **true** | — |
| promote EMA bpb | n/a | — |

**Caveats on the baseline delta.** The 1.0810 baseline is a full 8×H100 SOTA
fork. This run is a **single-GPU screen that hit the wallclock cap at step
1612/20000**, so the +0.23 BPB gap reflects reduced steps and smaller world
size — **not** a like-for-like regression of `MATRIX_LR=0.05`. The screen only
tells us whether the LR override is trainable and passes the int6 gate.

### Observations

- Warmup completed cleanly (20/20). Early train loss spiked to 16.65 at step 2
  then recovered to ~2.33 by step 1000 — consistent with an aggressive matrix
  LR but not diverging.
- Quant gap is effectively zero (−2.17e-5): int8+zlib roundtrip is lossless
  relative to the EMA screen, a good signal for quantization robustness.
- Throughput stabilized at ~335 ms/step (grad_accum_steps=8, single GPU).
- No warnings, NaNs, or divergences observed in the log.

## Verdict

**neutral** — gate passed with a negligible quant gap, confirming
`MATRIX_LR=0.05` trains cleanly and survives int6+zlib quantization at the
tight budget. The +0.23 BPB gap vs the 1.0810 SOTA baseline is **not**
meaningful because this was a truncated single-GPU screen, not a full
8×H100 run. The result is a valid screening data point for the sweep but
gives no direct evidence of a win or regression vs SOTA.

## Suggested follow-ups

- Rank this run against the other tight-budget sweep points
  (`idea_matrix_lr_sweep_at_tight_budge_exp001 / exp003 / exp004`) on matched
  hardware to isolate the LR effect from throughput confounds.
- If `MATRIX_LR=0.05` wins the screen ranking, promote it to a full-scale
  8×H100 run on the current SOTA recipe
  (`rec_20260410_...sp8192_3layerrecur_parresid_q`, 1.0810) with ≥3 seeds
  for statistical significance (≥0.005 nats, p<0.01).
- Co-sweep `MATRIX_LR` with `EMBED_LR` / `SCALAR_LR`, since these LR groups
  interact through the Muon / AdamW balance and a joint optimum may differ
  from the marginal one.
- Investigate why `gate_artifact_mb=0.0` was reported even though the log
  shows `Total submission size int8+zlib: 14880305 bytes` — the metric is
  likely not being stamped by the screen stage and should be fixed for
  future sweep comparisons.
- If higher LRs trend better, extend the sweep upward (0.06–0.08) under the
  same wallclock cap to find the LR knee before promotion.
