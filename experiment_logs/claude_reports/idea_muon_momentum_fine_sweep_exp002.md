# muon_mom=0.96

## Hypothesis

Muon momentum of **0.96** may be the optimal setting at this wallclock
budget. This is one point in a fine sweep above the default 0.95, probing
whether a small momentum bump improves convergence on a wallclock-capped
screening run before committing to a full 8×H100 promotion.

## Configuration

| Env Var          | Value |
|------------------|-------|
| `MUON_MOMENTUM`  | 0.96  |

- **Recipe id:** `null` (single env-var override on the default screening
  baseline — **not** stacked on the current SOTA chain)
- **Source ref:** _(none)_
- **Model params:** 17,059,912
- **Attention:** GQA, 8 heads / 4 KV heads, tied embeddings
- **Seq len / batch tokens:** 1024 / 524,288, grad_accum 8, `world_size:1`
- **LRs:** embed 0.05, matrix 0.04, scalar 0.04, head 0.0
- **Iterations / warmup / wallclock cap:** 20,000 / 20 / 480 s
- **Seed:** 1337

## Results

Training stopped early at step **1435 / 20000** via the 480 s wallclock
cap (~334.5 ms/step, no warnings in the log).

Key log lines:

```
model_params:17059912
step:1000/20000 val_loss:2.3015 val_bpb:1.3631 train_time:334387ms
step:1400/20000 train_loss:2.2823 train_time:468408ms step_avg:334.58ms
step:1435/20000 val_loss:2.2331 val_bpb:1.3226 train_time:480064ms
stopping_early: wallclock_cap train_time:480064ms step:1435/20000
peak memory allocated: 10303 MiB reserved: 10622 MiB
Serialized model int8+zlib: 14017080 bytes (payload_ratio:3.91x)
Total submission size int8+zlib: 14064773 bytes
final_int8_zlib_roundtrip val_loss:2.2348 val_bpb:1.3236 eval_time:10837ms
final_int8_zlib_roundtrip_exact val_loss:2.23483479 val_bpb:1.32359505
```

| Metric                       | Value       | Δ vs baseline (1.10625) |
|------------------------------|-------------|--------------------------|
| screen_ema_bpb               | 1.29013     | **+0.18388**             |
| gate_int6_bpb                | 1.32360     | +0.21735                 |
| gate_quant_gap               | 4.95e-06    | — (negligible)           |
| gate_artifact_mb (int8+zlib) | ~14.02      | under 16 MB cap          |
| gate_passed                  | ✅ true      | —                        |
| promote_ema_bpb              | null        | not promoted             |
| promote_int6_bpb             | null        | not promoted             |

**Important caveat:** the 1.10625 baseline is the current SOTA-stacked
recipe. This screen runs the **default** baseline with only
`MUON_MOMENTUM` changed, so the large absolute gap reflects the
SOTA-vs-default structural delta, not the effect of momentum itself. The
momentum sweep is only meaningfully ranked across its own cohort
(`exp001`–`exp004`).

## Verdict

**neutral.** 0.96 passes the screening gate with a near-zero quant gap
(~5e-6) and the int8+zlib artifact comfortably under the 16 MB cap; the
loss descent in the log is clean and monotonic. But the run is +0.184
nats above the SOTA baseline because it is not stacked on the SOTA
recipe, and ranking momentum values requires the sibling sweep points —
this single result does not support a win / regression call on its own.

## Suggested follow-ups

- Aggregate `screen_ema_bpb` across `exp001`–`exp004` of
  `idea_muon_momentum_fine_sweep` to locate the minimum on the momentum
  curve and confirm whether 0.96 beats 0.95 / 0.97 / 0.98.
- Whichever value wins the sweep, re-run it **stacked on the current
  SOTA recipe** rather than the default screening baseline, since
  momentum optima often do not transfer across recipes.
- Confirm the sweep winner with ≥3 seeds before treating it as signal —
  at this model size seed noise is comparable to the competition's
  0.005-nat / p<0.01 record bar.
- Probe interaction with `MUON_WD`, Parallel Muon, and `warmdown3500`:
  recent SOTA chains changed the effective update scale, which can shift
  the momentum optimum.
- If the sweep winner is not the current `train_gpt.py` default, consider
  bumping the default so every downstream experiment inherits the win.
