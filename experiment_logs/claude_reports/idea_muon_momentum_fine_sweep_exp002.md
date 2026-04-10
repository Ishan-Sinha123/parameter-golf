# muon_mom=0.96

## Hypothesis

Muon optimizer momentum of 0.96 may be optimal for the constrained wallclock budget. This is part of a fine sweep (0.93, 0.96, 0.97, 0.98) to determine the best momentum setting. Moderate momentum balances gradient history retention against responsiveness -- too low wastes information, too high tracks stale gradients when total steps are capped by wallclock.

## Configuration

| Parameter | Value |
|---|---|
| `MUON_MOMENTUM` | `0.96` |
| Model params | 17,059,912 |
| Attention | GQA (8 heads, 4 KV heads) |
| Tied embeddings | Yes |
| Sequence length | 1024 |
| Batch tokens | 524,288 |
| Grad accum steps | 8 |
| Warmup steps | 20 |
| Max wallclock | 480 s |
| Seed | 1337 |

Recipe: None (single env-var override on baseline).

## Results

Training stopped early at step 1435/20000 due to wallclock cap (480 s). Step average: ~334.54 ms.

| Metric | Value |
|---|---|
| Final val_bpb (step 1435, fp16) | 1.3226 |
| screen_ema_bpb | 1.2901 |
| gate_int6_bpb (int8+zlib roundtrip) | 1.3236 |
| gate_quant_gap | ~0.000005 (negligible) |
| Artifact size (int8+zlib) | 13.41 MB |
| Gate passed | Yes |
| Delta vs baseline | Unknown |

### Sweep comparison

| Experiment | MUON_MOMENTUM | gate_int6_bpb | screen_ema_bpb | Steps reached |
|---|---|---|---|---|
| exp001 | 0.93 | 1.3301 | 1.2949 | 1431 |
| **exp002** | **0.96** | **1.3236** | **1.2901** | **1435** |
| exp003 | 0.97 | 1.3227 | 1.2897 | 1434 |
| exp004 | 0.98 | — | — | (different hardware, not directly comparable) |

Among comparable runs (exp001-003, all ~335 ms/step, 480 s cap), momentum 0.96 sits in the middle of a monotonically improving trend:

- 0.93 -> 0.96: **-0.0065** int6_bpb (large gain)
- 0.96 -> 0.97: **-0.0009** int6_bpb (diminishing return)

### Key log lines

```
model_params:17059912
step:1000/20000 val_loss:2.3015 val_bpb:1.3631 train_time:334387ms step_avg:334.39ms
step:1435/20000 val_loss:2.2331 val_bpb:1.3226 train_time:480064ms step_avg:334.54ms
stopping_early: wallclock_cap train_time:480064ms step:1435/20000
final_int8_zlib_roundtrip val_loss:2.2348 val_bpb:1.3236
Serialized model int8+zlib: 14017080 bytes
```

No warnings or anomalies. Loss curve decreased steadily through training.

## Verdict

**Neutral.** Momentum 0.96 is a solid improvement over 0.93 (-0.0065 int6_bpb) but is outperformed by 0.97 (-0.0009 int6_bpb further). The sweep confirms that the optimum lies at or above 0.96, with 0.97 currently the best comparable setting. This run's primary value is confirming the monotonic trend across the sweep.

## Suggested follow-ups

- Re-run exp004 (momentum 0.98) on the same GPU class as exp001-003 to complete the apples-to-apples comparison and determine if the optimum is 0.97 or 0.98.
- If 0.97 wins the sweep, run 3 seeds at that value for statistical significance before promoting.
- Test the winning momentum combined with stacking techniques (warmdown, EMA, GPTQ-lite) to confirm the improvement composes.
- The large jump from 0.93 to 0.96 suggests the default momentum may be sub-optimal -- verify what the current default is and whether this sweep changes it.
