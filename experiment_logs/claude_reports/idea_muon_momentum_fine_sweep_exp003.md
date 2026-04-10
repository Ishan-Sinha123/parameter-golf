# muon_mom=0.97

## Hypothesis

Muon optimizer momentum of 0.97 may be optimal for the constrained wallclock budget. This is part of a fine sweep (0.93, 0.96, 0.97, 0.98) exploring the tradeoff between gradient smoothing (higher momentum) and responsiveness to local curvature (lower momentum) when total training steps are capped by a 480-second wallclock limit.

## Configuration

| Parameter | Value |
|---|---|
| `MUON_MOMENTUM` | `0.97` |
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

Training stopped early at step 1434/20000 due to wallclock cap (480 s). Step average: ~334.80 ms.

| Metric | Value |
|---|---|
| Final val_bpb (step 1434, fp16) | 1.3218 |
| screen_ema_bpb | 1.2897 |
| gate_int6_bpb (int8+zlib roundtrip) | 1.3227 |
| gate_quant_gap | ~0.001 |
| Artifact size (int8+zlib) | 13.65 MB |
| Gate passed | Yes |
| Delta vs baseline | Unknown |

### Sweep comparison (comparable runs, same hardware/wallclock)

| Experiment | MUON_MOMENTUM | gate_int6_bpb | screen_ema_bpb | Steps |
|---|---|---|---|---|
| exp001 | 0.93 | 1.3301 | 1.2949 | 1431 |
| exp002 | 0.96 | 1.3236 | 1.2901 | 1435 |
| **exp003** | **0.97** | **1.3227** | **1.2897** | **1434** |
| exp004 | 0.98 | pending | pending | pending |

Monotonic improvement with higher momentum, but gains are diminishing:

- 0.93 -> 0.96: **-0.0065** int6_bpb
- 0.96 -> 0.97: **-0.0009** int6_bpb

### Key log lines

```
model_params:17059912
step:1000/20000 val_loss:2.3013 val_bpb:1.3629
step:1434/20000 val_loss:2.2317 val_bpb:1.3218
stopping_early: wallclock_cap train_time:480102ms step:1434/20000
final_int8_zlib_roundtrip val_loss:2.2333 val_bpb:1.3227
Serialized model int8+zlib: 14307775 bytes
```

No warnings or anomalies. Loss curve decreased steadily throughout training.

## Verdict

**Promising.** Momentum 0.97 is the sweep leader so far, beating 0.96 on both int6_bpb (1.3227 vs 1.3236) and ema_bpb (1.2897 vs 1.2901). The consistent monotonic improvement from 0.93 to 0.97 is credible but the shrinking marginal gain (0.0065 -> 0.0009) suggests we are near the optimum. Final conclusion requires exp004 (0.98) to determine whether the trend continues or reverses.

## Suggested follow-ups

- Wait for exp004 (0.98) to complete the sweep and identify the rollover point.
- If 0.97 or 0.98 wins, run 3 seeds at that value for statistical significance before promoting.
- The improvement from 0.96 to 0.97 is small (0.0009 int6_bpb). Consider whether further momentum tuning is worthwhile vs moving on to higher-impact techniques.
- Test the winning momentum combined with stacking techniques (warmdown, EMA, GPTQ-lite, XSA) to confirm it composes.
- If 0.98 also improves, extend sweep to 0.99 to find the true optimum.
