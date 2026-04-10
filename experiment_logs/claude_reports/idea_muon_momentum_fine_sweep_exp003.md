# muon_mom=0.97

## Hypothesis

Muon optimizer momentum of 0.97 may be optimal for the constrained wallclock budget. This is part of a fine sweep (0.93, 0.96, 0.97, 0.98) to determine the best momentum setting. Higher momentum retains more gradient history, smoothing noisy updates in limited-step regimes, but excessive momentum can waste steps tracking stale gradients when total training steps are capped by wallclock.

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
| gate_quant_gap | ~0.0009 |
| Artifact size (int8+zlib) | 13.69 MB |
| Gate passed | Yes |
| Delta vs baseline | Unknown |

### Sweep comparison

| Experiment | MUON_MOMENTUM | gate_int6_bpb | screen_ema_bpb | Steps reached |
|---|---|---|---|---|
| exp001 | 0.93 | 1.3301 | 1.2949 | 1431 |
| exp002 | 0.96 | 1.3236 | 1.2901 | 1435 |
| **exp003** | **0.97** | **1.3227** | **1.2897** | **1434** |
| exp004 | 0.98 | 1.2901* | — | 2178* |

\* exp004 ran with ~165 ms/step (vs ~335 ms for others) and a 360 s wallclock cap, reaching ~50% more steps. Not directly comparable on the same hardware footing.

Among comparable runs (exp001-003, all ~335 ms/step, 480 s cap), the trend is monotonically improving with higher momentum:

- 0.93 -> 0.96: **-0.0065** int6_bpb
- 0.96 -> 0.97: **-0.0009** int6_bpb

The improvement from 0.96 to 0.97 is much smaller, suggesting diminishing returns as we approach the optimum.

### Key log lines

```
model_params:17059912
step:1000/20000 val_loss:2.3013 val_bpb:1.3629
step:1434/20000 val_loss:2.2317 val_bpb:1.3218
stopping_early: wallclock_cap train_time:480102ms step:1434/20000
final_int8_zlib_roundtrip val_loss:2.2333 val_bpb:1.3227
Serialized model int8+zlib: 14307775 bytes
```

No warnings or anomalies. Loss curve steady throughout training.

## Verdict

**Promising.** Momentum 0.97 is the best comparable result in the sweep, beating 0.96 by 0.0009 int6_bpb and 0.0004 ema_bpb. The monotonic improvement across 0.93 -> 0.96 -> 0.97 is clear but flattening, suggesting we are near the optimum. The exp004 (0.98) result is confounded by different hardware, so we cannot confirm whether 0.97 or 0.98 is better under identical conditions.

## Suggested follow-ups

- Re-run exp004 (momentum 0.98) on the same hardware as exp001-003 to complete the apples-to-apples comparison.
- If 0.97 or 0.98 wins, run 3 seeds to establish statistical significance before promoting.
- The improvement from 0.96 to 0.97 is small (0.0009). Consider whether the marginal gain justifies further tuning or whether momentum is effectively solved at 0.96-0.97.
- Test the winning momentum combined with other stacking techniques (warmdown, EMA, GPTQ-lite) to confirm the improvement composes.
- Try momentum 0.95 to confirm the valley shape and rule out non-monotonic behavior.
