# muon_mom=0.96

## Hypothesis

Muon optimizer momentum of 0.96 may be optimal for the constrained wallclock budget. This is part of a fine sweep (0.93, 0.96, 0.97, 0.98) to find the best momentum setting. Higher momentum preserves more gradient history, which can help smooth noisy updates in small-batch regimes, but too much momentum wastes steps tracking stale gradients when total training steps are limited.

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

Training stopped early at step 1435/20000 due to wallclock cap (480 s). Step average: ~334.5 ms.

| Metric | Value |
|---|---|
| Final val_bpb (step 1435, fp16) | 1.3226 |
| screen_ema_bpb | 1.2901 |
| gate_int6_bpb (int8+zlib roundtrip) | 1.3236 |
| gate_quant_gap | ~0.001 |
| Artifact size (int8+zlib) | 13.41 MB |
| Gate passed | Yes |
| Delta vs baseline | Unknown |

### Sibling comparison (partial sweep)

| Experiment | MUON_MOMENTUM | gate_int6_bpb | screen_ema_bpb |
|---|---|---|---|
| exp001 | 0.93 | 1.3301 | 1.2949 |
| **exp002** | **0.96** | **1.3236** | **1.2901** |
| exp003 | 0.97 | pending | pending |
| exp004 | 0.98 | pending | pending |

exp002 (0.96) improves over exp001 (0.93) by **0.0065 int6_bpb** and **0.0048 ema_bpb**.

Key log lines:

```
model_params:17059912
step:1000/20000 val_loss:2.3015 val_bpb:1.3631
step:1435/20000 val_loss:2.2331 val_bpb:1.3226
stopping_early: wallclock_cap train_time:480064ms step:1435/20000
final_int8_zlib_roundtrip val_loss:2.2348 val_bpb:1.3236
Serialized model int8+zlib: 14017080 bytes
```

No warnings or anomalies. Loss curve steady throughout training.

## Verdict

**Promising.** Momentum 0.96 clearly beats 0.93 across both ema and int6 metrics. This is the best result in the sweep so far. Final verdict depends on exp003 (0.97) and exp004 (0.98) completing to confirm 0.96 as the optimum versus a monotonic trend favoring even higher momentum.

## Suggested follow-ups

- Wait for exp003 (0.97) and exp004 (0.98) results to identify the sweep winner.
- If 0.96 wins the sweep, run 3 seeds to establish statistical significance before promoting.
- Test the winning momentum combined with other techniques (warmdown, EMA, GPTQ-lite) to measure stacking behavior.
- If the optimum appears between tested values (e.g., 0.95-0.97 range is flat), try a finer grid or treat momentum as insensitive and move on.
