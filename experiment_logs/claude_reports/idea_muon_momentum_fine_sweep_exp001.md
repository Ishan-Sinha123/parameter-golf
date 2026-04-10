# muon_mom=0.93

## Hypothesis

Reducing Muon optimizer momentum from the default to 0.93 may yield a better loss at the available wallclock budget. Lower momentum allows faster adaptation to local curvature, which can be beneficial when the total number of training steps is limited by the 480-second wallclock cap.

## Configuration

| Parameter | Value |
|---|---|
| `MUON_MOMENTUM` | `0.93` |
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

Training stopped early at step 1431/20000 due to wallclock cap (480 s). Step average: ~335.6 ms.

| Metric | Value |
|---|---|
| Final train_loss (step 1431) | 2.2438 (from val checkpoint) |
| val_bpb (step 1431, fp16) | 1.3289 |
| screen_ema_bpb | 1.2949 |
| gate_int6_bpb (int8+zlib roundtrip) | 1.3301 |
| gate_quant_gap | -0.00005 (negligible) |
| Artifact size (int8+zlib) | 13.52 MB (under 16 MB cap) |
| Gate passed | Yes |
| Delta vs baseline | Unknown (baseline not provided) |

Key log lines:

```
step:1000/20000 val_loss:2.3084 val_bpb:1.3672
step:1431/20000 val_loss:2.2438 val_bpb:1.3289
stopping_early: wallclock_cap train_time:480184ms step:1431/20000
final_int8_zlib_roundtrip val_loss:2.2459 val_bpb:1.3301
Serialized model int8+zlib: 13471458 bytes
```

No warnings or anomalies observed. Loss curve decreased steadily through training.

## Verdict

**Neutral.** The run completed cleanly and passed the gate, but without a known baseline for comparison (default Muon momentum), we cannot determine whether 0.93 is an improvement. The screen_ema_bpb of 1.2949 and gate_int6_bpb of 1.3301 are reasonable but need comparison against sibling experiments in the sweep (exp002-004) to draw conclusions.

## Suggested follow-ups

- Compare against the other momentum values in the sweep (exp002-004) to identify the optimal momentum setting.
- If 0.93 wins the sweep, run 3 seeds at that value for statistical significance.
- Combine the best momentum with other known-good techniques (e.g., warmdown, EMA, GPTQ-lite) and measure stacking behavior.
- Try momentum scheduling (e.g., cosine decay from 0.95 to 0.85) if fixed-momentum results are inconclusive.
