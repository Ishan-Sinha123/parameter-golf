# muon_mom=0.93

## Hypothesis

Reducing Muon optimizer momentum from the default 0.95 to 0.93 may improve convergence within the constrained 480s wallclock budget. Lower momentum reduces the effective smoothing window over past gradients, allowing faster adaptation to local curvature — potentially beneficial when the total training steps are capped at ~1400 by the wallclock limit.

## Configuration

| Parameter | Value |
|---|---|
| `MUON_MOMENTUM` | `0.93` (default: `0.95`) |
| Model params | 17,059,912 |
| Attention | GQA (8 heads, 4 KV heads) |
| Tied embeddings | Yes |
| Sequence length | 1024 |
| Batch tokens | 524,288 |
| Grad accum steps | 8 |
| Warmup steps | 20 |
| Muon momentum warmup | 0.85 -> 0.93 over 500 steps |
| Max wallclock | 480 s |
| Seed | 1337 |

Recipe: None (single env-var override on baseline). Sweep context: exp001 of 4-run sweep {0.93, 0.96, 0.97, 0.98}.

## Results

Training stopped early at step 1431/20000 due to wallclock cap (480 s). Step average: ~335.6 ms.

| Metric | Value |
|---|---|
| val_bpb @ step 1000 (fp32) | 1.3672 |
| val_bpb @ step 1431 (fp32) | 1.3289 |
| screen_ema_bpb | 1.2949 |
| gate_int8_zlib_bpb | **1.3301** |
| gate_quant_gap | -0.00005 (negligible) |
| Artifact size (int8+zlib) | 13.52 MB |
| Peak memory | 10,303 MiB |
| Gate passed | Yes |
| Delta vs baseline | Unknown (baseline BPB not provided) |

Key log lines:

```
step:1000/20000 val_loss:2.3084 val_bpb:1.3672
step:1431/20000 val_loss:2.2438 val_bpb:1.3289
stopping_early: wallclock_cap train_time:480184ms step:1431/20000
final_int8_zlib_roundtrip_exact val_loss:2.24589635 val_bpb:1.33014633
Serialized model int8+zlib: 13471458 bytes
```

No warnings or anomalies. Loss decreased steadily throughout training.

## Verdict

**Neutral.** The run completed cleanly and passed the gate, but without a controlled baseline at momentum=0.95 under identical conditions, we cannot determine whether 0.93 is an improvement. The screen_ema_bpb of 1.2949 and gate BPB of 1.3301 are in the expected baseline range. The verdict depends on comparison with sibling sweep experiments (exp002-004 at 0.96, 0.97, 0.98).

## Suggested follow-ups

- Compare all four sweep arms (0.93, 0.96, 0.97, 0.98) against a controlled baseline at default momentum 0.95 to identify the optimum
- If a winner emerges, run 3 seeds at that momentum for statistical significance
- Explore interaction between `MUON_MOMENTUM` and `MUON_MOMENTUM_WARMUP_STEPS` (currently 500) — lower final momentum may benefit from shorter warmup
- Test more extreme values (0.85, 0.90) if lower momentum shows a trend toward improvement
- Stack the best momentum with other known-good techniques (warmdown, EMA, GPTQ-lite) for a combined recipe
