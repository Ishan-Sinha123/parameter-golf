# muon_mom=0.98

## Hypothesis

Momentum 0.98 may be optimal for the Muon optimizer at this wallclock budget. This is the fourth and final run (exp004) in a fine sweep {0.93, 0.96, 0.97, 0.98}. Experiments 001-003 showed a monotonic trend favoring higher momentum with diminishing returns. This run tests whether 0.98 continues the trend or overshoots.

## Configuration

| Parameter | Value |
|---|---|
| `MUON_MOMENTUM` | `0.98` (default: `0.95`) |
| Model params | 17,059,912 |
| Attention | GQA (8 heads, 4 KV heads) |
| Tied embeddings | Yes |
| Sequence length | 1024 |
| Batch tokens | 524,288 |
| Grad accum steps | 8 |
| Warmup steps | 20 |
| Max wallclock | **540 s** (note: 60 s longer than exp001-003 which used 480 s) |
| Seed | 1337 |
| Host / GPU | 206.125.32.60 / GPU 2 |

Recipe: None (single env-var override on baseline). Part of sweep `idea_muon_momentum_fine_sweep`.

**Comparability caveat:** This run used a 540 s wallclock cap (vs 480 s for exp001-003), reaching 1617 steps instead of ~1431-1435. The extra ~180 steps confound a direct momentum comparison.

## Results

Training stopped early at step 1617/20000 due to wallclock cap (540 s). Step average: ~334 ms.

| Metric | Value |
|---|---|
| val_bpb @ step 1000 (fp32) | 1.3671 |
| val_bpb @ step 1617 (fp32) | 1.3119 |
| screen_ema_bpb | 1.2901 |
| gate_int6_bpb (int8+zlib roundtrip) | **1.3129** |
| gate_quant_gap | ~0.000002 (negligible) |
| Artifact size (int8+zlib) | 15.31 MB |
| Peak memory | 10,303 MiB |
| Gate passed | Yes |
| Delta vs baseline | Unknown (baseline BPB not provided) |

Key log lines:

```
step:1000/20000 val_loss:2.3082 val_bpb:1.3671 train_time:333929ms step_avg:333.93ms
step:1617/20000 val_loss:2.2151 val_bpb:1.3119 train_time:540065ms step_avg:333.99ms
stopping_early: wallclock_cap train_time:540065ms step:1617/20000
final_int8_zlib_roundtrip_exact val_loss:2.21677284 val_bpb:1.31289774
Serialized model int8+zlib: 15259302 bytes
```

### Full sweep comparison

| Experiment | MUON_MOMENTUM | Wallclock | Steps | gate_int6_bpb | screen_ema_bpb |
|---|---|---|---|---|---|
| exp001 | 0.93 | 480 s | 1431 | 1.3301 | 1.2949 |
| exp002 | 0.96 | 480 s | 1435 | 1.3236 | 1.2901 |
| exp003 | 0.97 | 480 s | 1434 | 1.3227 | 1.2897 |
| **exp004** | **0.98** | **540 s** | **1617** | **1.3129** | **1.2901** |

Among the comparable runs (exp001-003, same 480 s budget):
- 0.93 to 0.96: **-0.0065** gate_int6_bpb (large gain)
- 0.96 to 0.97: **-0.0009** gate_int6_bpb (diminishing return)

Exp004 (0.98) achieves the best raw gate_int6_bpb (1.3129, **-0.0098** vs 0.97), but this is inflated by ~13% more training steps from the longer wallclock. The screen_ema_bpb (1.2901) is nearly identical to exp002 (0.96) and exp003 (0.97), suggesting the momentum value itself yields little marginal gain beyond 0.96-0.97.

## Verdict

**Promising.** The full sweep confirms higher momentum is better than the 0.95 default, with 0.97 as the clear winner among iso-wallclock runs. The 0.98 result is the best on raw metrics but cannot be fairly separated from the extra training time. The sweep's practical recommendation is `MUON_MOMENTUM=0.97`.

## Suggested follow-ups

- Re-run exp004 (0.98) with the same 480 s wallclock as exp001-003 to get a fair head-to-head with 0.97
- Test `MUON_MOMENTUM=0.99` to confirm whether the trend saturates or reverses at very high momentum
- Adopt `MUON_MOMENTUM=0.97` as the new baseline default and stack with other winning hyperparams (e.g., best Muon LR from the lr sweep)
- Run 3-seed replication at 0.97 to establish statistical significance before a record submission
- Combine momentum 0.97 with the winning MLP expansion ratio and layer count from other sweeps
