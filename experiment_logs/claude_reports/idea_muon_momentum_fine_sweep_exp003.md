# muon_mom=0.97

## Hypothesis

Muon optimizer momentum of 0.97 may be optimal for the constrained
wallclock budget. This experiment is part of a fine sweep
(0.93, 0.96, 0.97, 0.98) exploring the tradeoff between gradient
smoothing (higher momentum) and responsiveness to local curvature
(lower momentum) when total training steps are capped by a 480-second
wallclock limit.

## Configuration

| Parameter | Value |
|---|---|
| `MUON_MOMENTUM` | `0.97` |
| Model params | 17,059,912 |
| Attention | GQA (8 heads, 4 KV heads) |
| Tied embeddings | Yes |
| embed_lr / matrix_lr / scalar_lr | 0.05 / 0.04 / 0.04 |
| Sequence length | 1024 |
| Batch tokens | 524,288 |
| Grad accum steps | 8 |
| Warmup steps | 20 |
| Max wallclock | 480 s |
| Seed | 1337 |

Recipe: None (single env-var override on the sweep screening baseline).
Stage: `gate`. Commit `1bbd7549df736d3b15bf95a346d6b4c848ff0be3` on
`autoresearch-deploy`.

## Results

Training stopped early at step 1434 / 20000 due to the wallclock cap
(480 s). Step average: ~334.80 ms.

| Metric | Value | Δ vs baseline (1.0810) |
|---|---|---|
| Final val_bpb (step 1434, fp16) | 1.3218 | +0.2408 |
| screen_ema_bpb | 1.28966815 | +0.20867 |
| gate_int6_bpb (int8+zlib roundtrip) | 1.32270 | +0.24170 |
| gate_quant_gap | 3.35e-06 | ~0 (lossless) |
| Artifact (int8+zlib) | 14,307,775 B (13.65 MB) | under 16 MB cap |
| Gate passed | ✅ true | — |
| Promoted | ❌ `promote_ema_bpb = null` | — |

Note: the 1.0810 baseline is the full-recipe SOTA
(`rec_20260410_..._sp8192_3layerrecur_parresid`, 10-min budget, sp8192
tokenizer). This sweep runs a compact 17 M-param, sp1024, 480 s
screening harness — so the +0.2 nats gap is expected and should only be
read relative to the sibling momentum points below, not the SOTA.

### Sweep comparison (comparable runs, same harness and wallclock)

| Experiment | MUON_MOMENTUM | gate_int6_bpb | screen_ema_bpb | Steps |
|---|---|---|---|---|
| exp001 | 0.93 | 1.3301 | 1.2949 | 1431 |
| exp002 | 0.96 | 1.3236 | 1.2901 | 1435 |
| **exp003** | **0.97** | **1.3227** | **1.28967** | **1434** |
| exp004 | 0.98 | pending | pending | pending |

Monotonic improvement with higher momentum; gains are diminishing:

- 0.93 → 0.96: **−0.0065** int6_bpb
- 0.96 → 0.97: **−0.0009** int6_bpb

### Key log lines

```
model_params:17059912
attention_mode:gqa num_heads:8 num_kv_heads:4
train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:20 max_wallclock_seconds:480.000
step:1000/20000 val_loss:2.3013 val_bpb:1.3629 train_time:334849ms
step:1434/20000 val_loss:2.2317 val_bpb:1.3218 train_time:480102ms
stopping_early: wallclock_cap train_time:480102ms step:1434/20000
Serialized model int8+zlib: 14307775 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
final_int8_zlib_roundtrip val_loss:2.2333 val_bpb:1.3227 eval_time:10853ms
final_int8_zlib_roundtrip_exact val_loss:2.23331788 val_bpb:1.32269665
```

No warnings or anomalies; loss decreased steadily after the usual
early-warmup spike (step 2 train_loss=16.74 is normal for cold Muon
warmup).

## Verdict

**promising.** Momentum 0.97 is the sweep leader so far, beating 0.96
on both gate_int6_bpb (1.3227 vs 1.3236) and screen_ema_bpb
(1.28967 vs 1.2901). The monotonic improvement from 0.93 → 0.97 is
credible, and the int8+zlib quant gap is effectively zero
(3.35e-06 nats). However, the marginal gain is shrinking fast
(0.0065 → 0.0009), so we are likely near the optimum, and exp003 was
gate-passed but not promoted. Final conclusion needs exp004 (0.98) to
confirm whether the trend continues or rolls over.

## Suggested follow-ups

- Wait for exp004 (0.98) to finish and identify the rollover point;
  if 0.98 is worse, exp003 wins the sweep.
- If 0.97 or 0.98 wins, run 3 seeds at that value for significance
  before promoting (record bar is 0.005 nats at p < 0.01).
- Because the marginal gain from 0.96 → 0.97 is only 0.0009 int6_bpb,
  weigh the cost of further momentum fine-tuning vs moving on to
  higher-impact levers (e.g. warmdown length, QK5, GPTQ calibration).
- Test the winning momentum stacked on the current SOTA recipe
  (`sp8192_3layerrecur_parresid`) at the full 10-minute budget — the
  screening-harness optimum may not transfer cleanly to the
  sp8192 + 3-layer-recurrence regime.
- If 0.98 also improves, extend the sweep to 0.985 / 0.99 to find the
  true optimum before it destabilizes early training.
- Consider pairing the winning momentum with Parallel-Muon or Muon-WD
  to see whether the effect composes with other optimizer variants
  already on the record track.
