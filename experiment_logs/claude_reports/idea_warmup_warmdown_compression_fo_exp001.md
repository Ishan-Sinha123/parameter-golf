# tight-schedule

## Hypothesis

Reducing warmup to just 10 steps and warmdown to 600 iterations compresses the learning-rate schedule so the model spends more of its training budget at peak LR, yielding faster loss descent and a better final BPB within the wallclock cap.

## Configuration

| Parameter | Value |
|---|---|
| `WARMUP_STEPS` | `10` |
| `WARMDOWN_ITERS` | `600` |
| Model params | 17,059,912 |
| Sequence length | 1024 |
| Batch tokens | 524,288 |
| Grad accum steps | 8 |
| World size | 1 (single GPU) |
| Wallclock cap | 480 s |
| Seed | 1337 |
| Recipe | None (default baseline) |

## Results

| Metric | Value | Delta vs baseline (1.2244) |
|---|---|---|
| EMA val_bpb (screen) | **1.2894** | +0.0650 (worse) |
| int8+zlib val_bpb (gate) | **1.3190** | +0.0946 (worse) |
| Quant gap | ~0.0000 | — |
| Artifact size (int8+zlib) | 14.75 MB | under 16 MB cap |
| Steps completed | 1,439 / 20,000 | — |
| Total train time | 480 s (wallclock cap hit) | — |
| Step avg | 333.69 ms | — |

### Key log lines

```
warmup_step:10/10                                           # warmup done in ~3.3 s
step:0/20000  val_bpb:4.1077                                # initial val
step:2/20000  train_loss:16.7413                            # spike after ultra-short warmup
step:1000/20000 val_bpb:1.3726
step:1439/20000 val_bpb:1.3179                              # final val before wallclock stop
stopping_early: wallclock_cap train_time:480184ms step:1439
final_int8_zlib_roundtrip val_bpb:1.31899940
```

### Observations

1. **Loss spike at step 2** — train_loss jumped to 16.74, suggesting the 10-step warmup is too aggressive; the optimizer overshoots before settling.
2. **Early stop at step 1,439** — the run used only 1 GPU (`world_size:1`) and hit the 480 s wallclock cap far short of the 20,000-step budget. With 8 GPUs the step count would be ~8x higher, making this comparison partly confounded by GPU count.
3. **Quant gap is negligible** — int8+zlib roundtrip added essentially zero BPB, meaning model weights quantize cleanly.
4. **Loss was still decreasing** — val_bpb dropped from 1.3726 (step 1000) to 1.3179 (step 1439), so the model was not yet converged.

## Verdict

**regression** — At 1.2894 EMA BPB, this is +0.065 worse than the 1.2244 baseline. The ultra-short warmup caused an early loss spike, and the single-GPU run completed only ~7% of the planned steps before hitting the wallclock cap. The tight schedule idea is not validated under these conditions; the result is confounded by the 1-GPU limitation.

## Suggested follow-ups

- **Re-run on 8 GPUs** to get a fair comparison: the single-GPU wallclock-limited run completed too few steps to properly evaluate the schedule compression idea.
- **Moderate warmup (50–100 steps)** instead of 10: the step-2 loss spike suggests 10 is too few to stabilize gradients. Try `WARMUP_STEPS=50` or `WARMUP_STEPS=100` with `WARMDOWN_ITERS=600`.
- **Sweep warmdown more aggressively**: try `WARMDOWN_ITERS=400` and `WARMDOWN_ITERS=800` to find the sweet spot between peak-LR time and final annealing quality.
- **Combine with Muon optimizer**: the tight schedule may interact differently with Muon's momentum; test jointly.
