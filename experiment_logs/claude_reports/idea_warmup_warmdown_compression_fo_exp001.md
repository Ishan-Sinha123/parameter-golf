# tight-schedule

## Hypothesis

Spend more of the 10-minute budget at peak learning rate by compressing the
warmup and warmdown phases. The theory: a near-instant warmup (10 steps)
plus a late warmdown (600 iters) should leave more wallclock at effective
peak LR and therefore accelerate loss descent within the cap.

## Configuration

| Env override | Value |
| --- | --- |
| `WARMUP_STEPS` | `10` |
| `WARMDOWN_ITERS` | `600` |

Recipe: **none** (ran against current default `train_gpt.py`, no forked recipe).
Reproduction: no. Source branch: (not recorded).

Run context from `train.log`:

- `model_params:17059912`
- `world_size:1 grad_accum_steps:8` (single GPU, not 8×H100)
- `attention_mode:gqa num_heads:8 num_kv_heads:4`
- `tie_embeddings:True embed_lr:0.05 head_lr:0.0 matrix_lr:0.04 scalar_lr:0.04`
- `train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:10 max_wallclock_seconds:480.000`
- `seed:1337`

## Results

| Metric | Value | Δ vs baseline (1.081) |
| --- | --- | --- |
| Screen EMA val_bpb | **1.28935** | **+0.20835** |
| Gate int6 / int8+zlib val_bpb | 1.31900 | +0.23800 |
| Quant gap (int6 − ema) | ~6e-7 | — |
| Artifact size (int8+zlib) | 14,748,471 B | under 16 MB cap |
| Steps completed | 1,439 / 20,000 | — |
| Total train time | 480.2 s (wallclock cap hit) | — |
| Step avg | ~333.7 ms | — |
| Gate passed | ✅ true | — |
| Promoted | ❌ no | — |

### Key lines from `train.log`

```
warmup_step:10/10
step:0/20000 val_loss:6.9357 val_bpb:4.1077 train_time:0ms
step:2/20000 train_loss:16.7413 train_time:658ms    # spike from ultra-short warmup
step:3/20000 train_loss:8.7525
step:4/20000 train_loss:6.5885
step:1000/20000 val_loss:2.3175 val_bpb:1.3726
step:1439/20000 val_loss:2.2251 val_bpb:1.3179
stopping_early: wallclock_cap train_time:480184ms step:1439/20000
Serialized model int8+zlib: 14700778 bytes (payload_ratio:3.91x)
final_int8_zlib_roundtrip val_loss:2.22707524 val_bpb:1.31899940
```

### Observations

1. **Step-2 loss spike (16.74)** — the 10-step warmup is below the
   stability floor for this optimizer config; training recovers by step 4
   but the spike costs early progress.
2. **Early stop at step 1,439 / 20,000** — the run used a single GPU
   (`world_size:1`) and hit the 480 s cap at only ~7 % of the planned
   schedule. `WARMDOWN_ITERS=600` is defined against a 20 k-step schedule
   the run never came close to completing, so the "warmdown" phase was
   never truly exercised at the intended relative position.
3. **Quant gap negligible (~6e-7)** — int8+zlib roundtrip added
   essentially zero BPB, weights quantize cleanly.
4. **Loss still descending at stop** — val_bpb 1.3726 → 1.3179 between
   steps 1000 and 1439, i.e., nowhere near converged.
5. **Confound vs baseline.** Baseline 1.081 was produced on 8×H100; this
   run was single-GPU. The +0.208 gap conflates the schedule change with
   a ~8× step-count deficit, so the experiment cannot cleanly falsify
   the hypothesis on its own.

## Verdict

**regression** — screen EMA 1.28935 is +0.208 BPB worse than the current
SOTA baseline (1.081). The compressed warmup triggered an early loss
spike, the single-GPU run hit the wallclock cap at step 1439, and the
gate result (1.319) is far above any promote threshold. The gate only
"passes" because int6 ≈ EMA, not because the score is competitive.

## Suggested follow-ups

- **Re-run on 8×H100.** Single-GPU confounds this result; repeat at
  `world_size:8` so step count isn't the dominant variable.
- **Raise the warmup floor.** Sweep `WARMUP_STEPS ∈ {50, 100, 250}` to
  kill the step-2 spike; 10 is clearly below the stability threshold.
- **Decouple warmdown from iteration count.** Drive warmdown by
  fraction-of-wallclock (or by reached-step estimate) rather than a
  fixed `WARMDOWN_ITERS` that is interpreted against an unreachable
  20 k schedule.
- **Re-anchor against the 1.081 recipe.** Fork from the current leader
  recipe (`rec_20260410_2026_04_09_sp8192_3layerrecur_parresid_q_...`)
  instead of default `train_gpt.py`, so schedule sweeps are measured on
  the composable frontier.
- **Sweep paired with Muon.** Schedule compression likely interacts with
  Parallel Muon / Muon WD momentum — joint test before declaring the
  idea dead.
