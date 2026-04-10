# tight-schedule

## Hypothesis

Spend more of the 10-minute training budget at peak learning rate by
compressing both ends of the schedule: a near-instant warmup
(`WARMUP_STEPS=10`) and a short warmdown (`WARMDOWN_ITERS=600`). Theory is
that the stock schedule over-invests in the LR ramp, so removing that
investment should accelerate loss descent within the wallclock cap and
drop final BPB.

## Configuration

| Env override | Value |
| --- | --- |
| `WARMUP_STEPS` | `10` |
| `WARMDOWN_ITERS` | `600` |

Recipe: **none** — ad-hoc env-override on default `train_gpt.py`
(`recipe_id: null`). Not a reproduction. Source ref not recorded.

Run context from `train.log`:

- `model_params:17059912`
- `world_size:1 grad_accum_steps:8` (single-GPU screen, not 8×H100)
- `attention_mode:gqa num_heads:8 num_kv_heads:4`
- `tie_embeddings:True embed_lr:0.05 matrix_lr:0.04 scalar_lr:0.04`
- `iterations:20000 warmup_steps:10 max_wallclock_seconds:480.000`
- `seed:1337`

## Results

| Metric | Value | Baseline | Δ vs baseline |
| --- | --- | --- | --- |
| Screen EMA val_bpb | **1.28935** | 1.10625 | **+0.18310** |
| Gate int6 val_bpb | 1.31900 | 1.10625 | +0.21275 |
| Quant gap (int6 − EMA) | ~6e-7 | — | ≈0 |
| Artifact size (int8+zlib) | 14,748,471 B | 16 MB cap | under cap |
| Steps completed | 1,439 / 20,000 | — | wallclock stop |
| Total train time | 480.2 s | — | cap hit |
| Step avg | ~333.7 ms | — | — |
| Gate passed | ✅ true | — | — |
| Promoted | ❌ no (`promote_*` null) | — | — |

### Key lines from `train.log`

```
warmup_step:10/10
step:0/20000 val_loss:6.9357 val_bpb:4.1077
step:2/20000 train_loss:16.7413          # spike from ultra-short warmup
step:3/20000 train_loss:8.7525
step:4/20000 train_loss:6.5885
step:1000/20000 val_loss:2.3175 val_bpb:1.3726
step:1439/20000 val_loss:2.2251 val_bpb:1.3179
stopping_early: wallclock_cap train_time:480184ms step:1439/20000
Serialized model int8+zlib: 14700778 bytes (payload_ratio:3.91x)
final_int8_zlib_roundtrip val_loss:2.22707524 val_bpb:1.31899940
```

No warnings emitted. Loss was still descending at the wallclock stop.

### Observations

1. **Step-2 loss spike (16.74).** The 10-step warmup sits below the
   stability floor for this optimizer config; training recovers by step 4
   but the spike costs early progress.
2. **Early stop at step 1,439 / 20,000.** The single-GPU screen hit the
   480 s cap at ~7% of the planned schedule, so `WARMDOWN_ITERS=600` was
   never exercised at its intended relative position inside a 20 k run.
3. **Quant gap ≈ 0.** int8+zlib roundtrip added essentially no BPB —
   weights quantize cleanly at this config.
4. **Baseline confound.** Baseline 1.10625 is the 8×H100 SOTA recipe;
   this run is single-GPU default `train_gpt.py`. The +0.183 gap
   conflates the schedule change with both a step-count deficit and the
   absence of the frontier feature stack, so this cannot cleanly falsify
   the hypothesis on its own.

## Verdict

**regression** — screen EMA 1.28935 is +0.18310 BPB above the SOTA
baseline, compressed warmup triggered an early loss spike at step 2, and
the system did not promote (`promote_ema_bpb` / `promote_int6_bpb` both
null). The gate "passes" only mechanically because int6 ≈ EMA, not
because the score is competitive. Given the single-GPU / truncated-
schedule confound, read this as "did not advance" rather than as strong
evidence against the idea itself.

## Suggested follow-ups

- **Raise the warmup floor.** Sweep `WARMUP_STEPS ∈ {50, 100, 250}` to
  kill the step-2 spike; 10 is clearly below the stability threshold for
  this LR.
- **Matched-budget control.** Re-run stock `WARMUP_STEPS` /
  `WARMDOWN_ITERS` at the same single-GPU screen budget so the
  tight-schedule delta is directly measurable instead of comparing to the
  8×H100 frontier.
- **Decouple warmdown from iteration count.** Drive warmdown by fraction
  of reached-step (or fraction of wallclock) rather than a fixed
  `WARMDOWN_ITERS` interpreted against an unreachable 20 k schedule.
- **Re-anchor against the 1.1063 recipe.** Fork from the current leader
  recipe instead of default `train_gpt.py`, so schedule sweeps are
  measured on the composable frontier where promotion is realistic.
- **Joint test with Muon momentum.** Schedule compression likely
  interacts with Parallel Muon / Muon WD; sweep them together before
  declaring the idea dead.
- **Grad-clip guard for tight warmup.** If tight warmup is retained, add
  a temporary grad clip for the first ~20 steps to absorb the spike
  without lengthening the LR ramp.
