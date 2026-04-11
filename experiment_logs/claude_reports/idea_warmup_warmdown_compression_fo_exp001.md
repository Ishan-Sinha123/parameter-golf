# tight-schedule

## Hypothesis

Spend more of the training budget at peak learning rate by compressing both ends of the LR schedule: a near-instant warmup (`WARMUP_STEPS=10`) and a short warmdown (`WARMDOWN_ITERS=600`). If the stock schedule over-invests in the LR ramp and tail, shaving those phases should accelerate loss descent within the 480 s wallclock cap and drop final BPB relative to the current frontier baseline.

## Configuration

| Env override | Value |
| --- | --- |
| `WARMUP_STEPS` | `10` |
| `WARMDOWN_ITERS` | `600` |

Recipe: **none** — ad-hoc env override on default `train_gpt.py` (`recipe_id: null`). Not a reproduction (`is_reproduction: false`); `source_ref` empty.

Run context from `train.log`:

- `model_params:17059912`
- `world_size:1 grad_accum_steps:8` (single-GPU screen, not 8×H100)
- `attention_mode:gqa num_heads:8 num_kv_heads:4`
- `tie_embeddings:True embed_lr:0.05 matrix_lr:0.04 scalar_lr:0.04`
- `iterations:20000 warmup_steps:10 max_wallclock_seconds:480.000`
- `train_batch_tokens:524288 train_seq_len:1024 seed:1337`

## Results

| Metric | Value | Baseline (1.081) | Δ vs baseline |
| --- | --- | --- | --- |
| Screen EMA val_bpb | **1.28935** | 1.081 | **+0.20835** |
| Gate int6 val_bpb | 1.31900 | 1.081 | +0.23800 |
| Quant gap (int6 − EMA) | ~6e-7 | — | ≈ 0 |
| Artifact size (int8+zlib) | 14,748,471 B | 16,000,000 cap | 1.25 MB under cap |
| Steps completed | 1,439 / 20,000 | — | wallclock stop |
| Total train time | 480.2 s | — | cap hit |
| Step avg | ~333.7 ms | — | — |
| Peak memory | 10,303 MiB | — | — |
| Gate passed | ✅ true | — | — |
| Promoted | ❌ no (`promote_*` null) | — | — |

### Key lines from `train.log`

```
train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:10 max_wallclock_seconds:480.000
warmup_step:10/10
step:0/20000 val_loss:6.9357 val_bpb:4.1077
step:1/20000 train_loss:6.9357
step:2/20000 train_loss:16.7413          # spike from ultra-short warmup
step:3/20000 train_loss:8.7525
step:4/20000 train_loss:6.5885
step:1000/20000 val_loss:2.3175 val_bpb:1.3726
step:1439/20000 val_loss:2.2251 val_bpb:1.3179
stopping_early: wallclock_cap train_time:480184ms step:1439/20000
peak memory allocated: 10303 MiB reserved: 10622 MiB
Serialized model int8+zlib: 14700778 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
Total submission size int8+zlib: 14748471 bytes
final_int8_zlib_roundtrip val_loss:2.2271 val_bpb:1.3190 eval_time:10812ms
final_int8_zlib_roundtrip_exact val_loss:2.22707524 val_bpb:1.31899940
```

No warnings emitted. Loss was still descending at the wallclock stop (val_bpb 1.3726 → 1.3179 between step 1000 and 1439).

### Observations

1. **Step-2 loss spike (16.74).** The 10-step warmup is below the stability floor for this optimizer config — training recovers by step 4 but ~10 early steps are burned absorbing the spike.
2. **Early stop at step 1,439 / 20,000.** The single-GPU screen hit the 480 s cap at ~7.2 % of the planned 20k schedule, so `WARMDOWN_ITERS=600` (warmdown would begin at step 19,400) was **never entered**. The tested schedule was effectively "10-step warmup + constant peak LR for 1,429 steps" — the warmdown half of the hypothesis was not actually exercised.
3. **Quant gap ≈ 0.** int8+zlib roundtrip added ~6 × 10⁻⁷ BPB; weights quantize cleanly and the artifact lands at 14.75 MB, 1.25 MB under the 16 MB cap.
4. **Baseline confound.** Baseline 1.081 is the 8×H100 stacked-SOTA frontier; this run is plain `train_gpt.py` on a single GPU. The +0.208 gap conflates the schedule change with a step-count deficit and the absence of the frontier feature stack, so this cannot cleanly falsify the hypothesis on its own.

## Verdict

**regression** — screen EMA `1.28935` sits +0.208 BPB above the 1.081 SOTA baseline, the compressed warmup triggered an early loss spike at step 2, and the system did not promote (`promote_ema_bpb` / `promote_int6_bpb` both null). The gate "passes" mechanically because int6 ≈ EMA, not because the score is competitive. Given the single-GPU / truncated-schedule confound (warmdown never actually ran), read this as "did not advance" rather than as strong evidence against schedule compression itself.

## Suggested follow-ups

- **Raise the warmup floor.** Sweep `WARMUP_STEPS ∈ {50, 100, 250}` to eliminate the step-2 spike; 10 is clearly below the stability threshold for this LR triple.
- **Matched-budget control.** Re-run stock `WARMUP_STEPS` / `WARMDOWN_ITERS` at the same single-GPU 480 s screen so the tight-schedule delta is measured against a like-for-like control instead of the 8×H100 frontier.
- **Decouple warmdown from iteration count.** Drive warmdown by fraction of reached-step (or fraction of wallclock) rather than a fixed `WARMDOWN_ITERS` interpreted against an unreachable 20k schedule — otherwise warmdown silently never runs at screen scale.
- **Re-anchor against the 1.081 recipe.** Fork from the current leader recipe instead of default `train_gpt.py`, so schedule sweeps are measured on the composable frontier where promotion is realistic.
- **Joint test with Muon momentum.** Schedule compression likely interacts with Parallel Muon / Muon WD; sweep them together before declaring the idea dead.
- **Grad-clip guard for tight warmup.** If ultra-short warmup is retained, add a temporary grad clip for the first ~20 steps to absorb the spike without lengthening the LR ramp.
- **≥ 3 seeds before any PR.** Once a winning schedule survives full 8×H100 training on the SOTA recipe, replicate with multiple seeds to clear the 0.005-nat, p < 0.01 record bar.
