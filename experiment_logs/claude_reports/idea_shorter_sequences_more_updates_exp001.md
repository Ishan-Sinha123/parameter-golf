# seq512 half-batch

## Hypothesis

Halving `TRAIN_SEQ_LEN` from 1024 → 512 while also halving
`TRAIN_BATCH_TOKENS` to 262,144 should shorten each forward/backward and
buy roughly 2× more optimizer steps inside the same wallclock window. The
bet is that at this tiny (~17M param) scale the extra gradient updates
compound faster than the long-range signal lost from the shorter context,
yielding a lower final BPB under the fixed budget.

## Configuration

| Env override         | Value      |
| -------------------- | ---------- |
| `TRAIN_SEQ_LEN`      | `512`      |
| `TRAIN_BATCH_TOKENS` | `262144`   |

- Recipe: `null` — raw `train_gpt.py` baseline with env overrides only.
- Source ref: _(none)_
- Reproduction: no
- Screening rig (from log, not the full 8×H100 target): `world_size:1`,
  `grad_accum_steps:8`, `iterations:20000`, `warmup_steps:20`,
  `max_wallclock_seconds:480.000`, `seed:1337`.
- Model: `model_params:17059912`, `attention_mode:gqa num_heads:8
  num_kv_heads:4`, `tie_embeddings:True`,
  `embed_lr:0.05 matrix_lr:0.04 scalar_lr:0.04`.
- Log file:
  `experiment_logs/idea_shorter_sequences_more_updates/idea_shorter_sequences_more_updates_exp001/train.log`

Key log lines (quoted verbatim):

```
train_batch_tokens:262144 train_seq_len:512 iterations:20000 warmup_steps:20 max_wallclock_seconds:480.000
step:1000/20000 val_loss:2.4654 val_bpb:1.4601 train_time:176175ms step_avg:176.18ms
step:2000/20000 val_loss:2.3270 val_bpb:1.3782 train_time:350938ms step_avg:175.47ms
step:2735/20000 val_loss:2.2548 val_bpb:1.3354 train_time:480159ms step_avg:175.56ms
stopping_early: wallclock_cap train_time:480159ms step:2735/20000
Serialized model int8+zlib: 15577728 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
final_int8_zlib_roundtrip val_loss:2.2583 val_bpb:1.3375 eval_time:9940ms
final_int8_zlib_roundtrip_exact val_loss:2.25825552 val_bpb:1.33746612
```

No warnings, no divergence. Peak memory: 5332 MiB allocated / 5752 MiB
reserved.

## Results

| Metric                                 | Value        | Δ vs baseline (1.081) |
| -------------------------------------- | ------------ | --------------------- |
| screen_ema_bpb                         | **1.31512**  | **+0.23412**          |
| gate_int6_bpb (int8+zlib roundtrip)    | 1.33750      | +0.25650              |
| gate_quant_gap                         | 3.388e-05    | negligible (✅)        |
| gate_artifact_mb (pipeline field)      | 0.00         | raw payload 15,577,728 B ≈ 15.58 MB — under 16 MB cap |
| gate_passed                            | **true**     | quant/artifact only, not a BPB gate |
| Steps completed                        | 2735 / 20000 | wallclock-capped at 480 s |
| step_avg                               | 175.56 ms    | —                     |
| promote_ema_bpb / promote_int6_bpb     | null         | not promoted          |

Important caveat: the 1.081 baseline is the current SOTA-chain reference
(8×H100, full budget), while this run is a 1-GPU / 480-second screen.
Some of the absolute BPB gap is a scale/budget artifact, not a technique
verdict. Still, the gate reported `gate_passed=true` only because the
quant gap is tiny and the artifact clears 16 MB — nothing about that
field implies BPB competitiveness.

## Verdict

**regression**

Screen EMA BPB of 1.31512 is +0.234 nats above the 1.081 baseline, and
the int8+zlib roundtrip at 1.33747 is worse still. Halving sequence
length *and* halving batch tokens appears to give back more per-step
signal than the hoped-for extra updates recover — and the run is still
step-starved (stopped at 2735/20000 when the 480 s cap fired), so the
promised "2× more updates" isn't even being realized in full within the
screen. Quant gap (3.4e-05) and artifact size (15.58 MB) are both
healthy, but neither compensates for the BPB deficit. Not a candidate
for promotion to the 8×H100 target.

## Suggested follow-ups

- **Isolate the lever.** Run a matched seq1024 control on the same
  1-GPU / 480 s screen rig, same seed, so the seq-length change is the
  only variable and the "more updates per wallclock" claim can actually
  be measured.
- **Keep batch tokens fixed.** Re-run with `TRAIN_SEQ_LEN=512` but leave
  `TRAIN_BATCH_TOKENS` at the baseline default — this matches the stated
  hypothesis without simultaneously halving tokens-per-step.
- **Softer midpoint.** Try `TRAIN_SEQ_LEN=768` with default batch tokens
  to buy some step-count headroom without flattening long-range signal as
  aggressively.
- **Retune LR/warmup for short context.** `warmup_steps:20` and the
  current `matrix_lr:0.04 / embed_lr:0.05` were tuned for seq1024; short
  sequences likely want a longer warmup and/or a modestly higher LR.
- **Curriculum.** seq-512 for the warmup + early phase, then switch to
  seq-1024+ for the final phase, so you burn warmup cheaply but still
  recover long-range BPB at the end.
- **Do not escalate to 8×H100 until a matched screen control wins.**
  Step-avg and memory scale very differently at full parallelism; spend
  that budget only once the screen rig shows a clear seq-length effect.
