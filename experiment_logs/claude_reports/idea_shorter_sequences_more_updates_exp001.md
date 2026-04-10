# seq512 half-batch

## Hypothesis

Halving `TRAIN_SEQ_LEN` from 1024 → 512 (with `TRAIN_BATCH_TOKENS` pinned
at 262,144) should shorten each forward/backward enough to roughly double
the number of optimizer steps that fit inside the wallclock cap. The bet
is that the extra gradient updates compound faster than the signal lost
from the shorter context, yielding a lower final BPB at the same budget.

## Configuration

| Env override | Value |
| --- | --- |
| `TRAIN_SEQ_LEN` | `512` |
| `TRAIN_BATCH_TOKENS` | `262144` |

Recipe: none (`recipe_id: null`) — raw baseline with env overrides only.
Screening rig: `world_size:1`, `grad_accum_steps:8`, `iterations:20000`,
`warmup_steps:20`, `max_wallclock_seconds:480`. Model:
`model_params:17059912`, `attention_mode:gqa num_heads:8 num_kv_heads:4`,
`tie_embeddings:True`, `embed_lr:0.05 matrix_lr:0.04 scalar_lr:0.04`.

Log file:
`experiment_logs/idea_shorter_sequences_more_updates/idea_shorter_sequences_more_updates_exp001/train.log`.

Key log lines:

```
train_batch_tokens:262144 train_seq_len:512 iterations:20000 warmup_steps:20 max_wallclock_seconds:480.000
step:1000/20000 val_loss:2.4654 val_bpb:1.4601
step:2000/20000 val_loss:2.3270 val_bpb:1.3782
step:2735/20000 val_loss:2.2548 val_bpb:1.3354 train_time:480159ms step_avg:175.56ms
stopping_early: wallclock_cap train_time:480159ms step:2735/20000
Serialized model int8+zlib: 15577728 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
final_int8_zlib_roundtrip val_loss:2.2583 val_bpb:1.3375 eval_time:9940ms
final_int8_zlib_roundtrip_exact val_loss:2.25825552 val_bpb:1.33746612
```

No warnings. Peak memory 5332 MiB reserved 5752 MiB.

## Results

| Metric | Value | Δ vs baseline 1.10625353 |
| --- | --- | --- |
| screen_ema_bpb | 1.31512373 | **+0.20887** |
| gate_int6_bpb (int8+zlib roundtrip) | 1.33750000 | +0.23125 |
| gate_quant_gap | 3.388e-05 | negligible |
| gate_artifact_mb (pipeline field) | 0.0 | raw int8+zlib payload 15,577,728 B ≈ 15.58 MB, under 16 MB cap |
| gate_passed | **true** | quant gap & artifact clear, not a BPB gate |
| Steps completed | 2735 / 20000 | wallclock-capped at 480 s |
| step_avg | 175.56 ms | — |
| promote_ema_bpb / promote_int6_bpb | null | not promoted |

Caveat: the 1.10625 baseline is the 8×H100 / 600 s full-scale reference
(the current record chain). This run is a 1×GPU / 480 s screen, so the
absolute gap is partially a scale artifact. Even so, at the same screen
scale the BPB is well above what competitive screens land at, and the
quant-gap/artifact side of `gate_passed` does not say anything about
BPB competitiveness.

## Verdict

**regression**

Screen EMA BPB 1.3151 is +0.209 nats above the 1.10625 baseline and the
int8+zlib gate at 1.3375 is +0.231 above. Halving sequence length while
also halving batch tokens hands back more per-token signal than the
extra updates recover, and the run is still step-capped (2735 / 20000)
inside the 480 s screen, so the "2× more updates" payoff is not even
being realized in full. `gate_passed=true` is only reflecting the
near-zero quant gap and the under-cap artifact, not a BPB win.

## Suggested follow-ups

- Disentangle "more steps" from "shorter context": run a step-matched
  control at the default `TRAIN_SEQ_LEN` on the same 1-GPU screen rig.
- Keep `TRAIN_BATCH_TOKENS` at the baseline value and only shrink
  `TRAIN_SEQ_LEN` — this is the version closest to the stated hypothesis
  and doesn't simultaneously halve tokens-per-step.
- Softer midpoint: `TRAIN_SEQ_LEN=768` with default batch tokens, to
  buy some step headroom without flattening long-range signal.
- Retune the LR schedule for short sequences — `warmup_steps:20` and the
  current `matrix_lr` / `embed_lr` are tuned for longer-context runs and
  may be under-baking the early steps at seq-512.
- Mixed-length curriculum: seq-512 for the first phase to burn through
  warmup cheaply, switch to seq-1024+ for the final phase to recover
  long-range BPB.
- Before drawing any final conclusion, re-run on the actual 8×H100 /
  600 s target. Step-avg and memory scale very differently at full
  parallelism, and that is the only setting where "more updates in
  wallclock" can actually convert into a record.
