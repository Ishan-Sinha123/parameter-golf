# seq512 half-batch

## Hypothesis

Halving the training sequence length to 512 while also dropping
`TRAIN_BATCH_TOKENS` to 262,144 was expected to roughly double the number
of optimizer steps that fit inside the 10-minute (480 s screen)
wallclock window. The bet: more gradient updates at shorter context
would compound faster than the signal we give up from the missing
long-range tokens, yielding a lower final BPB than the default-context
baseline.

## Configuration

| Env override | Value |
|---|---|
| `TRAIN_SEQ_LEN` | `512` |
| `TRAIN_BATCH_TOKENS` | `262144` |

Recipe: none (`recipe_id: null`) — run launched against the current repo
default. From the training log:

```
model_params:17059912
world_size:1 grad_accum_steps:8
attention_mode:gqa num_heads:8 num_kv_heads:4
tie_embeddings:True embed_lr:0.05 head_lr:0.0 matrix_lr:0.04 scalar_lr:0.04
train_batch_tokens:262144 train_seq_len:512 iterations:20000 warmup_steps:20 max_wallclock_seconds:480.000
seed:1337
```

## Results

Key lines from
`experiment_logs/idea_shorter_sequences_more_updates/idea_shorter_sequences_more_updates_exp001/train.log`:

```
step:1000/20000 val_loss:2.4654 val_bpb:1.4601
step:2000/20000 val_loss:2.3270 val_bpb:1.3782
step:2735/20000 val_loss:2.2548 val_bpb:1.3354 train_time:480159ms step_avg:175.56ms
stopping_early: wallclock_cap train_time:480159ms step:2735/20000
Serialized model int8+zlib: 15577728 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
Total submission size int8+zlib: 15625421 bytes
final_int8_zlib_roundtrip val_loss:2.2583 val_bpb:1.3375 eval_time:9940ms
final_int8_zlib_roundtrip_exact val_loss:2.25825552 val_bpb:1.33746612
```

| Metric | Value | Δ vs baseline (1.10625353) |
|---|---|---|
| screen_ema_bpb | 1.31512373 | **+0.20887** |
| gate_int6_bpb (int8+zlib roundtrip) | 1.33750000 | **+0.23125** |
| gate_quant_gap | 3.388e-05 | negligible |
| gate_artifact_mb (recorded) | 0.0 | pipeline reported 0; raw artifact = 15.625 MB, under cap |
| Steps completed | 2735 / 20000 | wallclock-capped at 480 s |
| step_avg | 175.56 ms | — |
| Gate passed (pipeline) | `true` | — |
| Promoted | no | `promote_*` fields null |

No warnings in the log. `gate_passed=true` is purely because the quant
gap was ~3.4e-5 nats and the artifact fit under the 16 MB cap — the
absolute BPB is a large regression from the 1.10625 baseline. The run
was wallclock-bound: at ~175 ms/step, the 480 s screen only buys ~2735 of
the 20000 planned steps, so the "2× more updates" story is better
tested against a same-setup seq_len sweep than against the current SOTA.

## Verdict

**regression** — screen EMA BPB 1.3151 is +0.2089 nats above the
1.10625 baseline, and the int8+zlib gate at 1.3375 is +0.2313 above.
Halving sequence length while also halving batch tokens gives up far
more per-token signal than the extra steps recover, and the run is still
step-capped inside the screen wallclock anyway.

## Suggested follow-ups

- Step-count-matched control at the default `TRAIN_SEQ_LEN` on the same
  1-GPU screen rig, so we isolate "more steps" from "shorter context."
- Keep `TRAIN_BATCH_TOKENS` at the baseline value and only shrink
  `TRAIN_SEQ_LEN` — microbatch count rises without halving tokens/step,
  which is the config closer to the original hypothesis.
- Softer midpoint: `TRAIN_SEQ_LEN=1024` with default batch tokens, to
  keep some extra step headroom without crushing long-range signal.
- Pair short sequences with a retuned LR schedule (higher peak LR,
  shorter warmdown) since the current 20-step warmup is tuned for
  longer-context budgets.
- Mixed-length curriculum: seq512 early for fast updates, switch to
  seq1024+ in the final phase to recover long-range loss.
- Re-measure on an 8×H100 run (not 1-GPU screen) before concluding —
  the 175 ms/step overhead will be very different at full parallelism,
  and the "more updates" claim may only materialize there.
