# seq512 half-batch

## Hypothesis

Halving the training sequence length to 512 while keeping
`TRAIN_BATCH_TOKENS=262144` was expected to roughly double the number of
optimizer steps that fit inside the 10-minute wallclock window. The bet:
more SGD updates at shorter context would compound faster than the loss
we give up from the missing long-range tokens, yielding a lower final
BPB than the default-context baseline.

## Configuration

| Env override | Value |
|---|---|
| `TRAIN_SEQ_LEN` | `512` |
| `TRAIN_BATCH_TOKENS` | `262144` |

Recipe: none (`recipe_id: null`) — run against the current repo default.
From the log: `model_params:17059912`, `attention_mode:gqa num_heads:8
num_kv_heads:4`, `tie_embeddings:True`, `embed_lr:0.05 matrix_lr:0.04
scalar_lr:0.04`, `warmup_steps:20`, planned `iterations:20000`,
`max_wallclock_seconds:480.000`, `world_size:1 grad_accum_steps:8`,
`seed:1337`.

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

| Metric | Value | Δ vs baseline (1.081) |
|---|---|---|
| screen_ema_bpb | 1.31512 | **+0.23412** |
| gate_int6_bpb (int8+zlib roundtrip) | 1.33747 | **+0.25647** |
| gate_quant_gap | 3.39e-05 | negligible |
| gate_artifact_mb (recorded) | 0.0 | (pipeline reported 0; raw artifact = 15.625 MB, under cap) |
| Steps completed | 2735 / 20000 | wallclock-capped at 480 s |
| step_avg | 175.56 ms | — |
| Gate passed (pipeline) | `true` | — |
| Promoted | no | `promote_*` fields null |

No warnings in the log. The pipeline flagged `gate_passed: true` purely
because the quant gap was tiny — the absolute BPB is nowhere near the
1.081 SOTA baseline (+0.256 nats worse). The run was step-capped: at
~175 ms/step the 480 s wall only buys ~2735 of the 20000 planned steps,
so the "2× more updates" claim is better tested against a same-setup
seq1024 screen than against SOTA.

## Verdict

**regression** — int8+zlib gate BPB 1.3375 is +0.2565 nats above the
1.081 baseline, and the screen EMA (1.3151) is +0.2341 above. The
shorter-sequence configuration does not come close to current SOTA; any
"more steps" benefit is swamped by lost per-token signal and by the run
still being wallclock-bound.

## Suggested follow-ups

- Step-count-matched control at the default seq_len on the same 1-GPU
  screen rig, so we isolate "more steps" from "shorter context."
- Softer midpoint: `TRAIN_SEQ_LEN=1024` with default batch tokens, to
  keep some extra step headroom without crushing long-range signal.
- Pair short sequences with a retuned LR schedule (higher peak LR,
  shorter warmdown) since the current 20-step warmup is tuned for
  longer-context budgets.
- Mixed-length curriculum: seq512 early for fast updates, switch to
  seq1024+ in the final phase to recover long-range loss.
- Cross-check exp002 in the same idea bucket before closing the idea,
  and avoid re-running this exact combo until the bottleneck is
  reclassified (step-cost vs token-cost).
