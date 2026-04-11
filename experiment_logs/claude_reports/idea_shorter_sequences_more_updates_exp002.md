# seq2048 double-batch

**Experiment ID:** `idea_shorter_sequences_more_updates_exp002`
**Reported:** 2026-04-11
**Stage:** gate (1 GPU, `max_wallclock_seconds:480`)

## Hypothesis

Longer context (`TRAIN_SEQ_LEN=2048`) may yield better BPB if the higher
per-step cost is offset by the richer contextual signal per token and by
needing fewer updates overall. Per-step batch tokens are also doubled to
1,048,576, so the number of sequences per step stays in the same ballpark
as the 1024-seq default — testing whether longer context can amortize
fewer total updates inside the gate wallclock budget.

## Configuration

| Env override         | Value        | Default |
|----------------------|--------------|---------|
| `TRAIN_SEQ_LEN`      | **2048**     | 1024    |
| `TRAIN_BATCH_TOKENS` | **1048576**  | 524288  |

- Recipe: *none* (`recipe_id = null`; plain `train_gpt.py` with env overrides only)
- Source ref: *(empty)* — not a reproduction
- Gate rig: `world_size:1`, `grad_accum_steps:8`, `iterations:20000`,
  `warmup_steps:20`, `max_wallclock_seconds:480`, `seed:1337`
- Model: 17,059,912 params, GQA `num_heads:8 num_kv_heads:4`, tied
  embeddings, `embed_lr:0.05 matrix_lr:0.04 scalar_lr:0.04`
- Log:
  `experiment_logs/idea_shorter_sequences_more_updates/idea_shorter_sequences_more_updates_exp002/train.log`

### Key log lines

```
model_params:17059912
world_size:1 grad_accum_steps:8
attention_mode:gqa num_heads:8 num_kv_heads:4
train_batch_tokens:1048576 train_seq_len:2048 iterations:20000 warmup_steps:20 max_wallclock_seconds:480.000
step:200/20000 train_loss:2.5989 train_time:149622ms step_avg:748.11ms
step:400/20000 train_loss:2.4223 train_time:299345ms step_avg:748.36ms
step:600/20000 train_loss:2.3496 train_time:448865ms step_avg:748.11ms
step:642/20000 val_loss:2.3137 val_bpb:1.3703 train_time:480149ms step_avg:747.89ms
stopping_early: wallclock_cap train_time:480149ms step:642/20000
peak memory allocated: 20245 MiB reserved: 20856 MiB
Serialized model int8+zlib: 10919521 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
final_int8_zlib_roundtrip val_loss:2.3262 val_bpb:1.3777 eval_time:12693ms
final_int8_zlib_roundtrip_exact val_loss:2.32620627 val_bpb:1.37771038
```

No NaNs or numerical warnings; the run exited cleanly via `wallclock_cap`.

## Results

| Metric                                | Value          | Δ vs baseline 1.081 |
|---------------------------------------|----------------|---------------------|
| `screen_ema_bpb`                      | **1.36271**    | **+0.28171**        |
| `gate_int6_bpb` (int8+zlib roundtrip) | 1.37770        | +0.29670            |
| `gate_quant_gap`                      | −1.04 × 10⁻⁵   | ≈ 0 (sanity ok)     |
| `gate_artifact_mb`                    | 0.00 reported  | ~10.92 MB in log, under 16 MB cap |
| `gate_passed`                         | true           | quant/artifact check only, not a BPB win |
| `promote_*_bpb`                       | null / null    | not promoted        |
| Steps completed                       | **642 / 20000**| wallclock-capped at 480 s |
| Step avg                              | 747.9 ms       | ~4× the short-ctx default |
| Peak memory                           | 20,245 MiB     | ~4× the seq512 sibling |

Caveats: the 1.081 baseline is the current SOTA-chain reference (8×H100,
full-budget run). This is a 1-GPU / 480 s screen, so some of the absolute
gap is a scale artifact rather than a technique verdict. But the
regression is far too large to close even under a fair budget. Train loss
was still descending monotonically (2.60 → 2.42 → 2.35 → 2.31) when the
cap fired, so the model never came near convergence.

## Verdict

**regression**

Doubling sequence length to 2048 at the gate budget is severely
step-starved: only 642 updates fit in the 480 s window. Final int8+zlib
val_bpb of **1.3777** is **+0.297 nats** above the 1.081 SOTA reference,
and screen EMA val_bpb of **1.3627** is **+0.282 nats** above — orders of
magnitude beyond the 0.005-nat record bar. `gate_passed = true` here only
reflects the near-zero quant gap, not any absolute win. The hypothesis
(long context amortizes fewer steps) isn't *validated* from this run
because training terminated long before any plausible crossover with the
short-context baseline, but it is clearly *falsified* under the current
gate budget on 1 GPU.

## Suggested follow-ups

- **Isolate the variable.** Re-run with `TRAIN_SEQ_LEN=2048` but leave
  `TRAIN_BATCH_TOKENS=524288` so sequence length is decoupled from the
  doubled token budget.
- **Sweep intermediate lengths.** Try `TRAIN_SEQ_LEN ∈ {768, 1024, 1536}`
  at fixed batch tokens to find the context/steps sweet spot without
  paying the full 2048 step-time tax.
- **Evaluate on promote, not gate.** Long-context runs are step-count
  bound and the 1-GPU / 480 s gate systematically penalizes them.
  Promote-stage 8×H100 evaluation would give a fair step count.
- **Curriculum.** Keep `seq_len=1024` for the bulk of training and
  switch to 2048 only in the warmdown tail, preserving early throughput.
- **Retune the schedule.** `warmup_steps:20` and the 20k-step target are
  meaningless at 642 completed steps — compress warmup/warmdown to a
  ~600-step horizon if seq2048 is retried at this budget.
- **Pair with cheaper attention.** Sliding-window or all-layer XSA could
  recover enough step time to make long context tractable; dense-GQA 2048
  is ~4× slower than the short-context default here.
- **Cross-check `exp001`.** The seq512 half-batch sibling tested the
  opposite lever; reading both together confirms whether either end of
  the range beats the default seq1024 setting.
