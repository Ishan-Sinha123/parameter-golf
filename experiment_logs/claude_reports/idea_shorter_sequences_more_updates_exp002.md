# seq2048 double-batch

**Experiment ID:** `idea_shorter_sequences_more_updates_exp002`
**Date:** 2026-04-10
**Stage:** gate (1 GPU, `max_wallclock_seconds:480`)

## Hypothesis

Longer context (seq_len = 2048) may yield better BPB if the higher per-step
cost is offset by the richer contextual signal per token, even though fewer
optimizer steps complete in the 480 s gate budget. Per-step batch tokens
are also doubled to 1,048,576 so the number of sequences per step stays in
the same ballpark as the 1024-seq default — testing whether longer context
amortizes fewer total updates.

## Configuration

| Env override         | Value        | Default |
|----------------------|--------------|---------|
| `TRAIN_SEQ_LEN`      | **2048**     | 1024    |
| `TRAIN_BATCH_TOKENS` | **1048576**  | 524288  |

- Recipe: *none* (`recipe_id = null`; plain `train_gpt.py` with env overrides)
- Source ref: _(empty)_ — not a reproduction
- Stage: `gate` (1 GPU, `grad_accum_steps:8`, `iterations:20000`,
  `max_wallclock_seconds:480`)
- Model: 17,059,912 params, GQA 8/4 heads, tied embeddings, seed 1337

### Key log lines

```
model_params:17059912
world_size:1 grad_accum_steps:8
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

No NaNs or numerical warnings; run exited cleanly via `wallclock_cap`.

## Results

Baseline for delta = **1.10625** val_bpb.

| Metric                                | Value                             | Δ vs baseline (1.10625) |
|---------------------------------------|-----------------------------------|-------------------------|
| `screen_ema_bpb`                      | 1.36271                           | **+0.25646**            |
| `gate_int6_bpb` (int8+zlib roundtrip) | 1.37770                           | **+0.27146**            |
| `gate_quant_gap`                      | −1.04 × 10⁻⁵                      | ≈ 0                     |
| `gate_artifact_mb`                    | 0.00 reported / ~10.97 MB in log  | well under 16 MB        |
| `gate_passed`                         | true                              | —                       |
| `promote_ema_bpb`                     | null                              | —                       |
| `promote_int6_bpb`                    | null                              | —                       |
| Steps completed                       | **642 / 20,000**                  | — (wallclock-capped)    |
| Step avg                              | 747.9 ms                          | — (~4.2× the short-ctx default) |
| Wallclock                             | 480,149 ms (early-stopped at cap) | —                       |
| Peak memory                           | 20,245 MiB                        | —                       |

At ~748 ms/step, only 642 of the intended 20,000 steps finished inside
the 480 s gate cap. Train loss was still descending monotonically (2.60 →
2.42 → 2.35 → 2.31) when the cap fired — the model was nowhere near
convergence. The int8+zlib quant gap is effectively zero, but the absolute
BPB is ~0.26–0.27 nats **worse** than the 1.10625 baseline. `gate_passed =
true` here reflects the near-zero quant gap (a sanity check), not an
absolute-BPB win.

## Verdict

**regression.**

Doubling sequence length to 2048 at the gate-stage budget is severely
step-starved: 642 updates in the 480 s window, versus thousands for
short-context siblings. Final int8+zlib val_bpb of **1.3777** is
**+0.27146 nats above the 1.10625 baseline**, and screen EMA val_bpb of
**1.36271** is **+0.25646 above** — both far outside the 0.005-nat record
bar. Memory also ballooned to ~20 GiB, narrowing headroom for stacking.
The hypothesis (longer context converges in fewer steps) cannot be
*validated* from this run because training terminated long before any
crossover with the short-context baseline, but it is clearly *falsified*
under the current gate budget.

## Suggested follow-ups

- **Isolate the variable:** re-run with `TRAIN_SEQ_LEN=2048` but keep
  `TRAIN_BATCH_TOKENS` at the default 524288 so the sequence-length
  effect isn't conflated with the doubled per-step token budget.
- **Intermediate points:** sweep `TRAIN_SEQ_LEN ∈ {768, 1024, 1536}` at
  fixed batch tokens to locate the sweet spot on the context-vs-steps
  frontier.
- **Promote stage, not gate:** long-context runs are step-count bound
  and the 1-GPU / 480 s gate systematically penalizes them. If the idea
  is worth pursuing, test directly on the 8×H100 promote budget so it
  gets a fair step count.
- **Curriculum:** keep seq_len=1024 for the bulk of training and switch
  to 2048 only in the warmdown tail, so the model sees long context
  without losing early-training update throughput.
- **Compressed schedule for long-seq:** if seq_len=2048 is kept at the
  current budget, retune warmup/warmdown for the realistic ~600-step
  horizon — the default 20k-step schedule is irrelevant at 642 completed
  steps.
- **Pair with cheaper attention:** sliding-window or XSA on all layers
  might recover enough step-time to make long-context tractable at this
  budget; the current dense-GQA 2048 path is ~4.2× slower than the
  short-context default.
- **Try the other half:** the sibling `exp001` tested shorter sequences
  with more updates; cross-reference that run to confirm the "more
  updates" half of the original idea direction outperforms, even if the
  "longer context" half here doesn't.
