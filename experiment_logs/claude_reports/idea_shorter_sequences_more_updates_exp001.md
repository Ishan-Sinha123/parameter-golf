# seq512 half-batch

**Experiment ID:** `idea_shorter_sequences_more_updates_exp001`
**Reported:** 2026-04-11
**Stage:** gate (1 GPU, `max_wallclock_seconds:480`)

## Hypothesis

Halving the sequence length (`TRAIN_SEQ_LEN=512`) and halving the per-step
token budget (`TRAIN_BATCH_TOKENS=262144`) should shorten each
forward/backward and buy roughly **2× more optimizer steps** inside the
same wallclock window. The bet is that at this tiny (~17M param) scale,
loss improvement is step-bound rather than token-bound, so the denser
update schedule compounds faster than the long-range signal lost from
shorter context — yielding a lower final BPB under the fixed budget.

## Configuration

| Env override         | Value       | Default |
|----------------------|-------------|---------|
| `TRAIN_SEQ_LEN`      | **512**     | 1024    |
| `TRAIN_BATCH_TOKENS` | **262144**  | 524288  |

- Recipe: *none* (`recipe_id = null`; raw `train_gpt.py` baseline with env overrides only)
- Source ref: *(empty)* — not a reproduction
- Gate rig: `world_size:1`, `grad_accum_steps:8`, `iterations:20000`,
  `warmup_steps:20`, `max_wallclock_seconds:480`, `seed:1337`
- Model: 17,059,912 params, GQA `num_heads:8 num_kv_heads:4`, tied
  embeddings, `embed_lr:0.05 matrix_lr:0.04 scalar_lr:0.04`
- Tokenizer: `fineweb_1024_bpe.model` (sentencepiece, vocab 1024)
- Log:
  `experiment_logs/idea_shorter_sequences_more_updates/idea_shorter_sequences_more_updates_exp001/train.log`

### Key log lines

```
model_params:17059912
world_size:1 grad_accum_steps:8
attention_mode:gqa num_heads:8 num_kv_heads:4
train_batch_tokens:262144 train_seq_len:512 iterations:20000 warmup_steps:20 max_wallclock_seconds:480.000
step:0/20000    val_loss:6.9357 val_bpb:4.1077
step:1000/20000 val_loss:2.4654 val_bpb:1.4601 train_time:176175ms step_avg:176.18ms
step:2000/20000 val_loss:2.3270 val_bpb:1.3782 train_time:350938ms step_avg:175.47ms
step:2735/20000 val_loss:2.2548 val_bpb:1.3354 train_time:480159ms step_avg:175.56ms
stopping_early: wallclock_cap train_time:480159ms step:2735/20000
peak memory allocated: 5332 MiB reserved: 5752 MiB
Serialized model int8+zlib: 15577728 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
Total submission size int8+zlib: 15625421 bytes
final_int8_zlib_roundtrip val_loss:2.2583 val_bpb:1.3375 eval_time:9940ms
final_int8_zlib_roundtrip_exact val_loss:2.25825552 val_bpb:1.33746612
```

No NaNs or numerical warnings. A brief early-warmup spike
(`step:2 train_loss:16.82`) recovered by step 4 and training descended
monotonically afterwards. The run exited cleanly via `wallclock_cap` with
val_bpb still descending (1.4601 → 1.3782 → 1.3354 at steps 1k / 2k /
2735).

## Results

| Metric                                | Value            | Δ vs baseline 1.081                     |
|---------------------------------------|------------------|-----------------------------------------|
| `screen_ema_bpb`                      | **1.31512**      | **+0.23412**                            |
| `gate_int6_bpb` (int8+zlib roundtrip) | 1.33747          | +0.25647                                |
| `gate_quant_gap`                      | 3.39 × 10⁻⁵      | ≈ 0 (sanity ok)                         |
| `gate_artifact_mb`                    | 0.00 reported    | ~15.63 MB in log, under 16 MB cap       |
| `gate_passed`                         | true             | quant/artifact check only, not a BPB win |
| `promote_*_bpb`                       | null / null      | not promoted                            |
| Steps completed                       | **2735 / 20000** | wallclock-capped at 480 s               |
| Step avg                              | 175.6 ms         | ~4.3× faster than sibling seq2048       |
| Peak memory                           | 5,332 MiB        | ~4× leaner than sibling seq2048         |

**Hypothesis check.** The sibling seq2048 run (`exp002`) completed only
642 steps in the same 480 s window, so seq512 delivered **~4.26× more
updates** — actually overshooting the predicted 2×. The denser schedule
did pay off *within* the gate: final gate int6 BPB is **1.3375 vs 1.3777
for seq2048** (−0.040 nats), confirming the direction of the hypothesis
relative to the opposite-endpoint run.

**Absolute comparison caveat.** The 1.081 baseline is the current
SOTA-chain reference (8×H100, full promote budget). This is a 1-GPU /
480 s screen, so a large portion of the +0.234 gap is a scale artifact
rather than a technique verdict. Even so, training was still descending
at the cap — the model is nowhere near convergence and would need a much
larger step budget to approach the baseline.

## Verdict

**regression**

Against the 1.081 SOTA reference, final gate int6 val_bpb of **1.3375** is
**+0.256 nats** above baseline and screen EMA val_bpb of **1.3151** is
**+0.234 nats** above — orders of magnitude beyond the 0.005-nat record
bar. `gate_passed = true` here only reflects the near-zero quant gap and
the 15.6 MB artifact clearing the 16 MB cap; nothing about that field
implies BPB competitiveness. That said, the *directional* claim of the
hypothesis holds: seq512 delivered many more updates per wallclock and
beat the opposite seq2048 endpoint cleanly at gate. Not a promotion
candidate as-is, but the "step-bound, not token-bound" signal is worth
carrying forward into schedule / context / attention follow-ups.

## Suggested follow-ups

- **Matched seq1024 control.** Rerun the default seq1024 config on the
  same 1-GPU / 480 s screen rig with seed 1337 — only then can the
  "more updates per wallclock" claim be measured directly rather than
  through the seq2048 sibling.
- **Decouple the two knobs.** Re-run with `TRAIN_SEQ_LEN=512` but leave
  `TRAIN_BATCH_TOKENS=524288` so seq-length effect is isolated from the
  halved token-throughput per step.
- **Sweep intermediate lengths.** `TRAIN_SEQ_LEN ∈ {384, 512, 768, 1024}`
  at fixed batch tokens to locate the context/steps optimum — this run
  only probed one endpoint.
- **Retune schedule for the step count.** `warmup_steps:20` and
  `iterations:20000` are sized for a much longer run; at 2735 completed
  steps, compress warmup/warmdown to a ~2.5k-step horizon so cosine /
  warmdown phases actually complete inside the wallclock window.
- **Retune LR for short context.** `matrix_lr:0.04 / embed_lr:0.05` were
  tuned for seq1024; shorter sequences with smaller per-step batches
  may want a longer warmup and/or modestly higher LR.
- **Curriculum.** seq-512 for the early phase (cheap warmup + dense
  updates), then switch to seq-1024 or seq-2048 in the warmdown tail to
  recover long-range BPB — inverse of the `exp002` curriculum suggestion.
- **Promote-stage re-evaluation once a screen control wins.** The 1-GPU
  480 s rig scales step-avg and memory very differently from 8×H100;
  only burn promote budget once a matched screen shows a clear effect.
- **Cross-read with `exp002`.** The pair {seq512 half-batch, seq2048
  double-batch} brackets the default; seq512 wins at gate but neither
  approaches SOTA. Promote the winner for a fair head-to-head rather
  than concluding from gate-only numbers.
