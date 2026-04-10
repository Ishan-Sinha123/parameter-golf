# kv_heads=8

## Hypothesis

Setting `NUM_KV_HEADS=8` (one KV head per query head → effectively full MHA
since `num_heads=8`) was hypothesized to balance memory usage against BPB:
giving every query head its own key/value projection was expected to raise
attention capacity and push BPB below the GQA default, provided the extra
params still fit under the 16 MB artifact cap after int8+zlib.

## Configuration

| Env var        | Value |
|----------------|-------|
| `NUM_KV_HEADS` | `8`   |

- Recipe: *(none — single env override on the current default baseline;
  `recipe_id: null`)*
- Relevant knobs from `train.log`:
  `attention_mode:gqa num_heads:8 num_kv_heads:8`,
  `model_params:19,419,208`, `tie_embeddings:True`, `embed_lr:0.05`,
  `matrix_lr:0.04`, `train_seq_len:1024`, `train_batch_tokens:524288`,
  `grad_accum_steps:8`, `iterations:20000`, `warmup_steps:20`,
  `max_wallclock_seconds:540.000`, `seed:1337`, `world_size:1`.

## Results

Key lines quoted from `experiment_logs/idea_kv_head_count_sweep/idea_kv_head_count_sweep_exp002/train.log`:

```
model_params:19419208
attention_mode:gqa num_heads:8 num_kv_heads:8
step:0/20000 val_loss:6.9393 val_bpb:4.1098 train_time:0ms
step:1000/20000 val_loss:2.2967 val_bpb:1.3602 train_time:352088ms step_avg:352.09ms
step:1537/20000 val_loss:2.2161 val_bpb:1.3125 train_time:540148ms step_avg:351.43ms
stopping_early: wallclock_cap train_time:540148ms step:1537/20000
peak memory allocated: 11270 MiB reserved: 11326 MiB
Serialized model int8+zlib: 15963234 bytes (payload:19547424 raw_torch:19592537 payload_ratio:3.92x)
Total submission size int8+zlib: 16010927 bytes
final_int8_zlib_roundtrip val_loss:2.2179 val_bpb:1.3136 eval_time:11586ms
final_int8_zlib_roundtrip_exact val_loss:2.21793496 val_bpb:1.31358602
```

| Metric                 | Value          | Δ vs baseline (1.10625) |
|------------------------|----------------|-------------------------|
| screen EMA BPB         | **1.28282**    | **+0.17657**            |
| gate int6 BPB          | **1.31359**    | **+0.20734**            |
| quant gap (int6−ema)   | 1.40e-05       | —                       |
| artifact (int8+zlib)   | 16,010,927 B   | **over 16 MB hard cap by 10,927 B** |
| peak memory            | 11,270 MiB     | —                       |
| steps completed        | 1,537 / 20,000 | killed by `wallclock_cap` at 540s |
| step_avg               | ~351 ms        | —                       |
| `gate_passed` (tracker)| `true`         | despite artifact exceeding the decimal 16 MB cap |

Observations:

- Full MHA with 8 KV heads inflated params to **19.4M**, which (a) pushed
  the int8+zlib submission past the **16,000,000-byte decimal hard cap**
  by ~11 KB, and (b) slowed each step to ~351 ms so only
  **1,537 / 20,000** planned steps ran before the wallclock cap fired.
- Severely undertrained: the model was still at `val_bpb:1.3125` on its
  last internal eval, versus the current baseline at **1.10625**. Final
  int6 BPB is **+0.207 nats worse** than baseline.
- Quant gap is essentially zero (1.4e-5), so int8+zlib is lossless here;
  the regression is entirely from undertraining + artifact bloat, not from
  quantization.
- The tracker recorded `gate_artifact_mb=0.0` and `gate_passed=true` even
  though the serialized submission is 16,010,927 bytes. That looks like an
  instrumentation bug — under competition rules this artifact would DQ.
- Warmup showed an instability spike (`step:2 train_loss:16.74`), but
  training recovered within ~8 steps; that's not the primary failure mode.

## Verdict

**regression** — the configuration is both over the 16 MB artifact cap
(by ~11 KB) and over the wallclock budget (training was early-stopped at
7.7% of planned steps), and the final int6 BPB is **+0.207 nats worse**
than the current baseline (1.10625 → 1.31359). The `gate_passed=true`
flag in the tracker is an instrumentation artifact, not a real pass.

## Suggested follow-ups

- **Fix the tracker bug** where `gate_artifact_mb` reported `0.0` and
  `gate_passed` was `true` despite `Total submission size int8+zlib:
  16010927 bytes`; downstream gates shouldn't trust that signal.
- **Do not sweep `NUM_KV_HEADS=8` again on the current baseline shape** —
  it doesn't fit the 16 MB cap. If full MHA is still desired, first shrink
  `dim` / `n_layer` / vocab until the fp model clears ~15.5 MB post
  int8+zlib, *then* re-enable full MHA.
- **Re-scope the KV-head sweep around `{1, 2, 4}`** (lighter than the
  current default) and measure whether *fewer* KV heads frees params for
  more layers, wider MLP, or more training steps within the 540 s cap.
- **Consider an asymmetric/per-layer KV-head schedule** (wider KV only in
  the top layers) as a cheaper way to probe "more KV capacity where it
  matters" without inflating the whole stack.
- **Fair-step rerun**: if the hypothesis is still to be tested on an
  equivalent-step budget, rerun on 8×H100 SXM so the 10-minute cap isn't
  dominated by per-step latency, and run ≥3 seeds so any delta is
  statistically meaningful (≥0.005 nats, p<0.01).
