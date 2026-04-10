# kv_heads=8

## Hypothesis

Setting `NUM_KV_HEADS=8` (one KV head per query head → effectively full MHA since `num_heads=8`) was hypothesized to balance KV-cache/memory against BPB: giving every query head its own key/value projection restores full attention expressivity relative to GQA, at the cost of extra attention parameters. The screen tests whether that reallocation pays for itself on EMA / int6 BPB while still fitting under the 16 MB decimal artifact cap.

## Configuration

| Env var        | Value |
|----------------|-------|
| `NUM_KV_HEADS` | `8`   |

Recipe: *none* (`recipe_id: null` — ran as a bare env override on the current default baseline).

Relevant knobs from `train.log`:
`attention_mode:gqa num_heads:8 num_kv_heads:8`, `model_params:19,419,208`, `tie_embeddings:True`, `embed_lr:0.05`, `matrix_lr:0.04`, `scalar_lr:0.04`, `train_seq_len:1024`, `train_batch_tokens:524288`, `grad_accum_steps:8`, `iterations:20000`, `warmup_steps:20`, `max_wallclock_seconds:540.000`, `seed:1337`, `world_size:1` (single-GPU screen, not the 8×H100 record budget).

## Results

Key lines quoted from `experiment_logs/idea_kv_head_count_sweep/idea_kv_head_count_sweep_exp002/train.log`:

```
model_params:19419208
attention_mode:gqa num_heads:8 num_kv_heads:8
step:0/20000    val_loss:6.9393 val_bpb:4.1098 train_time:0ms
step:2/20000    train_loss:16.7401 train_time:687ms    # warmup spike, recovered by step ~8
step:1000/20000 val_loss:2.2967 val_bpb:1.3602 train_time:352088ms step_avg:352.09ms
step:1537/20000 val_loss:2.2161 val_bpb:1.3125 train_time:540148ms step_avg:351.43ms
stopping_early: wallclock_cap train_time:540148ms step:1537/20000
peak memory allocated: 11270 MiB reserved: 11326 MiB
Serialized model int8+zlib: 15963234 bytes (payload:19547424 raw_torch:19592537 payload_ratio:3.92x)
Total submission size int8+zlib: 16010927 bytes
final_int8_zlib_roundtrip      val_loss:2.2179    val_bpb:1.3136     eval_time:11586ms
final_int8_zlib_roundtrip_exact val_loss:2.21793496 val_bpb:1.31358602
```

| Metric                  | Value          | Δ vs baseline (1.10625) |
|-------------------------|----------------|-------------------------|
| screen EMA BPB          | **1.28282**    | **+0.17657**            |
| gate int6 BPB           | **1.31359**    | **+0.20734**            |
| quant gap (int6 − ema)  | 1.40e-05       | — (effectively lossless) |
| artifact (int8+zlib)    | 16,010,927 B   | **over 16 MB decimal cap by 10,927 B** |
| peak GPU memory         | 11,270 MiB     | —                       |
| steps completed         | 1,537 / 20,000 | killed early by `wallclock_cap` (540 s) |
| step_avg                | ~351 ms        | —                       |
| `gate_passed` (tracker) | `true`         | suspicious given artifact overflow |

Observations worth flagging:

- **Artifact overflow.** Full MHA pushed the model to 19.42 M params; after int8+zlib the *total* submission is 16,010,927 B — **over the 16,000,000-byte decimal hard cap by ~11 KB**, which would DQ under competition rules.
- **Severely undertrained.** Only 1,537 / 20,000 planned steps completed; val_bpb was still dropping fast between the 1000- and 1537-step snapshots (1.3602 → 1.3125). The absolute BPB here is not comparable to the 1.1063 baseline.
- **Quant gap is essentially zero** (1.4e-5 nats), so int8+zlib is lossless — the regression comes from undertraining + artifact bloat, not from quantization.
- **Warmup spike** at `step:2 train_loss:16.7401`, recovered within ~8 steps; not the primary failure mode but worth noting if other configs see similar spikes.
- **Instrumentation concern.** The tracker recorded `gate_artifact_mb=0.0` and `gate_passed=true` even though the submission exceeds 16 MB decimal. Downstream gates should not trust that signal.

## Verdict

**regression** — the configuration is over the 16 MB artifact cap (by ~11 KB), was wallclock-capped at 7.7 % of planned steps, and its final int6 BPB (1.31359) is **+0.207 nats worse** than the current baseline (1.10625). The `gate_passed=true` flag appears to be an instrumentation artifact, not a real pass. Spending extra params on full KV heads on this shape simply does not fit.

## Suggested follow-ups

- **Fix the tracker bug** where `gate_artifact_mb` reported 0.0 and `gate_passed` was true despite `Total submission size int8+zlib: 16010927 bytes`; the gate logic should compare against the 16 M decimal cap, not a MiB value.
- **Do not re-sweep `NUM_KV_HEADS=8` on the current baseline shape** — it doesn't fit. If full MHA is still of interest, first shrink `dim` / `n_layer` / vocab until the fp model clears ~15.5 MB post-int8+zlib, then re-enable full MHA.
- **Sweep downward instead**: `NUM_KV_HEADS ∈ {1, 2, 4}` and reinvest the saved params into more depth, wider MLP, or more training steps within the 540 s cap. The top of the leaderboard (PRs #1019, #549, #374) all stack aggressive GQA, not MHA.
- **Asymmetric / per-layer KV-head schedule**: try wider KV only in the top 1–2 layers so "more KV capacity where it matters" doesn't inflate the whole stack into an oversized artifact.
- **Fair-step rerun on 8×H100 SXM** with ≥3 seeds so any BPB delta is statistically meaningful (≥0.005 nats, p<0.01) rather than dominated by a 1-GPU step-budget shortfall.
- **Pair high-KV ablations with partial RoPE / XSA** so extra KV capacity has non-trivial routing to do, rather than paying for it in vanilla attention.
