# MLP_MULT=3

## Hypothesis

Widening the MLP inner expansion to 3× should trade a larger parameter count (and therefore a larger post-quant artifact) for faster per-step convergence, with the bet that a wider MLP nets a lower validation BPB inside the 480 s training wallclock. The idea only wins if the extra capacity still fits under the 16,000,000-byte decimal artifact cap after int8/zlib compression *and* if the slower per-step time does not starve the schedule.

## Configuration

| Env var | Value |
|---|---|
| `MLP_MULT` | `3` |

- **Recipe:** none — single env-override on the default baseline (`recipe_id = null`).
- **Source ref:** *(not set)*
- **Reproduction:** no
- **Arch (from log):** GQA `num_heads=8 num_kv_heads=4`, tied embeddings, SentencePiece vocab 1024, `model_params=21,778,504`.
- **Schedule (from log):** `iterations=20000 warmup_steps=20 train_batch_tokens=524288 train_seq_len=1024 max_wallclock_seconds=480 seed=1337`.
- **Hardware (from log):** `world_size:1 grad_accum_steps:8` (single-GPU screen, not the full 8×H100 budget).

## Results

Key lines from `idea_mlp_expansion_sweep_2x_3x_4x/idea_mlp_expansion_sweep_2x_3x_4x_exp001/train.log`:

```
model_params:21778504
step:1000/20000 val_loss:2.2794 val_bpb:1.3500 train_time:382895ms
step:1200/20000 train_loss:2.2339 train_time:459558ms step_avg:382.96ms
step:1254/20000 val_loss:2.2374 val_bpb:1.3251 train_time:480267ms
stopping_early: wallclock_cap train_time:480267ms step:1254/20000
Serialized model int8+zlib: 16671539 bytes (payload:21906720 raw_torch:21951833 payload_ratio:3.93x)
Total submission size int8+zlib: 16719232 bytes
final_int8_zlib_roundtrip val_loss:2.2394 val_bpb:1.3263 eval_time:11859ms
final_int8_zlib_roundtrip_exact val_loss:2.23936838 val_bpb:1.32628009
```

| Metric | Value | Δ vs baseline (1.0810) |
|---|---|---|
| `screen_ema_bpb` | 1.31622 | **+0.23522** |
| `gate_int6_bpb` (int8+zlib roundtrip) | 1.32628 | **+0.24528** |
| Quant gap (ema → int6) | 1.99e-05 | — (effectively lossless) |
| **Total int8+zlib artifact** | **16,719,232 B** | **+719,232 over 16,000,000 cap** |
| `model_params` | 21,778,504 | — |
| Steps completed | 1254 / 20000 | early-stopped by `wallclock_cap` |
| `step_avg` | ~383 ms | — |
| `gate_passed` (metadata) | `true` | misleading — see anomalies |
| `gate_artifact_mb` (metadata) | 0.0 | not populated |

Anomalies and warnings:

- **Hard-rule violation — artifact cap.** Post-int8+zlib total is 16,719,232 bytes, exceeding the 16,000,000-byte *decimal* cap by ~719 KB. Any submission built from this config would be disqualified. The metadata `gate_passed=true` appears to reflect only the negligible quant gap, not the absolute size, and `gate_artifact_mb=0.0` is clearly a reporting gap rather than a true zero-byte artifact.
- **Severely under-trained.** `stopping_early: wallclock_cap` fires at step 1254/20000 — 6.3% of the planned schedule. At `step_avg ≈ 383 ms` the 3× MLP spends more wallclock per step than the 2× baseline, so the schedule cannot complete within 480 s on one GPU.
- **Loss still descending at stop.** val_bpb dropped 1.3500 → 1.3251 between steps 1000 and 1254, so the model was nowhere near convergence — the final BPB is not a fair estimate of the 3×-MLP asymptote, just of a starved run.
- **Early-step instability.** Train loss spiked to 17.03 at step 2 before recovering by step 10; not catastrophic (warmup absorbs it) but worth noting for any follow-up at this width.

## Verdict

**broken**

Not a fair regression — this run violates a hard competition rule (16 MB decimal artifact cap) and is simultaneously starved for wallclock (6% of planned steps). The +0.245 BPB gap vs the 1.0810 baseline is partly genuine and partly an artifact of under-training. The config cannot be submitted as-is regardless of how training continues, so the correct call is "broken config, needs shrinking before the idea can be evaluated fairly" rather than "3× MLP loses to 2× MLP on the merits".

The `gate_passed=true` flag in metadata should be treated as a gate instrumentation bug: the gate is checking quant gap but not the absolute artifact size.

## Suggested follow-ups

- **Fix the gate first.** Add an artifact-size precheck (`total_bytes <= 16_000_000`) so any over-cap run is flagged and not promoted, regardless of quant gap. Also backfill `gate_artifact_mb` — it is currently always 0.0.
- **Shrink to fit, then re-test.** Re-run `MLP_MULT=3` with a reduced `DIM` and/or fewer layers so `model_params` drops enough that the int8+zlib artifact lands ≤15.5 MB with headroom. Only then is the 3× vs 2× comparison meaningful.
- **Couple to stronger quant.** Try `MLP_MULT=3` with int5 / int4 / ternary MLP weights (leaderboard has ternary at 1.1570 and 1-bit at 1.1239); the 3× width may fit if the MLP params are quantized harder than the rest.
- **Sweep ratios at matched param count.** Once size is fixed, run `MLP_MULT ∈ {2, 2.5, 3}` at matched `model_params` (compensating with `DIM` or depth) to isolate the effect of *shape* from the effect of *total capacity*.
- **Composability check.** If a shrunk `MLP_MULT=3` variant ever beats its matched-param 2× control, stack it onto the current SOTA chain (`rec_20260411_..._sp8192_parallelresid_scorefir_6ad17e09`, val_bpb=1.0822) before claiming a win — the tuned backbone changes the FLOP/param tradeoff significantly.
- **Kill sibling sweep cells early** if `MLP_MULT=4` (exp002) also exceeds the cap at default `DIM`/depth — no point burning the remaining budget on configurations that cannot be submitted.
