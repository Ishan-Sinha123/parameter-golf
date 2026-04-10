# MLP_MULT=3

## Hypothesis
Increasing the MLP inner expansion ratio to 3× is expected to trade a
larger parameter count (and a larger post-quant artifact) for faster
per-step convergence, hopefully netting a lower validation BPB within
the 8-minute training wallclock cap.

## Configuration
| Env var | Value |
|---|---|
| `MLP_MULT` | `3` |

- Recipe: _none_ (single env override on default baseline)
- Branch / commit: `autoresearch-deploy` @ `071de03a`
- Stage: `gate`
- Tokenizer: `fineweb_1024_bpe` (SentencePiece, vocab 1024)
- Arch: GQA 8/4 heads, tied embeddings, `model_params=21,778,504`
- Train: `iterations=20000`, `warmup_steps=20`, `train_batch_tokens=524288`,
  `seq_len=1024`, `max_wallclock_seconds=480`, `seed=1337`
- Hardware: single GPU (`world_size:1`, `grad_accum_steps:8`, GPU 2 on
  host `206.125.32.60`)

## Results

Quoted from `train.log`:

```
model_params:21778504
step:1000/20000 val_loss:2.2794 val_bpb:1.3500 train_time:382895ms
step:1254/20000 val_loss:2.2374 val_bpb:1.3251 train_time:480267ms
stopping_early: wallclock_cap train_time:480267ms step:1254/20000
Serialized model int8+zlib: 16671539 bytes (payload:21906720 raw_torch:21951833 payload_ratio:3.93x)
Total submission size int8+zlib: 16719232 bytes
final_int8_zlib_roundtrip val_loss:2.2394 val_bpb:1.32628009
```

| Metric | Value | Δ vs baseline (1.081) |
|---|---|---|
| Screen EMA val_bpb | 1.31622 | **+0.2352** |
| Gate int6 val_bpb (int8+zlib roundtrip) | 1.32628 | **+0.2453** |
| Quant gap (ema→int6) | 1.99e-05 | — |
| Total submission (int8+zlib) | **16,719,232 bytes** | **+719,232 over 16,000,000 cap** |
| Steps completed | 1254 / 20000 | early-stopped by `wallclock_cap` |
| `gate_passed` flag | `true` (metadata) | — |

Anomalies:
- **Artifact-cap violation.** Post-quant submission is 16.72 MB, over the
  16,000,000-byte decimal cap. The `gate_passed=true` flag in
  `experiment.json` appears to only reflect the negligible quant gap, not
  the absolute size constraint — this is misleading.
- **Severely under-trained.** Hit `wallclock_cap` at step 1254/20000
  (6.3% of the planned schedule). Step time ≈ 383 ms because the 3× MLP
  blew up both params (21.8 M) and per-step FLOPs.
- Training loss was still descending (val_bpb 1.3500 → 1.3251 between
  step 1000 and 1254), so the model never approached convergence.

## Verdict
**regression** — +0.245 BPB vs the 1.081 baseline, produces a
disqualifying >16 MB artifact, and starves the 480 s wallclock budget so
the model only completes ~6% of planned steps. The `gate_passed=true` in
metadata is misleading because the size cap is violated.

## Suggested follow-ups
- Return `MLP_MULT` to the baseline (2×) and instead sweep depth
  (10L / 11L), where leaderboard history shows the real BPB unlocks live.
- If 3× MLP is still desired, pair it with shrinkage elsewhere (`DIM`,
  layer count, head count) to land params well under 14 M so int6/zlib
  fits ≤16,000,000 bytes with headroom.
- Try `MLP_MULT=2` with MLP int5/ternary quant to see if width savings
  buy enough additional steps to beat the baseline.
- Add an artifact-size precheck to the gate so over-cap configs are
  killed before the full 8-minute burn.
- Fix the `gate_passed` logic to fail any run whose `Total submission
  size` exceeds 16,000,000 bytes regardless of quant gap.
