# shallower-wider 7L x 640d

## Hypothesis

Fewer layers mean fewer sync points per forward pass and higher tokens/sec,
which may offset the capacity loss from dropping depth. This sweep point
pushes the shape toward 7 layers × 640 dim with 10 heads / 2 KV heads to see
whether a wider-but-shallower configuration can close the gap with the
current deeper-narrower SOTA baseline within the 10-minute budget.

## Configuration

| Env var        | Value |
|----------------|-------|
| `NUM_LAYERS`   | 7     |
| `MODEL_DIM`    | 640   |
| `NUM_HEADS`    | 10    |
| `NUM_KV_HEADS` | 2     |

- **Recipe:** None (`recipe_id` is `null` — env-only fork from default baseline)
- **Source ref:** —
- **Is reproduction:** false

Other relevant run settings (from the previously inspected `train.log`):
- `model_params:19025350` (~19.0 M)
- `attention_mode:gqa num_heads:10 num_kv_heads:2`
- `tie_embeddings:True embed_lr:0.05 head_lr:0.0 matrix_lr:0.04 scalar_lr:0.04`
- `train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:20`
- `max_wallclock_seconds:480.000`
- `world_size:1 grad_accum_steps:8` (single-GPU screening run)
- `seed:1337`

## Results

Key lines from `train.log` (captured in prior inspection; log not re-read this pass):
- `step:0/20000 val_loss:6.9469 val_bpb:4.1143 train_time:0ms`
- `step:1000/20000 val_loss:2.3194 val_bpb:1.3737 train_time:322557ms`
- `step:1489/20000 val_loss:2.2444 val_bpb:1.3293 train_time:480124ms`
- `stopping_early: wallclock_cap train_time:480124ms step:1489/20000`
- `peak memory allocated: 9537 MiB reserved: 9628 MiB`
- `Serialized model int8+zlib: 15133368 bytes (payload:19135512 raw_torch:19170649 payload_ratio:3.91x)`
- `Total submission size int8+zlib: 15181061 bytes`
- `final_int8_zlib_roundtrip_exact val_loss:2.24736472 val_bpb:1.33101597`

Baseline val_bpb for delta calc: **1.10625353**

| Metric               | Value        | Delta vs baseline (1.10625) |
|----------------------|--------------|-----------------------------|
| screen_ema_bpb       | **1.29712**  | **+0.19087 (worse)**        |
| gate_int6_bpb        | **1.33100**  | **+0.22475 (worse)**        |
| gate_quant_gap       | −1.597e−05   | — (negligible, effectively zero) |
| gate_artifact_mb     | 0.0 reported (actual int8+zlib ≈ 15.18 MB, under 16 MB cap) | — |
| gate_passed          | true         | —                           |
| wallclock            | 480.124 s (hit cap at step 1489/20000) | — |
| peak GPU mem         | 9537 MiB     | —                           |
| promote_ema_bpb      | null         | —                           |
| promote_int6_bpb     | null         | —                           |

Notes:
- Near-zero quantization gap is excellent — the 7L/640d shape quantizes
  cleanly to int6.
- Screening run was **single-GPU** (`world_size:1`) with a 480 s cap, so the
  "fewer sync points" lever from the hypothesis was not actually exercised —
  multi-GPU collectives were absent from this environment.
- Only 1489/20000 steps were completed before the wallclock cap, so the run
  is deeply undertrained relative to the 8×H100 SOTA regime that produced
  the 1.106 baseline.
- No warnings or divergence in the log; loss curve is monotone after the
  initial spike at step 2 (`train_loss:19.6042`) which recovers by step 4.
- Gate flag flipped to passed against a weaker screening threshold, not
  against the 1.106 leaderboard baseline.

## Verdict

**regression** — absolute val_bpb (EMA 1.297, int6 1.331) is ~0.19–0.22 nats
worse than the 1.10625 baseline. The tactical gate flag flipped to passed
(against the weaker screening reference), but this point in the shape sweep
is a large regression against the current SOTA. The single-GPU,
wallclock-truncated screening harness also undersells the hypothesis's core
claim (sync-point reduction), so the result does not by itself falsify
shallower-wider — it just shows that at ~19 M params and 1489 steps,
7L × 640d does not approach SOTA BPB.

## Suggested follow-ups

- **Deeper-narrower control:** Leaderboard SOTA clusters at 10–11 layers.
  Screen 10L/480d and 11L/448d at matched param count under the same gate to
  confirm depth dominance at this scale.
- **Multi-GPU rerun:** Re-run 7L × 640d on 8×H100 with the full 10-minute
  budget so the tokens/sec vs sync-point trade-off is actually tested — the
  single-GPU gate masked the hypothesis's core lever.
- **Matched-param shape grid:** Sweep {7L×576, 8L×576, 9L×512, 10L×480} at a
  fixed ~19 M params to isolate depth-vs-width from total capacity.
- **7L/640d + MLP 3× expansion:** If width is retained for throughput,
  recover capacity through wider FFN rather than more attention layers.
- **7L/640d + XSA last-4:** Cross sliding attention could partially
  compensate for reduced depth by extending effective receptive field.
- **Fix screening budget:** The 480 s single-GPU gate starves shallow
  configs of steps. Before declaring further shape regressions, match the
  effective step count of the 1.106 baseline or gate on tokens/sec-adjusted
  projected BPB.
- **Defer env-only forks:** Further shallow-wide experiments should layer on
  top of the current SOTA recipe (`sp8192 3-layer recur parresid`) rather
  than forking from the default baseline, to stay composable with the
  frontier.
