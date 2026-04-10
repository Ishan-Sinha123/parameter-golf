# kv_heads=2

## Hypothesis

Reducing KV heads to 2 (GQA 8→2) should cut KV cache memory and the
parameters allocated to key/value projections while leaving
language-modeling quality roughly intact, striking a better memory/BPB
tradeoff than the default head configuration.

## Configuration

| Env var | Value |
|---|---|
| `NUM_KV_HEADS` | `2` |

- Recipe: default `train_gpt.py` baseline (no `recipe_id` override).
- Attention mode (from log): `gqa num_heads:8 num_kv_heads:2`.
- Model params: 15,880,264.
- Optim: `embed_lr=0.05 head_lr=0.0 matrix_lr=0.04 scalar_lr=0.04`, tied embeddings.
- Train: `train_batch_tokens=524288 seq_len=1024 iterations=20000 warmup=20`,
  `max_wallclock_seconds=540`.
- Seed: `1337`.

Key log excerpts:

```
attention_mode:gqa num_heads:8 num_kv_heads:2
model_params:15880264
step:1000/20000 val_loss:2.3143 val_bpb:1.3706 train_time:322337ms
step:1680/20000 val_loss:2.2201 val_bpb:1.3149 train_time:540126ms
stopping_early: wallclock_cap train_time:540126ms step:1680/20000
Serialized model int8+zlib: 14560584 bytes (payload_ratio:3.64x)
final_int8_zlib_roundtrip val_loss:2.2221 val_bpb:1.3160 eval_time:10387ms
final_int8_zlib_roundtrip_exact val_loss:2.22205572 val_bpb:1.31602656
```

## Results

| Metric | Value | Δ vs baseline (1.0810) |
|---|---|---|
| screen EMA val_bpb        | 1.30302   | **+0.22202** |
| gate int6 val_bpb         | 1.31600   | **+0.23500** |
| fp → quant gap            | −2.66e-05 | ~0 (essentially lossless) |
| artifact size (int8+zlib) | ~14.61 MB | under 16 MB cap |
| quant/gate passed         | true      | (gate is a quant-gap check, not a BPB win) |
| peak GPU memory           | 9,814 MiB | — |
| training halted at        | step 1680 / 20000 | `wallclock_cap` (540 s) |

Run was terminated at the 540 s wallclock cap having completed only
1680 / 20000 iterations (step_avg ≈ 321 ms ⇒ ~8.4% of the planned
schedule). The val_bpb curve was still descending steeply at cutoff
(1.3706 at step 1000 → 1.3149 at step 1680).

## Verdict

**regression**

Both the screen EMA BPB (1.3030) and the gated int6 BPB (1.3160) are
≈0.22–0.23 nats above the 1.0810 SOTA baseline, so this configuration
does not threaten SOTA. The quant gate does pass, but that only reflects
a near-zero fp→int8+zlib round-trip gap, not an ML win. The dominant
signal is that the run wallclock-capped at 1680 / 20000 steps, so the
BPB measured is heavily confounded by undertraining rather than cleanly
attributable to the KV-head reduction. The hypothesis cannot be
confirmed or falsified from this run in isolation.

## Suggested follow-ups

- Re-run the sweep with matched training budgets: pair `NUM_KV_HEADS=2`
  against `NUM_KV_HEADS ∈ {1, 4, 8}` on the same recipe, same seed,
  same wallclock, so the comparison isolates the KV-head dimension
  instead of confounding it with schedule length.
- Lower `iterations` (e.g. 2000–3000) so warmdown actually fires inside
  the 540 s budget; `iterations=20000` leaves the LR schedule in the
  flat-warmup plateau for essentially the whole run.
- Shrink per-step cost (smaller batch or shorter seq_len) so more steps
  fit into the wallclock cap and the KV-head variable can be measured
  on a trained, not undertrained, model.
- Compose `NUM_KV_HEADS=2` on top of the current 1.0810 SOTA recipe
  rather than the default baseline — 3× seeds — to check composability.
  Tight KV cache could free headroom for TTT or longer context at eval.
- Try `NUM_KV_HEADS=1` (MQA) as the extreme of this sweep under the
  matched-budget protocol; if 2 regresses cleanly at matched compute,
  MQA sets an informative lower bound on KV-count sensitivity.
