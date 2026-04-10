# muon_mom=0.97

## Hypothesis

Muon optimizer momentum of `0.97` may be optimal for the wallclock-capped
screening harness. This is one point in a fine sweep (`0.93`, `0.96`,
`0.97`, `0.98`) probing just above the default `0.95`, trading gradient
smoothing (higher momentum) against responsiveness to local curvature
(lower momentum) when total update count is constrained by the 480 s
wallclock budget.

## Configuration

| Env var | Value |
| --- | --- |
| `MUON_MOMENTUM` | `0.97` |

- **Recipe id:** `null` (single env-var override on the screening default;
  **not** stacked on the current SOTA chain)
- **Source ref:** _(none)_
- **Stage:** `gate`
- **Branch / commit:** `autoresearch-deploy` @ `1bbd7549`
- **Model params:** 17,059,912
- **Attention:** GQA, 8 heads / 4 KV heads, tied embeddings
- **Seq len / batch tokens:** 1024 / 524,288, `grad_accum_steps=8`
- **LRs:** `embed_lr=0.05`, `matrix_lr=0.04`, `scalar_lr=0.04`, `head_lr=0.0`
- **Iterations / warmup / wallclock cap:** 20,000 / 20 / 480 s
- **Seed:** 1337
- **GPUs:** 1 (`world_size=1`)

## Results

Training stopped early at step **1434 / 20000** via the 480 s wallclock
cap (~334.80 ms/step, no warnings in the log).

Key log lines:

```
model_params:17059912
attention_mode:gqa num_heads:8 num_kv_heads:4
train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:20 max_wallclock_seconds:480.000
step:1000/20000 val_loss:2.3013 val_bpb:1.3629 train_time:334849ms step_avg:334.85ms
step:1434/20000 val_loss:2.2317 val_bpb:1.3218 train_time:480102ms step_avg:334.80ms
stopping_early: wallclock_cap train_time:480102ms step:1434/20000
Serialized model int8+zlib: 14307775 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
final_int8_zlib_roundtrip val_loss:2.2333 val_bpb:1.3227 eval_time:10853ms
final_int8_zlib_roundtrip_exact val_loss:2.23331788 val_bpb:1.32269665
```

No anomalies; the early-warmup spike at step 2 (`train_loss=16.74`) is
the usual cold-Muon transient, and loss descends cleanly thereafter.

| Metric | Value | Δ vs baseline (1.10625) |
| --- | --- | --- |
| Final val_bpb (step 1434, fp16) | 1.3218 | +0.21555 |
| `screen_ema_bpb` | 1.28966815 | **+0.18341** |
| `gate_int6_bpb` (int8+zlib roundtrip) | 1.32270 | +0.21645 |
| `gate_quant_gap` | 3.35e-06 | ~0 (effectively lossless) |
| Artifact (int8+zlib payload) | 14,307,775 B (~14.31 MB) | under 16 MB cap |
| `gate_passed` | ✅ true | — |
| `promote_ema_bpb` | `null` | not promoted |
| `promote_int6_bpb` | `null` | not promoted |

**Important caveat:** the 1.10625 baseline is the current SOTA-stacked
recipe (`rec_20260410_..._sp8192_3layerrecur_parresid_q`, 10-min budget,
sp8192 tokenizer). This run uses the **default screening baseline**
(17 M params, sp1024, 480 s) with only `MUON_MOMENTUM` overridden, so the
+0.18 nats absolute gap reflects the SOTA-vs-screen structural delta,
not the effect of momentum itself. The sweep only makes sense compared
across its own cohort.

### Sweep comparison (same screening harness)

| Experiment | `MUON_MOMENTUM` | `gate_int6_bpb` | `screen_ema_bpb` | Steps | Wallclock cap |
| --- | --- | --- | --- | --- | --- |
| exp001 | 0.93 | 1.33015 | 1.29492 | 1431 | 480 s |
| exp002 | 0.96 | 1.32360 | 1.29013 | 1435 | 480 s |
| **exp003** | **0.97** | **1.32270** | **1.28967** | **1434** | **480 s** |
| exp004 | 0.98 | 1.31290 | _pending_ | 1617 | **540 s** ⚠ |

Among the three 480 s runs, momentum improves monotonically
(`0.93 → 0.96 → 0.97`), with diminishing returns:

- `0.93 → 0.96`: **−0.00655** `gate_int6_bpb`
- `0.96 → 0.97`: **−0.00090** `gate_int6_bpb`

`exp004` (0.98) posts a better `gate_int6_bpb` (1.31290) but ran at a
**540 s** cap, so it used ~183 more steps (1617 vs 1434) — the result is
not directly comparable to exp001–exp003 and cannot be used to rank
momentum in isolation.

## Verdict

**neutral.** 0.97 is the best of the three iso-wallclock points (480 s)
and the int8+zlib quant gap is effectively zero (3.35e-06 nats), so the
gate passed cleanly and the artifact sits comfortably under 16 MB. But
the marginal gain from `0.96 → 0.97` is only 0.00090 `gate_int6_bpb` —
well below seed noise at this model size and far below the 0.005-nat
record bar. Combined with the exp004 wallclock mismatch leaving the
`0.97` vs `0.98` ranking unresolved, this result alone does not support
a win call, and the run was gate-passed but not promoted.

## Suggested follow-ups

- Re-run `exp004` (0.98) at the **same 480 s cap** as exp001–exp003 to
  get a comparable point and resolve the `0.97` vs `0.98` ranking.
- Once the 480 s curve is complete, run ≥3 seeds at the sweep winner
  before treating it as a real signal — at 17 M params, seed noise is
  comparable to the 0.005-nat / p<0.01 record bar.
- Test the winning momentum **stacked on the current SOTA recipe**
  (`sp8192_3layerrecur_parresid_q`, 10-min budget) rather than the
  screening default — the 17 M / sp1024 optimum may not transfer to the
  sp8192 + 3-layer-recurrence regime.
- Because `0.96 → 0.97` only buys 0.00090 nats, weigh the cost of further
  momentum fine-tuning against higher-impact levers: `warmdown3500`,
  QK5, self-gen GPTQ calibration, or all-layer XSA.
- If the winner stabilizes above `0.97`, extend the sweep to `0.985` /
  `0.99` to find where the optimum rolls over (likely via early-training
  instability rather than late-training bias).
- Probe interaction with `MUON_WD`, Parallel Muon, and `warmdown3500`:
  recent SOTA chains changed the effective update scale, which can shift
  the momentum optimum.
- If the sweep winner differs from the current `train_gpt.py` default,
  consider bumping the default so every downstream experiment inherits
  the win.
