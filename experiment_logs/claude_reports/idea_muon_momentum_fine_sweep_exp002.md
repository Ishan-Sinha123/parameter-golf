# muon_mom=0.96

## Hypothesis

Muon optimizer momentum of 0.96 may be optimal at the current screening wallclock budget. This run is part of a fine sweep (0.93 / 0.96 / 0.97 / 0.98) probing just above the default 0.95 — moderate momentum balances gradient history retention against responsiveness, so when total steps are capped by wallclock a small bump can meaningfully change the effective trajectory.

## Configuration

| Env var | Value |
| --- | --- |
| `MUON_MOMENTUM` | `0.96` |

- **Recipe id:** `null` (single env-var override on the screening default)
- **Source ref:** _(none)_
- **Model params:** 17,059,912
- **Attention:** GQA, 8 heads / 4 KV heads, tied embeddings
- **Seq len / batch tokens:** 1024 / 524,288, grad_accum 8
- **LRs:** embed 0.05, matrix 0.04, scalar 0.04, head 0.0
- **Warmup / wallclock cap:** 20 steps / 480 s
- **Seed:** 1337

## Results

Training stopped early at step 1435 / 20000 via the 480 s wallclock cap (~334.5 ms/step). Baseline for Δ is the current SOTA `val_bpb=1.0810`.

Key log lines:

- `step:1000/20000 val_loss:2.3015 val_bpb:1.3631 train_time:334387ms`
- `step:1435/20000 val_loss:2.2331 val_bpb:1.3226 train_time:480064ms`
- `stopping_early: wallclock_cap train_time:480064ms step:1435/20000`
- `Serialized model int8+zlib: 14017080 bytes (payload_ratio:3.91x)`
- `final_int8_zlib_roundtrip val_loss:2.2348 val_bpb:1.3236`
- `final_int8_zlib_roundtrip_exact val_bpb:1.32359505`

| Metric | Value | Δ vs baseline (1.0810) |
| --- | --- | --- |
| screen_ema_bpb | 1.29013 | +0.20913 |
| gate_int6_bpb | 1.32360 | +0.24260 |
| gate_quant_gap | 4.95e-06 | — |
| gate_artifact_mb | ~13.4 | under 16 MB cap |
| gate_passed | ✅ | — |
| promote_ema_bpb | — | not promoted |
| promote_int6_bpb | — | not promoted |

Sweep context (comparable exp001–exp003 all ran ~335 ms/step under the same 480 s cap):

| Exp | MUON_MOMENTUM | screen_ema_bpb | gate_int6_bpb |
| --- | --- | --- | --- |
| exp001 | 0.93 | 1.2949 | 1.3301 |
| **exp002** | **0.96** | **1.2901** | **1.3236** |
| exp003 | 0.97 | 1.2897 | 1.3227 |
| exp004 | 0.98 | (different hw, not directly comparable) | — |

Within the comparable subset, 0.93 → 0.96 → 0.97 is a monotonically improving trend: 0.93 → 0.96 drops int6_bpb by −0.0065 (large), 0.96 → 0.97 drops it a further −0.0009 (diminishing). No warnings in the log; loss curve decreased steadily.

## Verdict

**neutral.** 0.96 passes the screening gate with a near-zero quant gap and a clean int8+zlib roundtrip, but the screen EMA BPB of 1.2901 is +0.209 nats above the SOTA 1.0810 baseline and it is outperformed within its own sweep by 0.97. This run's value is confirming the monotonic sweep trend, not an independent win.

## Suggested follow-ups

- Re-run exp004 (0.98) on the same GPU class as exp001–003 to finish the apples-to-apples comparison and pin the optimum at 0.97 vs 0.98.
- If 0.97 wins the sweep, run 3 seeds at that value for the ≥0.005-nat / p<0.01 significance bar before promoting.
- Stack the winning momentum on top of the current SOTA recipe (`rec_20260410_2026_04_09_sp8192_3layerrecur_parresid_q`, 1.0810) rather than the screening default, since the screening baseline is far off-SOTA.
- Sanity-check interaction with Parallel Muon and warmdown3500, which changed the effective update scale in recent SOTA chains and may shift the momentum optimum.
- If the default Muon momentum in `train_gpt.py` is still 0.95, consider bumping it to the sweep winner to benefit every downstream experiment.
