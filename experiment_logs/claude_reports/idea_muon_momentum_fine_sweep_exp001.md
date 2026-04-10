# muon_mom=0.93

## Hypothesis

Momentum 0.93 may be optimal at this wallclock. Fine-sweep point for
`MUON_MOMENTUM=0.93` probing whether a small reduction from the default
yields better train/EMA BPB under the 480 s wallclock cap. This is a
standalone sweep on the plain baseline config (`recipe_id=null`,
`source_ref=""`), not stacked on top of the current SOTA recipe.

## Configuration

| Env var          | Value |
|------------------|-------|
| `MUON_MOMENTUM`  | 0.93  |

Recipe: none (single env-var override on baseline train_gpt.py). Sweep
context: `exp001` of the `idea_muon_momentum_fine_sweep` arm.

Observed setup from `train.log`:

- `model_params:17059912`
- `world_size:1 grad_accum_steps:8`
- `attention_mode:gqa num_heads:8 num_kv_heads:4`
- `tie_embeddings:True embed_lr:0.05 head_lr:0.0 matrix_lr:0.04 scalar_lr:0.04`
- `train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:20 max_wallclock_seconds:480.000`
- `seed:1337`

## Results

Run hit the 480 s wallclock cap at step 1431/20000 (severely undertrained
relative to the nominal 20 000-step horizon). Step average ~335.6 ms.

Key quoted lines:

```
step:1000/20000 val_loss:2.3084 val_bpb:1.3672 train_time:335580ms
step:1400/20000 train_loss:2.2927 train_time:469839ms step_avg:335.60ms
step:1431/20000 val_loss:2.2438 val_bpb:1.3289 train_time:480184ms step_avg:335.56ms
stopping_early: wallclock_cap train_time:480184ms step:1431/20000
Serialized model int8+zlib: 13471458 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
final_int8_zlib_roundtrip val_loss:2.2459 val_bpb:1.3301 eval_time:10934ms
final_int8_zlib_roundtrip_exact val_loss:2.24589635 val_bpb:1.33014633
peak memory allocated: 10303 MiB reserved: 10622 MiB
```

| Metric                | Value       | Δ vs baseline (1.0810) |
|-----------------------|-------------|------------------------|
| screen EMA BPB        | **1.29489** | **+0.21389**           |
| int8+zlib gate BPB    | 1.33010     | +0.24910               |
| gate quant gap        | −0.0000463  | — (negligible)         |
| Artifact size         | ~13.47 MB   | under 16 MB cap        |
| Peak memory           | 10 303 MiB  | —                      |
| Gate passed           | True        | —                      |

No warnings or anomalies in the log. Loss decreased monotonically after
warmup. Note: the supplied baseline `1.0810` is the current stacked SOTA
(sp8192 + 3-layer recurrence); this sweep point runs the plain baseline
config, so the headline Δ is dominated by the absent recipe stack, not
by the momentum change itself. The internal sweep gate still passed.

## Verdict

**regression** — Screen EMA BPB 1.29489 is +0.214 worse than the SOTA
baseline (1.0810). The result about `MUON_MOMENTUM=0.93` itself is only
meaningful when compared to sibling sweep arms (`exp002`–`exp004`) at
the same baseline config, not against the stacked SOTA. The nearly-zero
quant gap (−4.6 × 10⁻⁵) is a nice property and the artifact is well
under the 16 MB cap.

## Suggested follow-ups

- Cross-compare `exp001`–`exp004` screen EMA BPBs within the
  `idea_muon_momentum_fine_sweep` arm to pick the momentum that actually
  minimizes BPB at this wallclock; 0.93 becomes meaningful only relative
  to the other points.
- Re-run the winning momentum on top of the current SOTA recipe
  (`rec_20260410_2026_04_09_sp8192_3layerrecur_parresid_q_3710821c`) —
  a baseline-config win does not automatically transfer to the stacked
  recipe.
- The run burned the full 480 s on only 1431/20000 steps at
  `world_size:1`; confirm whether this sweep is intended to run
  single-GPU or should be on 8×H100 so the sweep ordering reflects the
  real 10-minute budget.
- Once the winner is selected, run ≥3 seeds to satisfy the 0.005-nat,
  p<0.01 significance bar before promoting.
- Probe the interaction with `MUON_MOMENTUM_WARMUP_STEPS` — a lower
  steady-state momentum may prefer a shorter warmup ramp.
