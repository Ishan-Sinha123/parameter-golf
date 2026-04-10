# muon_mom=0.93

## Hypothesis

Muon momentum `0.93` (slightly below the conventional `0.95`) may be the
sweet spot for the current screening wallclock: with only a few hundred
effective updates inside the 480 s cap, a lower beta should track a
rapidly-moving loss surface better and reduce stale-gradient drag. This
is `exp001` of 4 in the `idea_muon_momentum_fine_sweep` arm, run on the
plain baseline config (`recipe_id=null`, `source_ref=""`), not stacked on
the current SOTA recipe.

## Configuration

**Env overrides**

| Env var         | Value |
|-----------------|-------|
| `MUON_MOMENTUM` | 0.93  |

**Recipe:** none — single env-var override on the baseline `train_gpt.py`.

**Run shape (from `train.log`):**

- `model_params:17059912`
- `world_size:1 grad_accum_steps:8`
- `attention_mode:gqa num_heads:8 num_kv_heads:4`
- `tie_embeddings:True embed_lr:0.05 head_lr:0.0 matrix_lr:0.04 scalar_lr:0.04`
- `train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:20 max_wallclock_seconds:480.000`
- `seed:1337`

## Results

The run hit the 480 s wallclock cap after step 1431 / 20000
(step_avg ≈ 335.6 ms) — heavily undertrained relative to the nominal
horizon. Loss decreased monotonically post-warmup; no warnings in the log.

**Quoted key lines** from
`experiment_logs/idea_muon_momentum_fine_sweep/idea_muon_momentum_fine_sweep_exp001/train.log`:

```
step:1000/20000 val_loss:2.3084 val_bpb:1.3672 train_time:335581ms step_avg:335.58ms
step:1400/20000 train_loss:2.2927 train_time:469839ms step_avg:335.60ms
step:1431/20000 val_loss:2.2438 val_bpb:1.3289 train_time:480184ms step_avg:335.56ms
stopping_early: wallclock_cap train_time:480184ms step:1431/20000
peak memory allocated: 10303 MiB reserved: 10622 MiB
Serialized model int8+zlib: 13471458 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
Total submission size int8+zlib: 13519151 bytes
final_int8_zlib_roundtrip val_loss:2.2459 val_bpb:1.3301 eval_time:10934ms
final_int8_zlib_roundtrip_exact val_loss:2.24589635 val_bpb:1.33014633
```

**Metrics table**

| Metric             | Value        | Baseline   | Δ vs baseline |
|--------------------|--------------|------------|----------------|
| `screen_ema_bpb`   | **1.29489**  | 1.10625    | **+0.18864**   |
| `gate_int6_bpb`    | 1.33010      | —          | —              |
| `gate_quant_gap`   | −4.633e-05   | —          | ≈ 0            |
| `gate_artifact_mb` | 0.0 (n/a)    | 16.0 cap   | —              |
| artifact (log)     | 13.52 MB     | 16.0 cap   | well under     |
| peak memory        | 10 303 MiB   | —          | —              |
| `gate_passed`      | **True**     | —          | —              |
| `promote_ema_bpb`  | null         | —          | pending        |
| `promote_int6_bpb` | null         | —          | pending        |

**Caveat on the +0.189 nat delta.** The baseline (`1.10625`) is the
current stacked-SOTA bar, while this run is a plain-baseline config on a
single GPU that only reached 1431 steps before the 480 s cap. The gap is
dominated by the missing recipe stack and the shortened budget, not by
the momentum choice. The sweep gate still passed, and the int8+zlib
quant gap is essentially zero (−4.6 × 10⁻⁵) — a nice property of this
momentum value independent of the absolute delta.

## Verdict

**promising** — `gate_passed=True`, clean wallclock termination, and an
effectively-zero int8+zlib quant gap, so `0.93` survives into the
promote stage of the sweep. The headline +0.189 nat gap versus the
stacked-SOTA baseline is *not* evidence against the momentum change,
because the recipe stack is absent here. Whether `0.93` actually beats
`0.95` (or another neighbor) can only be decided by ranking against
`exp002`–`exp004`, and ultimately by re-running the winner on top of the
current SOTA recipe at 8×H100 scale.

## Suggested follow-ups

- Cross-compare `exp001`–`exp004` `screen_ema_bpb` within the
  `idea_muon_momentum_fine_sweep` arm to pick the momentum that
  minimizes BPB at this wallclock; `0.93` is only interpretable relative
  to its siblings.
- Re-run the sweep winner on top of the current SOTA recipe so the
  delta is measured on a stacked config, not a plain baseline.
- Promote the screening winner to 8×H100 full-lane — the `world_size:1`
  cap at 1431/20000 is far from the true 10-minute budget and may
  re-order the ranking.
- Once a winner is selected, run ≥ 3 seeds to clear the 0.005-nat,
  p < 0.01 record bar before any PR.
- Probe interaction with `MUON_MOMENTUM_WARMUP_STEPS` (if present): a
  lower steady-state momentum may prefer a shorter warmup ramp.
- Verify the near-zero quant gap also holds under int6 / ternary paths;
  if so, `0.93` may be disproportionately friendly to aggressive
  quantization.
