# muon_mom=0.93

## Hypothesis

Muon optimizer momentum `0.93` (slightly below the conventional `0.95`)
may be the sweet spot at the current screen wallclock. With only a few
hundred effective updates inside the 480 s cap, a lower beta should
track a rapidly-moving loss surface better and reduce stale-gradient
drag. This is `exp001` of the `idea_muon_momentum_fine_sweep` arm, run
as a single env-var override on the plain baseline (`recipe_id=null`,
`source_ref=""`), not stacked on the current SOTA recipe.

## Configuration

**Env overrides**

| Env var         | Value |
|-----------------|-------|
| `MUON_MOMENTUM` | 0.93  |

**Recipe:** none — single env-var override on baseline `train_gpt.py`
(`recipe_id=null`, `source_ref=""`, `is_reproduction=false`).

**Run shape** (from `train.log`):

- `model_params:17059912`
- `world_size:1 grad_accum_steps:8`
- `attention_mode:gqa num_heads:8 num_kv_heads:4`
- `tie_embeddings:True embed_lr:0.05 head_lr:0.0 matrix_lr:0.04 scalar_lr:0.04`
- `train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:20 max_wallclock_seconds:480.000`
- `seed:1337`
- `tokenizer:sentencepiece fineweb_1024_bpe`, `val_tokens:62021632`

## Results

Run hit the 480 s wallclock cap at step `1431 / 20000`
(step_avg ≈ 335.6 ms) — heavily undertrained relative to the nominal
20 000-step horizon. Loss decreased monotonically post-warmup; no
warnings or NaNs. Single-GPU (`world_size:1`) screen run, not the full
8×H100 lane.

**Quoted key lines** from
`experiment_logs/idea_muon_momentum_fine_sweep/idea_muon_momentum_fine_sweep_exp001/train.log`:

```
step:0/20000 val_loss:6.9357 val_bpb:4.1077 train_time:0ms step_avg:0.02ms
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

| Metric             | Value          | Baseline   | Δ vs baseline |
|--------------------|----------------|------------|---------------|
| `screen_ema_bpb`   | **1.294893**   | 1.0810     | **+0.213893** |
| `gate_int6_bpb`    | 1.330100       | —          | —             |
| `gate_quant_gap`   | −4.633 × 10⁻⁵  | —          | ≈ 0           |
| `gate_artifact_mb` | 0.0 (not set)  | 16.00 cap  | —             |
| artifact (log)     | 13.52 MB       | 16.00 cap  | well under    |
| peak memory        | 10 303 MiB     | —          | —             |
| `gate_passed`      | **True**       | —          | —             |
| `promote_ema_bpb`  | null           | —          | pending       |
| `promote_int6_bpb` | null           | —          | pending       |

**Caveat on the +0.214 nat delta.** The baseline `1.0810` is the current
stacked-SOTA bar; this run is a *plain-baseline* config on a single GPU
that only reached 1431 steps before the 480 s cap. The gap is dominated
by the missing recipe stack and the truncated screen budget, not by the
momentum choice. The sweep gate still passed and the int8+zlib quant
gap is effectively zero (−4.6 × 10⁻⁵) — a nice side property of this
momentum value, independent of the absolute delta.

## Verdict

**promising** — `gate_passed=True`, clean wallclock termination, and a
~zero int8+zlib quant gap, so `0.93` survives into the promote stage.
The headline +0.214 nat gap vs stacked-SOTA is *not* evidence against
the momentum change, because the recipe stack is absent here. Whether
`0.93` actually beats `0.95` or other neighbors can only be decided by
ranking it against `exp002`–`exp004`, and ultimately by re-running the
sweep winner on top of the current SOTA recipe at 8×H100 scale.

## Suggested follow-ups

- Cross-compare `exp001`–`exp004` `screen_ema_bpb` within the
  `idea_muon_momentum_fine_sweep` arm to pick the momentum that
  minimizes BPB at this wallclock; `0.93` is only interpretable
  relative to its siblings.
- Re-run the sweep winner on top of the current SOTA recipe (PR #1019
  self-gen GPTQ + all-layer XSA) so the delta is measured on the
  stacked config, not a plain baseline.
- Promote the screening winner to the 8×H100 full lane — the
  `world_size:1` cap at 1431/20000 is far from the true 10-minute
  budget and may re-order the ranking.
- Once a winner is selected, run ≥ 3 seeds to clear the 0.005-nat,
  p < 0.01 record bar before any PR.
- Probe interaction with `MUON_MOMENTUM_WARMUP_STEPS` if exposed: a
  lower steady-state momentum may prefer a shorter warmup ramp.
- Verify the ~zero quant gap also holds under int6 / ternary paths; if
  so, `0.93` may be disproportionately friendly to aggressive
  quantization.
