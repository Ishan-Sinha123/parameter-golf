# muon_mom=0.98

## Hypothesis

Momentum 0.98 may be optimal for the Muon optimizer at this screening
wallclock. This is the fourth leg in a fine sweep (`exp001`–`exp004`)
walking Muon momentum upward; the run tests whether pushing momentum to
0.98 continues to improve short-run optimization on the baseline
screening recipe or overshoots and starts to hurt. The intuition is
that a high-momentum average smooths gradient noise across the small
number of effective updates reachable in a 540 s single-GPU screen,
but momentum this close to 1 can also lag curvature changes and stall
progress late in the short run.

## Configuration

| Env var         | Value  |
|-----------------|--------|
| `MUON_MOMENTUM` | `0.98` |

- **Recipe id:** `null` — single env-override on the screening default;
  not stacked on the current SOTA chain.
- **Source ref / reproduction:** none · no.
- **Stage reached:** `gate` (passed, **not** promoted).

Relevant config pulled from
`experiment_logs/idea_muon_momentum_fine_sweep/idea_muon_momentum_fine_sweep_exp004/train.log`:

- `model_params`: 17,059,912 (~17.06 M)
- `attention_mode`: `gqa` 8 heads / 4 kv heads, `tie_embeddings=True`
- `train_batch_tokens`: 524,288 · `train_seq_len`: 1024 · `iterations`: 20,000
- `warmup_steps`: 20 · `max_wallclock_seconds`: **540**
- `embed_lr`: 0.05 · `matrix_lr`: 0.04 · `scalar_lr`: 0.04 · `head_lr`: 0.0
- `world_size`: 1 · `grad_accum_steps`: 8 · `seed`: 1337
- Data: `fineweb10B_sp1024`, 10 train shards, val tokens 62,021,632

## Results

Training stopped cleanly on the 540 s wallclock cap at step
**1617 / 20000** (~334 ms/step). The usual cold-Muon transient shows
up at step 2 (`train_loss=16.74`) and descends cleanly thereafter. No
NaNs, no divergence, no stability warnings. Peak memory allocated
**10,303 MiB** / reserved **10,622 MiB**.

Key log lines:

```
model_params:17059912
attention_mode:gqa num_heads:8 num_kv_heads:4
train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:20 max_wallclock_seconds:540.000
step:0/20000    val_loss:6.9357 val_bpb:4.1077
step:1000/20000 val_loss:2.3082 val_bpb:1.3671 train_time:333929ms step_avg:333.93ms
step:1617/20000 val_loss:2.2151 val_bpb:1.3119 train_time:540065ms step_avg:333.99ms
stopping_early: wallclock_cap train_time:540065ms step:1617/20000
peak memory allocated: 10303 MiB reserved: 10622 MiB
Serialized model int8+zlib: 15259302 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
final_int8_zlib_roundtrip val_loss:2.2168 val_bpb:1.3129 eval_time:10853ms
final_int8_zlib_roundtrip_exact val_loss:2.21677284 val_bpb:1.31289774
```

| Metric                                 | Value                    | Δ vs baseline (1.0810) |
|----------------------------------------|--------------------------|------------------------|
| val_bpb @ step 1000 (fp)               | 1.36710                  | +0.28610               |
| val_bpb @ step 1617 (fp, final)        | 1.31190                  | +0.23090               |
| `screen_ema_bpb`                       | **1.29011505**           | **+0.20912**           |
| `gate_int6_bpb` (int8+zlib roundtrip)  | 1.31290                  | +0.23190               |
| `gate_quant_gap`                       | ~2.26e-6 (≈ 0)           | effectively lossless   |
| Artifact (int8+zlib payload)           | 15,259,302 B (~15.26 MB) | under the 16 MB cap    |
| Peak GPU memory                        | 10,303 MiB allocated     | —                      |
| `gate_passed`                          | true                     | —                      |
| `promote_ema_bpb` / `promote_int6_bpb` | `null` / `null`          | not promoted           |

**Apples-to-apples caveat.** The 1.0810 reference is a full-budget
multi-GPU SOTA-stacked number. This run is a 17 M-param, `sp1024`,
single-GPU, 540 s screen with only `MUON_MOMENTUM` overridden, so the
+0.21 nat absolute EMA gap is almost entirely the SOTA-vs-screen
structural delta, **not** a regression of the momentum knob. The real
signal is horizontal against siblings `exp001`–`exp003` under the
matched gate.

## Verdict

**neutral.** The gate passed, the int8+zlib quantization gap is
effectively zero (~2.26e-6 nats), the artifact is comfortably under the
16 MB cap at ~15.26 MB, and training was stable. But no promote metrics
were recorded (`promote_ema_bpb: null`), so there is no full-budget
signal that 0.98 is the sweep winner, and the run cannot be judged
against the 1.0810 SOTA baseline in isolation. Win / regression verdict
awaits the within-sweep cohort comparison. Note also that exp003
(`0.97`) ran under a 480 s cap and exp004 (`0.98`) under 540 s, so any
direct exp003↔exp004 comparison is contaminated by the wallclock
mismatch.

## Suggested follow-ups

- Aggregate `screen_ema_bpb` across `idea_muon_momentum_fine_sweep_exp001–004`
  under a **matched wallclock cap** (pick one: 480 s or 540 s) and pick
  the cohort leader; only promote that one.
- Multi-seed (≥3) the top 1–2 momentum values at the same screening
  wallclock to measure the seed-level noise floor and confirm whether
  differences exceed the 5 × 10⁻³ nat / p < 0.01 PR bar.
- Probe one step further (`MUON_MOMENTUM=0.985` / `0.99`) only if 0.98
  sits at the top edge of the matched cohort — otherwise stop
  fine-tuning momentum and redirect effort to higher-impact levers
  (all-layer XSA, self-gen GPTQ calibration, LeakyReLU², 11L depth,
  1-bit / ternary quant from maintainers' open requests).
- Try a short momentum warmup (e.g. `0.85 → 0.98` over the first
  ~100 steps) — late-stage high momentum sometimes beats a fixed high
  value on very short runs.
- Couple momentum with a `MUON_WD` sweep — momentum and weight decay
  interact, and a higher momentum typically prefers a slightly
  different effective WD.
- Port the sweep winner onto the current SOTA chain (11L + Self-Gen
  GPTQ + all-layer XSA) rather than the screening default; the sp1024 /
  1-GPU optimum may not transfer to the full 8×H100 / 10-minute target.
