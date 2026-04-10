# MLP_MULT=4

## Hypothesis

Widening the MLP inner expansion from the 2× baseline to 4× trades a larger
parameter count (and a larger post-quant artifact) for faster per-step
convergence. The bet is that the extra FFN capacity lowers validation BPB
inside the 480 s screening budget by enough to offset slower steps and a
heavier compressed artifact — and ideally also shrinks the fp16→int
quantization gap. The idea only wins if the configuration still fits under
the 16,000,000-byte decimal artifact cap and if the schedule has time to
converge.

## Configuration

| Env override | Value |
|---|---|
| `MLP_MULT` | `4` |

- **Recipe:** none — single env-var override on the default baseline.
- **Source ref:** *(none provided)*
- **Reproduction:** no
- **Arch (log L5, L8):** GQA 8 heads / 4 KV heads, tied embeddings,
  `model_params=26,497,096` (~26.5 M).
- **Schedule (log L10, L11):** `iterations=20000 warmup_steps=20
  train_batch_tokens=524288 train_seq_len=1024
  max_wallclock_seconds=480 seed=1337`.
- **Hardware (log L6):** `world_size:1 grad_accum_steps:8`.

## Results

Key lines from
`experiment_logs/idea_mlp_expansion_sweep_2x_3x_4x/idea_mlp_expansion_sweep_2x_3x_4x_exp002/train.log`:

```
model_params:26497096
train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:20 max_wallclock_seconds:480.000
step:1000/20000 val_loss:2.2588 val_bpb:1.3378 train_time:400637ms step_avg:400.64ms
step:1199/20000 val_loss:2.2260 val_bpb:1.3184 train_time:480205ms step_avg:400.50ms
stopping_early: wallclock_cap train_time:480205ms step:1199/20000
peak memory allocated: 13017 MiB reserved: 13386 MiB
Serialized model int8+zlib: 19775260 bytes (payload:26634528 raw_torch:26679641 payload_ratio:3.94x)
Total submission size int8+zlib: 19822953 bytes
final_int8_zlib_roundtrip val_loss:2.2280 val_bpb:1.3195 eval_time:13493ms
final_int8_zlib_roundtrip_exact val_loss:2.22798977 val_bpb:1.31954104
```

Baseline for delta: **val_bpb = 1.10625353**.

| Metric | Value | Δ vs baseline |
|---|---|---|
| `screen_ema_bpb` | 1.31026301 | **+0.20401** |
| `gate_int6_bpb` (int8+zlib roundtrip) | 1.31954104 | **+0.21329** |
| `gate_quant_gap` | −4.10e-05 | ≈0 (lossless) |
| `gate_artifact_mb` (metadata) | 0.0 | not populated — see anomalies |
| Total int8+zlib artifact (log) | **19,822,953 B ≈ 19.82 MB** | **+3,822,953 B over 16 MB cap** |
| `model_params` | 26,497,096 | — |
| Steps completed | 1,199 / 20,000 | ~6 % of plan |
| `step_avg` | ~400.5 ms | ~2.1× slower than 2× baseline |
| Peak VRAM | 13,017 MiB | — |
| `gate_passed` (metadata) | `true` | misleading — see anomalies |

**Anomalies and warnings**

- **Hard-rule violation — artifact cap.** The int8+zlib total is
  19,822,953 bytes, ~3.82 MB over the 16,000,000-byte *decimal* cap. Any
  submission built from this config would be disqualified regardless of
  BPB.
- **Severely under-trained.** `stopping_early: wallclock_cap` fires at
  step 1199 / 20000, 6.0 % of the planned schedule. At ~400 ms/step the
  4× MLP is ~2× slower than the 2× baseline and cannot complete the
  schedule.
- **Loss still descending at stop.** val_bpb drops 1.3378 → 1.3184 between
  steps 1000 and 1199, so the reported final BPB reflects a starved run,
  not a 4×-MLP asymptote.
- **Quant gap essentially zero.** int8+zlib roundtrip is −4.10e-05 vs EMA
  (i.e. marginally *better*). This is the one encouraging signal but is
  measured far from convergence — treat as directional only.
- **Gate reporter bug.** `gate_passed=true` and `gate_artifact_mb=0.0`
  were recorded while the log shows a 19.82 MB artifact. The gate is
  checking quant gap but not absolute artifact size.
- **Early-step transient.** train_loss spikes to 16.65 at step 2 (log
  L34) before recovering by step 10 — absorbed by warmup, not fatal.

## Verdict

**regression**

MLP_MULT=4 on the default shape blows both competition budgets: ~26.5 M
params → 19.82 MB int8+zlib artifact (over the 16 MB decimal cap by
~3.82 MB) and ~400 ms/step → only 1,199 of 20,000 steps in 480 s,
leaving int8+zlib val_bpb at **+0.21329 nats** vs the 1.10625 baseline.
The convergence-speed hypothesis is effectively unfalsified — the run
never converged — and the artifact cap would disqualify this config as
a submission even if quality had been competitive. The near-zero
quant gap is noted but not load-bearing. Do not stack on top of the
current SOTA chain as-is.

## Suggested follow-ups

- **Shrink-then-widen.** Retry MLP_MULT=4 with reduced `DIM` and/or fewer
  layers so params land in ~6–9 M, int8+zlib artifact is ≤15.5 MB with
  headroom, and step time drops back toward ~150 ms. Only then can the
  wide-FFN convergence-speed claim be tested honestly.
- **Iso-artifact sweep.** Compare `MLP_MULT ∈ {2, 3, 4}` at matched
  *compressed* artifact size (not matched layers), so the shape tradeoff
  is measured in competition-legal terms.
- **Matched-param ablation for quant gap.** The ~0 int8 gap is the only
  interesting signal — run MLP_MULT=2 deeper vs MLP_MULT=4 shallower at
  matched params and a legal artifact to test whether wide FFN actually
  reduces the fp16→int gap.
- **Harder quantization on a shrunk wide-FFN variant.** Given the
  lossless int8 behavior, try GPTQ-lite / int5 / int4 / ternary on the
  MLP weights specifically — leaderboard has ternary at 1.1570 and 1-bit
  at 1.1239, so aggressive MLP quant is the only way a 4× FFN fits.
- **Fix the gate first.** Add an artifact-size precheck
  (`total_int8_zlib_bytes <= 16_000_000`) and backfill
  `gate_artifact_mb` so runs over the cap are never marked
  `gate_passed=true`.
- **Stacking test deferred.** Only re-evaluate against the current SOTA
  chain (PR #1019, 11L AR Self-Gen GPTQ + XSA, 1.1147) once a
  budget-compliant wide-FFN variant beats its matched-param 2× control
  on the current backbone — the 11L XSA/EMA shape changes the FLOP/param
  tradeoff significantly.
- **Kill sibling cells early** if MLP_MULT ∈ {3, 4} also exceed the cap
  at default `DIM`/depth — no reason to burn wallclock on configs that
  cannot be submitted.
