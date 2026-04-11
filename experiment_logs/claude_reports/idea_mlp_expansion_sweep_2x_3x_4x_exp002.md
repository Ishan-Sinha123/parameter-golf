# MLP_MULT=4

## Hypothesis

MLP expansion 4× trades artifact size for convergence speed. Widening the FFN
inner dim to 4× over the default should give each step more representational
capacity and lower val_bpb faster inside the 480 s screening budget, at the
cost of a larger compressed artifact and slower steps. The bet only wins if
the run still fits under the 16,000,000-byte decimal artifact cap and reaches
a competitive BPB before the wallclock early-stop fires.

## Configuration

| Env override | Value |
|---|---|
| `MLP_MULT` | `4` |

- **Recipe:** none — single env-var override on the default baseline.
- **Source ref:** *(none provided)*
- **Reproduction:** no
- **Arch (log L5, L8):** GQA 8 heads / 4 KV heads, tied embeddings,
  `model_params = 26,497,096` (~26.5 M).
- **Schedule (log L10):** `iterations=20000 warmup_steps=20
  train_batch_tokens=524288 train_seq_len=1024
  max_wallclock_seconds=480 seed=1337`.
- **Hardware (log L6):** `world_size:1 grad_accum_steps:8`.

## Results

Key lines from
`experiment_logs/idea_mlp_expansion_sweep_2x_3x_4x/idea_mlp_expansion_sweep_2x_3x_4x_exp002/train.log`:

```
model_params:26497096
train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:20 max_wallclock_seconds:480.000
step:0/20000 val_loss:6.9357 val_bpb:4.1077
step:1000/20000 val_loss:2.2588 val_bpb:1.3378 train_time:400637ms step_avg:400.64ms
step:1199/20000 val_loss:2.2260 val_bpb:1.3184 train_time:480205ms step_avg:400.50ms
stopping_early: wallclock_cap train_time:480205ms step:1199/20000
peak memory allocated: 13017 MiB reserved: 13386 MiB
Serialized model int8+zlib: 19775260 bytes (payload:26634528 raw_torch:26679641 payload_ratio:3.94x)
Total submission size int8+zlib: 19822953 bytes
final_int8_zlib_roundtrip val_loss:2.2280 val_bpb:1.3195 eval_time:13493ms
final_int8_zlib_roundtrip_exact val_loss:2.22798977 val_bpb:1.31954104
```

Baseline for delta: **val_bpb = 1.081**.

| Metric | Value | Δ vs baseline (1.081) |
|---|---|---|
| `screen_ema_bpb` | 1.31026301 | **+0.22926** |
| `gate_int6_bpb` (int8+zlib roundtrip) | 1.31954104 | **+0.23854** |
| `gate_quant_gap` | −4.10e-05 | ≈0 (lossless) |
| `gate_artifact_mb` (metadata) | 0.0 | not populated — see anomalies |
| Total int8+zlib artifact (log) | **19,822,953 B ≈ 19.82 MB** | **+3,822,953 B over 16 MB cap** |
| `model_params` | 26,497,096 | — |
| Steps completed | 1,199 / 20,000 | ~6 % of plan |
| `step_avg` | ~400.5 ms | ~2× slower than 2× baseline |
| Peak VRAM | 13,017 MiB | — |
| `gate_passed` (metadata) | `true` | misleading — see anomalies |

**Anomalies and warnings**

- **Hard-rule violation — artifact cap.** int8+zlib total is 19,822,953 B,
  ~3.82 MB over the 16,000,000-byte *decimal* cap. A submission from this
  config would be disqualified regardless of BPB.
- **Severely under-trained.** `stopping_early: wallclock_cap` fires at step
  1199 / 20000 (~6 % of plan). At ~400 ms/step the 4× MLP cannot complete
  the schedule.
- **Loss still descending at stop.** val_bpb drops 1.3378 → 1.3184 between
  steps 1000 and 1199, so the reported final is a starved-run number, not
  a 4×-MLP asymptote.
- **Quant gap essentially zero.** int8+zlib roundtrip is −4.10e-05 vs EMA
  (marginally *better*). Encouraging but measured far from convergence.
- **Gate reporter bug.** `gate_passed=true` with `gate_artifact_mb=0.0`
  while the log shows a 19.82 MB artifact — the gate checks the quant gap
  but not absolute artifact size.
- **Early-step transient.** train_loss spikes to 16.65 at step 2 before
  recovering by step 10 — absorbed by warmup, not fatal.

## Verdict

**regression**

MLP_MULT=4 on the default shape blows both competition budgets: ~26.5 M
params → 19.82 MB int8+zlib artifact (over the 16 MB decimal cap by
~3.82 MB) and ~400 ms/step → only 1,199 / 20,000 steps in 480 s, leaving
int8+zlib val_bpb at **+0.23854 nats** vs the 1.081 baseline. The
convergence-speed hypothesis is effectively unfalsified — the run never
converged — and the artifact cap alone would disqualify this config. The
near-zero quant gap is noted but not load-bearing. Do not stack on top of
the current SOTA chain as-is.

## Suggested follow-ups

- **Shrink-then-widen.** Retry `MLP_MULT=4` with reduced `DIM` and/or fewer
  layers so params land in ~6–9 M and int8+zlib artifact is ≤15.5 MB with
  headroom. Only then can the wide-FFN convergence-speed claim be tested.
- **Iso-artifact sweep.** Compare `MLP_MULT ∈ {2, 3, 4}` at matched
  *compressed* artifact size, so the shape tradeoff is measured in
  competition-legal terms.
- **Harder quantization on a wide-FFN variant.** Given the ~lossless int8
  behavior, try GPTQ-lite / int5 / int4 / ternary on the MLP weights
  specifically — leaderboard has ternary at 1.1570 and 1-bit at 1.1239,
  so aggressive MLP quant is the only path that lets a 4× FFN fit.
- **Matched-param quant-gap ablation.** MLP_MULT=2 deeper vs MLP_MULT=4
  shallower at matched params and a legal artifact, to test whether wide
  FFN really shrinks the fp16→int gap.
- **Fix the gate first.** Add an artifact-size precheck
  (`total_int8_zlib_bytes <= 16_000_000`) and backfill `gate_artifact_mb`
  so over-cap runs can never be marked `gate_passed=true`.
- **Kill sibling cells early** if `MLP_MULT ∈ {3, 4}` also exceed the cap
  at default `DIM`/depth — no reason to burn wallclock on configs that
  cannot be submitted.
- **Defer stacking vs SOTA.** Only re-evaluate against the current chain
  (PR #1019, 11L AR Self-Gen GPTQ + XSA, 1.1147) once a budget-compliant
  wide-FFN variant beats its matched-param 2× control on the current
  backbone — 11L XSA/EMA changes the FLOP/param tradeoff enough that the
  2x/3x/4x comparison must be repeated on that shape before layering on.
