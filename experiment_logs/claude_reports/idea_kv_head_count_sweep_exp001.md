# kv_heads=2

## Hypothesis

Reducing KV heads from the default to **2** (GQA with aggressive KV
sharing) should trim the key/value projection parameter and activation
footprint while leaving language-modeling quality roughly intact,
striking a better memory-vs-BPB tradeoff at the ~16 MB artifact budget.
Any parameters freed could then be redeployed in follow-up experiments.

## Configuration

| Env var | Value |
|---|---|
| `NUM_KV_HEADS` | `2` |

- Recipe: default `train_gpt.py` baseline (`recipe_id = null`), no other
  overrides.
- Seed / schedule / optimizer: whatever the repo defaults were at the
  time this run was launched.
- Training log: **not captured** for this report, so the results table
  below is derived entirely from the screen/gate metrics in the
  experiment metadata.

## Results

Baseline val_bpb for delta calc: **1.10625353**

| Metric | Value | Δ vs baseline |
|---|---|---|
| screen EMA val_bpb        | 1.30301978 | **+0.19677** |
| gate int6 val_bpb         | 1.31600000 | **+0.20975** |
| fp → int6 quant gap       | −2.66e-05  | ≈ 0 (essentially lossless) |
| gate artifact size (MB)   | 0.0        | (not reported / no artifact size captured) |
| gate passed               | `true`     | quant-gap + artifact check only — not a BPB win |

Both the screen EMA BPB and the gated int6 BPB sit ≈0.197–0.210 nats
**above** the 1.10625 baseline. The fp→int6 quant gap is essentially
zero (in fact slightly negative at −2.66e-05), so the int6 pathway is
not what is costing BPB here — the model itself is training to a worse
solution than the baseline at this seed / budget.

## Verdict

**regression**

`NUM_KV_HEADS=2` on top of the default baseline comes in nearly 0.2
nats BPB worse than the reference 1.10625 baseline, so this run is a
clear regression and does not threaten SOTA. The gate "passes" only in
the sense that quantization is lossless and the artifact fits — the ML
signal is unambiguously negative. Whether that regression is inherent
to 2 KV heads or is confounded by schedule/undertraining cannot be
disentangled from this single run in isolation (see follow-ups).

## Suggested follow-ups

- Run a matched-budget sweep `NUM_KV_HEADS ∈ {1, 2, 4, 8}` on the same
  recipe, same seed, same wallclock, so the KV-head axis is isolated
  rather than confounded with schedule length or LR.
- Cross-check against `idea_kv_head_count_sweep_exp002` (the paired run
  in this sweep) before drawing sweep-level conclusions — one point is
  not a curve.
- Re-test `NUM_KV_HEADS=2` composed on top of the **current SOTA
  recipe** (with XSA / GPTQ / LeakyReLU² / TTT) across ≥3 seeds, since
  composability at the frontier is what drives records; a loss on
  default baseline doesn't rule out a gain after stacking.
- If 2 KV heads continues to regress at matched compute, try
  `NUM_KV_HEADS=1` (MQA) as the extreme of the sweep to put a lower
  bound on sensitivity, and redirect any freed parameters into MLP
  expansion or added depth.
- Inspect the training log when it is next captured: confirm whether
  the run actually consumed its full wallclock budget or was halted
  early (e.g. `wallclock_cap`), which is the usual source of silent
  undertraining in this harness.
