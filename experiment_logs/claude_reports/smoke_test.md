# smoke

## Hypothesis

Smoke test — a minimal run intended to verify that the experiment harness, logging, and gate-metric plumbing work end-to-end. No ML-level hypothesis is under test; the goal is simply to confirm that screen, gate, and promote metrics propagate through the pipeline without error.

## Configuration

| Env var | Value |
|---|---|
| _(none)_ | _(baseline defaults)_ |

- **Recipe:** _none_ (`recipe_id` is `null` — pure baseline configuration)
- **Source ref:** _none_
- **Is reproduction:** false

## Results

Training log at `experiment_logs/02_scaled_ablations_4xH100/p1_baseline.log` was **not found on disk**, so no `train_bpb`, `ema_bpb`, gate transitions, or warnings could be quoted. Only the metadata-reported metrics are available below.

| Metric | Value | Δ vs baseline (1.2000) |
|---|---|---|
| screen_ema_bpb | 1.1800 | −0.0200 |
| gate_int6_bpb | 1.2200 | +0.0200 |
| gate_quant_gap | 0.0400 | — |
| gate_artifact_mb | 15.90 | — (under 16.00 MB decimal cap) |
| gate_passed | _null_ | — |
| promote_ema_bpb | _null_ | — |
| promote_int6_bpb | _null_ | — |

Notes:
- `gate_passed` is null — the gate stage was not evaluated/recorded.
- Quant gap of 0.04 nats is wider than the <0.02 target typical of recent SOTA entries.
- Artifact size (15.9 MB) fits the 16.0 MB decimal cap with ~0.1 MB headroom.

## Verdict

**neutral** — Smoke test with no ML hypothesis. Screen EMA beats the 1.2 baseline by 0.02 nats, but the int6 gate regresses by 0.02 nats and the training log is missing, so this cannot be treated as a record-class signal. Harness-level smoke is the only takeaway.

## Suggested follow-ups

- Re-run with the training log persisted to the declared path so `train_bpb` / `ema_bpb` / gate transitions can be inspected.
- Investigate the 0.04-nat fp16→int6 gap — candidates: GPTQ-lite calibration, self-gen calibration data (PR #1019 style), or mixed int5/int6.
- Populate `gate_passed` and promote metrics so downstream composition tooling can trigger on smoke runs.
- Once harness is verified, swap the 1.2000 screening baseline for the current SOTA chain (SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT, 1.0810) as the reference.
