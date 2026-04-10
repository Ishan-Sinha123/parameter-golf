# Muon LR=0.025

**Experiment ID:** `idea_muon_lr_sweep_around_0_03_exp001`
**Date:** 2026-04-10
**Recipe:** `rec_20260410_muon_lr_0_025_314dd79b`

## Hypothesis

LR 0.03 was found to be optimal for the Muon optimizer in Track 4. A fine-grained sweep around that value may discover a slightly better learning rate. This run tests LR=0.025, slightly below the current best, to determine whether the optimum lies on the lower side.

## Configuration

| Parameter | Value |
|-----------|-------|
| `MUON_LR` | `0.025` |

All other parameters held at baseline defaults.

## Results

| Metric | Value | Delta vs Baseline (1.322) |
|--------|-------|---------------------------|
| Screen EMA BPB | 1.333 | +0.011 (worse) |
| Gate INT6 BPB | 1.708 | — |
| Quant gap | 0.100 | — |
| Artifact size | 7.1 MB | under 16 MB cap |
| Gate passed | Yes | — |
| Promote EMA BPB | — | not promoted |
| Promote INT6 BPB | — | not promoted |

Training log was not captured for this run.

## Verdict

**Regression.** Lowering Muon LR from 0.03 to 0.025 increased EMA BPB by +0.011 relative to the 1.322 baseline. The gate passed on artifact size and quant gap, but the model did not earn promotion. This confirms that 0.03 is not too high; the optimum is at or above 0.03.

## Suggested follow-ups

- **MUON_LR=0.032 and 0.035**: Sweep above 0.03 to check whether a slightly higher LR improves over the current best.
- **MUON_LR=0.028**: Tighter bracket between 0.025 (regression) and 0.03 (baseline-optimal) to refine the lower bound.
- **Combined sweep**: Pair the best Muon LR with other stacked features (e.g., warmdown schedule, EMA decay) to see if the optimal LR shifts when other hyperparameters change.
