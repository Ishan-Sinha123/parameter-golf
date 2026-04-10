# Muon LR=0.028

**Experiment ID:** `idea_muon_lr_sweep_around_0_03_exp002`  
**Date:** 2026-04-10  
**Recipe:** `rec_20260410_muon_lr_0_028_210ab48b`

## Hypothesis

LR 0.03 was identified as optimal for Muon in Track 4. A fine-grained sweep around that value may discover a slightly better learning rate. This run tests LR=0.028 — slightly below the known optimum — to check whether the loss landscape favors a marginally lower step size.

## Configuration

| Parameter | Value |
|-----------|-------|
| `MUON_LR` | `0.028` |

All other settings are baseline defaults. No other env overrides.

## Results

| Metric | Value | Delta vs Baseline (1.322) |
|--------|-------|---------------------------|
| Screen EMA BPB | 1.330 | **+0.008** (worse) |
| Gate Int6 BPB | 1.705 | — |
| Gate Quant Gap | 0.11 | — |
| Gate Artifact Size | 7.1 MB | — |
| Gate Passed | Yes | — |
| Promote EMA BPB | — | (not promoted) |
| Promote Int6 BPB | — | (not promoted) |

**Note:** No raw training log was captured for this run. The gate passed the artifact and quant-gap thresholds, but the EMA BPB regressed by +0.008 relative to the 1.322 baseline (presumably at LR=0.03), so the run was not promoted.

The int6 quantized BPB of 1.705 shows a very large quantization degradation (~0.375 above EMA), far exceeding the reported gate_quant_gap of 0.11. This suggests the int6 evaluation may have encountered issues, or the quant gap metric is computed on a different basis.

## Verdict

**Regression.** Lowering Muon LR from 0.03 to 0.028 worsened BPB by +0.008. The baseline LR=0.03 remains the better setting. This data point suggests the optimum is at or above 0.03, not below it.

## Suggested follow-ups

- **LR=0.032 sweep:** Since 0.028 regressed, probe slightly above 0.03 (e.g., 0.031, 0.032, 0.033) to see if the optimum is on the high side.
- **LR=0.029 data point:** A single intermediate point between 0.028 and 0.03 would confirm whether the loss curve is steep or shallow in this region.
- **Investigate int6 quantization anomaly:** The 0.375 gap between EMA and int6 BPB is unusually large — worth diagnosing whether this is a quantization calibration issue or an inherent sensitivity of this LR setting.
- **Combined sweep with warmdown:** LR sensitivity may interact with warmdown schedule length; a joint sweep of MUON_LR × warmdown steps could find a better combination.
