# Muon LR=0.030

**Experiment ID:** `idea_muon_lr_sweep_around_0_03_exp003`
**Recipe:** `rec_20260410_muon_lr_0_030_f758e39c`
**Date:** 2026-04-10

## Hypothesis

LR 0.03 was previously identified as optimal for the Muon optimizer in Track 4. A fine-grained sweep around this value may reveal a slightly better learning rate that improves convergence and final BPB.

## Configuration

| Parameter | Value |
|-----------|-------|
| `MUON_LR` | `0.030` |

All other parameters held at baseline defaults.

## Results

| Metric | Value | Delta vs Baseline (1.322) |
|--------|-------|---------------------------|
| EMA BPB (screen) | 1.327 | **+0.005** (worse) |
| Int6 BPB (gate) | 1.702 | — |
| Quant gap | 0.120 | — |
| Artifact size | 7.1 MB | under 16 MB cap |
| Gate passed | Yes | — |
| Promoted | No | — |

**Training log:** Not captured.

The EMA BPB of 1.327 is 0.005 nats worse than the 1.322 baseline, indicating that MUON_LR=0.030 does not improve over the default. The int6 quantized BPB of 1.702 is notably poor, with a quant gap of 0.120, suggesting the learned weights at this LR are harder to quantize cleanly.

## Verdict

**Regression.** MUON_LR=0.030 yields a small but clear regression in EMA BPB (+0.005) relative to baseline and exhibits a large quantization gap (0.12). Not worth pursuing further at this exact value.

## Suggested follow-ups

- **Narrow the sweep below 0.030:** If 0.030 is worse and a prior point (e.g. 0.025) was better, try 0.026–0.029 in finer increments.
- **Investigate the quant gap:** The 0.12 gap is unusually large — check whether MUON_LR interacts with weight magnitude distributions that hurt int6 GPTQ calibration.
- **Combine LR sweep with warmdown schedule:** The learning rate at end-of-training matters for quantization quality; pairing LR tuning with warmdown3500 may close the quant gap.
- **Try LR=0.032–0.035 range:** If the optimum is above 0.030, a sweep in the other direction would confirm whether the landscape is monotonic or has a second basin.
