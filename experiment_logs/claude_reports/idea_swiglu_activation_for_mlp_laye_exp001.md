# SwiGLU baseline comparison

**Experiment ID:** `idea_swiglu_activation_for_mlp_laye_exp001`
**Recipe:** `rec_20260410_swiglu_baseline_comparison_043bbb7d`
**Date:** 2026-04-10

## Hypothesis

SwiGLU activation with an adjusted MLP expansion ratio (2.67x instead of the default) should match or beat the GELU baseline in BPB. The gated linear unit structure of SwiGLU provides smoother gradients and better feature gating, and the reduced MLP ratio compensates for the extra gate projection to keep parameter count comparable.

## Configuration

| Parameter    | Value     |
|-------------|-----------|
| `ACTIVATION` | `swiglu`  |
| `MLP_MULT`   | `2.67`    |

All other hyperparameters at baseline defaults (9 layers, 512 dim, 1024 vocab, tied embeddings, 4 KV heads).

## Results

| Metric              | Value   | Delta vs baseline (1.322) |
|---------------------|---------|---------------------------|
| EMA BPB (float)     | 1.318   | **-0.004**                |
| Int6 BPB (quantized)| 1.695   | +0.373                    |
| Quant gap           | 0.098   |                           |
| Artifact size       | 7.2 MB  | under 16 MB cap           |
| Gate passed         | yes     |                           |

**Training log:** Not captured for this run. Historical SwiGLU logs in `experiment_logs/03_fullscale_8xH100_600s/p1_arch_swiglu.log` show similar architectural runs converging normally during warmup.

### Key observations

- The float EMA model shows a marginal 0.004 BPB improvement over baseline, directionally positive but well below the 0.005 record significance threshold.
- The int6 quantized model regresses severely to 1.695 BPB (+0.373 over baseline). This indicates SwiGLU's gate projection weights do not quantize well under the current int6 scheme.
- The reported quant gap metric (0.098) does not reconcile with the raw EMA-to-int6 delta (0.377), suggesting the gate metric may use a different reference or normalization.
- Artifact size of 7.2 MB leaves substantial headroom under the 16 MB cap.

## Verdict

**Promising.** SwiGLU matches baseline in float precision with a small directional win (-0.004 BPB). However, the quantization gap is prohibitively large for competition submission. The activation itself is viable if paired with quantization-aware training or a more SwiGLU-friendly quantization strategy (e.g., mixed-precision gate weights or GPTQ calibration tuned for gated activations).

## Suggested follow-ups

- **SwiGLU + GPTQ calibration:** Use self-generated calibration data specifically tuned for the SwiGLU gate projection weights to reduce the int6 quantization gap.
- **SwiGLU + late QAT ramp:** Increase QAT scale or extend QAT warmup to help the gate projection learn quantization-robust weight distributions during training.
- **Mixed-precision quantization:** Keep gate projection weights at int8 while quantizing other layers at int6 to preserve gating fidelity.
- **MLP_MULT sweep:** Try 2.5x and 3.0x expansion ratios — the 2.67x choice may not be optimal for the current parameter budget after quantization.
- **SwiGLU + deeper architectures:** Test SwiGLU with 11L configs where the activation benefit may compound across more layers, matching the current SOTA depth.
- **LeakyReLU-squared comparison:** The leaderboard shows LeakyReLU^2 at 1.1194 BPB — direct A/B with SwiGLU on the same base config would clarify which activation variant quantizes better.
