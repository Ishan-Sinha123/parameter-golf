# MLP_MULT=3

## Hypothesis

Increasing the MLP expansion factor from the default 2x to 3x trades a larger artifact size for faster per-step convergence. The wider MLP hidden dimension gives each layer more representational capacity, potentially reaching lower BPB in fewer training steps — important given the tight 10-minute wallclock budget.

## Configuration

| Parameter | Value |
|-----------|-------|
| `MLP_MULT` | `3` (default: `2`) |
| Model params | ~21.8M |
| Layers | 9 (default) |
| Dim | 512 (default) |
| Vocab | 1024 (default) |
| KV heads | 4 (default) |
| GPUs | 1 × H100 (screening) |
| Wallclock cap | 480 s |
| Steps completed | 1,254 / 20,000 |
| Step avg | ~383 ms |
| Seed | 1337 |

Recipe: none (single env-var override on default baseline).

## Results

| Metric | Value |
|--------|-------|
| Screen EMA val_bpb | **1.3162** |
| Gate int6 val_bpb | **1.3263** |
| Gate quant gap | ~0.00002 (negligible) |
| Gate artifact MB | 0.0 (not captured) |
| Gate passed | Yes |
| Promoted | No |
| Delta vs baseline | unknown (no matched baseline) |

Comparison with sibling run (exp002, MLP_MULT=4):

| Metric | MLP_MULT=3 (this) | MLP_MULT=4 (exp002) |
|--------|-------------------|---------------------|
| Screen EMA val_bpb | 1.3162 | 1.3103 |
| Gate int6 val_bpb | 1.3263 | 1.3195 |
| Steps completed | 1,254 | 1,199 |
| Model params | ~21.8M | ~26.5M |
| int8+zlib artifact | ~16.7 MB | ~19.8 MB |

**Key observations:**

- Training hit the 480 s wallclock cap at step 1,254, completing only ~6% of planned iterations.
- The gate quant gap is essentially zero (~0.00002 BPB), confirming quantization is not a bottleneck for this config.
- The gate passed, meaning the int6 quantized model roundtrips cleanly.
- The int8+zlib artifact was ~16.7 MB in earlier measurement, which exceeds the 16 MB hard cap by ~4.5%. However, the gate used int6 quantization which may compress further.
- MLP_MULT=4 (exp002) achieves ~0.007 better BPB but at significantly larger artifact size.
- Without a matched MLP_MULT=2 screening run at the same wallclock budget, a direct delta cannot be computed. The competition baseline (9L/512d/MLP_2x on 8×H100) achieves 1.2244 — this single-GPU screening BPB of 1.316 is expected to be higher due to ~1/8 compute and ~6% of planned steps.

## Verdict

**Neutral.** The screening run completed cleanly and the gate passed with negligible quantization gap. However, the artifact size remains borderline at int8 compression, and without a matched MLP_MULT=2 baseline we cannot quantify the benefit of 3x expansion. The run was not promoted to a full 8×H100 evaluation. MLP_MULT=3 is a viable configuration but needs artifact size resolution (int6/GPTQ) and/or parameter reduction (fewer layers, smaller dim) to be competition-eligible.

## Suggested follow-ups

- **Run matched MLP_MULT=2 baseline** on same GPU/wallclock config to quantify the delta from 2x to 3x expansion.
- **Try int6 or GPTQ quantization** on MLP_MULT=3 checkpoint to verify the artifact fits under 16 MB.
- **Combine MLP_MULT=3 with fewer layers** (e.g., 8L) to offset the parameter increase and fit under the artifact cap.
- **Test MLP_MULT=3 with reduced dim** (e.g., 448) as an alternative approach to fit under 16 MB while retaining wider MLP.
- **Full 8×H100 run** of MLP_MULT=3 only after confirming artifact budget compliance.
- **Multi-seed runs (n>=3)** of the best MLP expansion factor to establish statistical significance before promoting.
