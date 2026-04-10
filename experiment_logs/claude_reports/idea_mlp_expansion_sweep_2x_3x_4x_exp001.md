# MLP_MULT=3

## Hypothesis

Increasing the MLP expansion factor from the default 2x to 3x trades a larger artifact size for faster per-step convergence. The wider MLP hidden dimension gives each layer more representational capacity, potentially reaching lower BPB in fewer training steps — important given the tight wallclock budget.

## Configuration

| Parameter | Value |
|-----------|-------|
| `MLP_MULT` | `3` (default: `2`) |
| Model params | 21,778,504 |
| GPUs | 1 × H100 (screening) |
| Wallclock cap | 480 s |
| Steps completed | 1,254 / 20,000 |
| Seed | 1337 |
| Sequence length | 1024 |
| Batch tokens | 524,288 |
| Step avg | ~383 ms |

Recipe: none (single env-var override on default baseline).

## Results

| Metric | MLP_MULT=3 (this) | MLP_MULT=4 (exp002) | Default (MLP_MULT=2) |
|--------|-------------------|---------------------|----------------------|
| Model params | 21.8M | 26.5M | ~17M (est.) |
| Steps completed | 1,254 | 1,199 | — |
| val_bpb (last checkpoint) | 1.3251 | 1.3184 | — |
| int8+zlib val_bpb | **1.3263** | 1.3195 | — |
| Quant gap | 0.0012 (negligible) | 0.0011 | — |
| int8+zlib artifact | **16.72 MB** | 19.82 MB | — |
| Artifact under 16 MB? | **No** | No | — |

**Key log lines:**

```
model_params:21778504
step:1254/20000 val_loss:2.2374 val_bpb:1.3251  (wallclock cap)
Serialized model int8+zlib: 16671539 bytes
final_int8_zlib_roundtrip_exact val_loss:2.23936838 val_bpb:1.32628009
```

**Observations:**

- Training hit the 480 s wallclock cap at step 1,254, completing only ~6% of planned iterations.
- The int8+zlib artifact is 16,671,539 bytes of model + 47,693 bytes of code = **16,719,232 bytes total**, which **exceeds the 16,000,000-byte hard cap** by ~719 KB.
- Even with the artifact size issue, the quant gap is only ~0.001 BPB — quantization is not the bottleneck.
- Compared to the MLP_MULT=4 sibling (exp002), this run is slightly worse in BPB (1.3263 vs 1.3195) but much closer to the artifact cap.
- Without a matched MLP_MULT=2 screening run, we cannot compute a direct delta. However, the competition baseline (9L/512d/MLP_2x) achieves 1.2244 on 8×H100 — this single-GPU screening BPB of 1.326 is expected to be higher since it used ~1/8 the compute and ~6% of planned steps.

## Verdict

**Neutral.** The screening run completed without errors and the quant gap is minimal, but the int8+zlib artifact exceeds the 16 MB hard cap by ~4.5%. Without aggressive quantization (int6, GPTQ) or parameter reduction, MLP_MULT=3 cannot fit a valid submission. The run provides useful signal for the sweep but is not directly promotable.

## Suggested follow-ups

- **Run the MLP_MULT=2 baseline** on a single GPU with the same 480 s cap to get a matched comparison for the sweep.
- **Try int6 or GPTQ quantization** on the MLP_MULT=3 checkpoint to see if the artifact can be squeezed under 16 MB while preserving BPB.
- **Combine MLP_MULT=3 with fewer layers** (e.g., 8L instead of 9L) to offset the parameter increase and fit under the cap.
- **Test MLP_MULT=2.5** (non-integer, if supported) as a midpoint that may stay under 16 MB with int8+zlib.
- **Full 8×H100 run** of MLP_MULT=3 only if artifact size can be solved first — no point scaling compute on an over-budget artifact.
