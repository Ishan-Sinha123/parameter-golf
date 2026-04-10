# MLP_MULT=4

## Hypothesis

MLP expansion factor 4x trades artifact size for faster convergence and lower per-step loss. A wider MLP should increase model capacity and improve BPB, but the larger parameter count may push the int8+zlib artifact past the 16 MB competition limit and reduce total training steps within the wallclock budget.

## Configuration

| Parameter | Value |
|-----------|-------|
| `MLP_MULT` | `4` (default: `2`) |
| Model params | 26,497,096 (~26.5M) |
| Layers | 9 (default) |
| Dim | 512 (default) |
| Vocab | 1024 (default) |
| KV heads | 4 (default) |
| Seed | 1337 |
| Wallclock cap | 480 s |
| GPU | 1 × H100 (gate screening) |
| Commit | `165884039f` |
| Recipe | None (single env-var override on default baseline) |

Companion run: `exp001` with `MLP_MULT=3` on the same commit and wallclock budget.

## Results

| Metric | MLP_MULT=4 (this run) | MLP_MULT=3 (exp001) |
|--------|----------------------|---------------------|
| EMA val_bpb (screen) | **1.3103** | 1.3162 |
| Final val_bpb (fp32) | 1.3184 | 1.3251 |
| int8+zlib val_bpb | **1.3195** | 1.3263 |
| Quant gap (gate) | ~-0.00004 (noise) | ~0.00002 (noise) |
| Steps completed | 1,199 | 1,254 |
| Step avg | ~401 ms | ~383 ms |
| Peak VRAM | 13,017 MiB | 11,416 MiB |
| Model params | 26.5M | 21.8M |
| int8+zlib artifact | **19.8 MB** | **16.7 MB** |
| Delta vs baseline | unknown | unknown |

**Key log lines (exp002):**
```
model_params:26497096
step:1000/20000 val_loss:2.2588 val_bpb:1.3378
step:1199/20000 val_loss:2.2260 val_bpb:1.3184
stopping_early: wallclock_cap train_time:480205ms step:1199/20000
peak memory allocated: 13017 MiB reserved: 13386 MiB
Serialized model int8+zlib: 19775260 bytes (payload_ratio:3.94x)
Total submission size int8+zlib: 19822953 bytes
final_int8_zlib_roundtrip val_loss:2.2280 val_bpb:1.3195
```

No warnings, divergence, or anomalies. Training loss decreased monotonically through logged checkpoints.

**Delta vs MLP_MULT=3:** int8+zlib BPB improves by **-0.0068** (1.3195 vs 1.3263). However, artifact size grows from 16.7 MB to 19.8 MB — both exceeding the 16,000,000-byte hard cap. The 4x variant is ~3 MB further over budget. The 4x model also completes 55 fewer steps (1,199 vs 1,254) due to ~5% slower step time (401 vs 383 ms/step).

## Verdict

**Neutral.** MLP_MULT=4 delivers a modest BPB gain over 3x (-0.0068 int8 BPB) with effectively zero quantization gap, confirming that wider MLPs help convergence. However, the int8+zlib artifact at 19.8 MB exceeds the 16 MB hard cap by ~24%, making this configuration not competition-viable without aggressive quantization (GPTQ, int5, or lower) or substantial parameter reduction elsewhere. The 55 fewer training steps and ~5% slower step time further limit its advantage. The sweet spot for MLP expansion likely lies at 3x combined with dimension/layer reduction to fit the artifact cap.

## Suggested follow-ups

- **GPTQ int6 quantization on the 4x model** to determine if the 19.8 MB artifact can compress below 16 MB while preserving the BPB advantage.
- **MLP_MULT=4 with fewer layers (7L or 8L)** to bring param count closer to the artifact budget while retaining the wider MLP — testing width vs depth tradeoff.
- **MLP_MULT=4 with reduced dim (e.g., 448)** as an alternative way to fit under the cap while keeping more MLP capacity.
- **Run MLP_MULT=2 baseline** on same screening config to quantify the full 2x→3x→4x progression.
- **Multi-seed runs (n>=3)** of the best MLP configuration to establish statistical significance before promoting.
- **Combined sweep: MLP_MULT × num_layers** to find the Pareto-optimal (BPB, artifact size) configuration.
