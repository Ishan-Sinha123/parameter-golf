# MLP_MULT=4

## Hypothesis

MLP expansion factor 4x trades artifact size for faster convergence and lower per-step loss. A wider MLP should increase model capacity and improve BPB, but the larger parameter count may push the int8+zlib artifact past the 16 MB competition limit and reduce total training steps within the wallclock budget.

## Configuration

| Parameter | Value |
|---|---|
| `MLP_MULT` | `4` |
| Model params | 26,497,096 |
| Layers | 9 (default) |
| Dim | 512 (default) |
| Vocab | 1024 (default) |
| KV heads | 4 (default) |
| Seed | 1337 |
| Wallclock cap | 480 s |
| GPU | 1x H100 (gate run) |
| Commit | `165884039f` |
| Recipe | None (single env override) |

Companion run: `exp001` with `MLP_MULT=3` on the same commit and wallclock budget.

## Results

| Metric | MLP_MULT=4 (this run) | MLP_MULT=3 (exp001) |
|---|---|---|
| EMA val_bpb (screen) | **1.3103** | -- |
| Final val_bpb (fp32) | 1.3184 | 1.3251 |
| int8+zlib val_bpb | **1.3195** | 1.3263 |
| Quant gap (int8-ema) | -0.0000 (noise) | -- |
| Steps completed | 1,199 | 1,254 |
| Step avg | 400.5 ms | 383.0 ms |
| Peak VRAM | 13,017 MiB | 11,416 MiB |
| Model params | 26.5 M | 21.8 M |
| int8+zlib artifact | **19.8 MB** | **16.7 MB** |

**Key log lines (exp002):**
```
model_params:26497096
step:1000/20000 val_loss:2.2588 val_bpb:1.3378
step:1199/20000 val_loss:2.2260 val_bpb:1.3184
stopping_early: wallclock_cap train_time:480205ms step:1199/20000
Serialized model int8+zlib: 19775260 bytes
final_int8_zlib_roundtrip val_loss:2.2280 val_bpb:1.3195
```

**Delta vs MLP_MULT=3:** int8+zlib BPB improves by **-0.0068** (1.3195 vs 1.3263). However, artifact size grows from 16.7 MB to 19.8 MB, both exceeding the 16 MB hard cap. The 4x variant is ~3 MB further over budget.

## Verdict

**Promising.** MLP 4x delivers a meaningful BPB gain over 3x (-0.0068 int8 BPB), with essentially zero quantization gap. However, neither the 3x nor 4x variant fits within the 16 MB artifact limit at int8+zlib. The 4x variant would require aggressive quantization (int6/int5/GPTQ) to become competition-eligible. The 55 fewer training steps (1199 vs 1254) and ~5% slower step time are acceptable tradeoffs if the artifact budget can be solved.

## Suggested follow-ups

- **GPTQ int6 quantization on the 4x model** to determine if the 19.8 MB artifact can compress below 16 MB while preserving the BPB advantage.
- **MLP_MULT=4 with fewer layers (8L)** to bring param count closer to the 3x baseline while keeping the wider MLP, testing whether width beats depth at this scale.
- **MLP_MULT=4 with reduced dim (e.g., 448 or 480)** as an alternative way to fit under the artifact cap while retaining more MLP capacity.
- **Multi-seed runs (n>=3)** of the best MLP configuration to establish statistical significance of the BPB delta before promoting.
- **Combined sweep: MLP_MULT x num_layers** to find the Pareto-optimal (BPB, artifact size) configuration.
