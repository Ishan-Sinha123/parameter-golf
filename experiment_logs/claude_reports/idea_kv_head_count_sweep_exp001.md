# kv_heads=2

## Hypothesis

Reducing KV heads from the default 4 to 2 balances memory savings against BPB degradation. Fewer KV heads reduce the parameter count allocated to key/value projections, freeing capacity for other components, and may improve throughput — but risk losing representational power in multi-head attention.

## Configuration

| Parameter | Value |
|-----------|-------|
| `NUM_KV_HEADS` | `2` |
| Recipe | None (baseline + override) |
| Attention mode | GQA, 8 heads, 2 KV heads |
| Model params | 15,880,264 |
| Steps completed | 1,680 / 20,000 (wallclock cap at 540s) |
| Seed | 1337 |

All other settings at baseline defaults (9 layers, 512 dim, 1024 vocab, tied embeddings).

## Results

| Metric | Value |
|--------|-------|
| EMA BPB (screen) | **1.3030** |
| Final val_bpb (step 1680) | 1.3149 |
| Int8+zlib val_bpb | 1.3160 |
| Gate int6 BPB | 1.3160 |
| Quant gap | -0.00003 (negligible) |
| Artifact size (int8+zlib) | 14.61 MB |
| Gate passed | Yes |
| Peak memory | 9,814 MiB |

**Delta vs project baseline (1.2244 BPB, 4 KV heads):** +0.079 BPB (regression).

### Key log lines

```
attention_mode:gqa num_heads:8 num_kv_heads:2
step:1000/20000 val_loss:2.3143 val_bpb:1.3706
step:1680/20000 val_loss:2.2201 val_bpb:1.3149
stopping_early: wallclock_cap train_time:540126ms step:1680/20000
final_int8_zlib_roundtrip val_loss:2.2221 val_bpb:1.3160
```

Training hit the wallclock cap at step 1680 — only 8.4% of the 20k step budget was completed. Step throughput (~321 ms/step) was similar to baseline, so reducing KV heads did not yield meaningful speed gains.

## Verdict

**regression** — 2 KV heads degrades BPB by ~0.079 vs the 4-KV-head baseline with no compensating throughput benefit. The loss curve was still clearly descending at the wallclock cutoff, but the gap at matched training time is significant.

## Suggested follow-ups

- **KV heads = 6 or 8**: sweep upward from the default 4 to see if more KV heads improve BPB within the parameter budget.
- **KV heads = 2 with deeper model**: pair 2 KV heads with 10-11 layers to check if the parameter savings from fewer KV heads are better spent on depth.
- **KV heads = 2 with larger MLP expansion**: redirect saved parameters into MLP width (e.g., 3x or 4x expansion) to test the compute-vs-capacity tradeoff.
- **Longer training budget**: re-run with a higher wallclock cap or on multi-GPU to see if 2 KV heads eventually converges closer to baseline.
