# seq2048 double-batch

## Hypothesis

Increasing sequence length from 512 to 2048 while scaling batch tokens proportionally (262K to 1M) might yield better BPB if the model benefits from longer context, even though the per-step cost is higher and fewer gradient updates fit within the wallclock budget. This tests whether richer per-step signal from longer sequences can offset the reduced number of optimization steps.

## Configuration

| Parameter | exp001 (screen, seq512) | exp002 (gate, seq2048) |
|---|---|---|
| `TRAIN_SEQ_LEN` | 512 | 2048 |
| `TRAIN_BATCH_TOKENS` | 262144 | 1048576 |
| Model params | 17,059,912 | 17,059,912 |
| GPUs | 1 | 1 |
| Grad accum steps | 8 | 8 |
| Step avg | ~176ms | ~748ms |
| Steps completed | 2162 | 642 |
| Wallclock | 380s | 480s |
| Attention | GQA (8h/4kv) | GQA (8h/4kv) |

No recipe; default baseline model architecture (9L, 512 dim, 1024 vocab, tied embeddings).

## Results

| Metric | exp001 (seq512) | exp002 (seq2048) | Delta |
|---|---|---|---|
| Final val_bpb | 1.3519 | 1.3703 | +0.0184 (worse) |
| EMA BPB | 1.3519 | 1.3627 | +0.0108 (worse) |
| Int8+zlib BPB | 1.3534 | 1.3777 | +0.0243 (worse) |
| Quant gap | +0.0015 | +0.0074 | +0.0059 (worse) |
| Peak memory | 5,332 MiB | 20,245 MiB | +14,913 MiB |
| Tokens seen | ~566M (2162 * 262K) | ~673M (642 * 1M) | +107M |

Key training log lines (exp002):
```
train_batch_tokens:1048576 train_seq_len:2048 iterations:20000 warmup_steps:20 max_wallclock_seconds:480.000
step:642/20000 val_loss:2.3137 val_bpb:1.3703 train_time:480149ms step_avg:747.89ms
stopping_early: wallclock_cap train_time:480149ms step:642/20000
peak memory allocated: 20245 MiB reserved: 20856 MiB
final_int8_zlib_roundtrip_exact val_loss:2.32620627 val_bpb:1.37771038
```

Despite seeing ~19% more total tokens, exp002 performed significantly worse across all metrics. The 3.4x reduction in gradient updates (642 vs 2162) was not compensated by longer context. The quantization gap also widened considerably (0.0015 to 0.0074), suggesting the int8 compression struggles more with the longer-context weight distributions, and memory usage nearly quadrupled.

## Verdict

**Regression.** Longer sequences (2048) with proportionally scaled batch tokens hurt BPB by +0.024 on the quantized model compared to the seq512 baseline. The number of gradient updates matters far more than per-step context length at this model scale and training budget. The gate passed the absolute threshold but the configuration is strictly worse than exp001.

## Suggested follow-ups

- **Shorter sequences (256 or 128):** If fewer tokens per sequence with more gradient updates helps, push the tradeoff further. Shorter sequences = more steps = potentially better optimization.
- **Seq2048 with higher LR or adjusted warmdown:** The 642-step budget may have been too few for the LR schedule to reach its optimal range. A compressed schedule (faster warmup, earlier warmdown) tuned for ~600 steps could close part of the gap.
- **Mixed sequence lengths:** Train with seq512 for most of the budget, then fine-tune with seq2048 in the final phase to capture long-range dependencies without sacrificing gradient updates.
- **Seq512 with more GPUs (multi-GPU scaling):** exp001's second run on 2 GPUs achieved ~85ms/step. Scaling to more GPUs while keeping seq512 could push past 4000+ steps and further improve BPB.
- **Gradient accumulation tuning:** Instead of scaling batch tokens with sequence length, keep batch tokens fixed at 262K with seq2048 (fewer sequences per batch, same token budget, same step count).
