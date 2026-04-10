# seq2048 double-batch

**Experiment ID:** `idea_shorter_sequences_more_updates_exp002`
**Date:** 2026-04-10
**Host:** 206.125.32.60 (1x GPU, slot 4)

## Hypothesis

Longer context (seq_len=2048) may yield better BPB if the per-step cost of processing longer sequences is offset by richer contextual signal per token, even though fewer optimizer steps are completed in the same wallclock budget. Batch tokens were doubled to 1048576 to maintain the same number of sequences per batch as the default configuration.

## Configuration

| Parameter | Value | Default |
|---|---|---|
| `TRAIN_SEQ_LEN` | **2048** | 1024 |
| `TRAIN_BATCH_TOKENS` | **1048576** | 524288 |

All other parameters at defaults: 17M params, GQA 8/4 heads, tied embeddings, 480s wallclock cap, seed 1337. No recipe applied.

**Sibling experiment:** `exp001` used seq_len=512, batch_tokens=262144 (shorter sequences, more steps).

## Results

| Metric | exp002 (seq 2048) | exp001 (seq 512) |
|---|---|---|
| Steps completed | 642 / 20000 | 2735 / 20000 |
| Step avg | 748 ms | 176 ms |
| Tokens processed | ~673M | ~717M |
| Throughput | ~1.40M tok/s | ~1.49M tok/s |
| Final val_bpb (fp16) | 1.3703 | 1.3354 |
| EMA BPB (screen) | 1.3627 | -- |
| Int8+zlib BPB (gate) | **1.3777** | **1.3375** |
| Quant gap | ~0.007 | ~0.002 |
| Peak memory | 20,245 MiB | 5,332 MiB |
| Artifact size (int8+zlib) | 10.97 MB | 15.63 MB |

**Delta (exp002 vs exp001):** +0.0402 int6 BPB -- a large regression.

### Key log lines (exp002)

```
train_batch_tokens:1048576 train_seq_len:2048 iterations:20000 warmup_steps:20 max_wallclock_seconds:480.000
step:642/20000 val_loss:2.3137 val_bpb:1.3703 train_time:480149ms step_avg:747.89ms
stopping_early: wallclock_cap train_time:480149ms step:642/20000
peak memory allocated: 20245 MiB reserved: 20856 MiB
final_int8_zlib_roundtrip_exact val_loss:2.32620627 val_bpb:1.37771038
```

## Verdict

**regression**

Doubling the sequence length to 2048 is clearly harmful in this wallclock-constrained regime. The 4.25x higher per-step cost (748ms vs 176ms for seq 512) resulted in only 642 optimizer steps vs 2735 for the short-sequence variant, while total token throughput was similar (~673M vs ~717M). The final int8 BPB of 1.3777 is +0.040 worse than exp001's 1.3375. Memory usage also ballooned to 20 GiB (nearly 4x that of exp001), and the quantization gap widened from ~0.002 to ~0.007.

The results strongly confirm that at this model scale and wallclock budget, **more optimizer steps with shorter sequences dominate longer context**. The model does not extract enough value from 2048-token context to offset losing ~75% of its update steps.

## Suggested follow-ups

- **Test seq_len=256 or 128** with proportionally smaller batch tokens to push the "more steps" direction further and find the optimum.
- **Combine seq_len=512 with default batch_tokens=524288** (2x sequences per batch) to test whether increased batch diversity at short context helps.
- **Curriculum schedule:** start with seq_len=512 for fast early convergence, then switch to 1024 in the final training phase to capture longer-range dependencies without sacrificing step count.
- **Sweep seq_len={512, 768, 1024}** at fixed batch_tokens=524288 to locate the optimal sequence length vs step count tradeoff.
- **Compressed LR schedule for long-seq runs:** the 642-step budget may be too few for the default schedule. A faster warmup and earlier warmdown tuned for ~600 steps could partly close the gap (though the magnitude of regression suggests schedule alone won't fix it).
