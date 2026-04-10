# kv_heads=8

## Hypothesis

Increasing the number of KV heads to 8 (full multi-head attention, matching query heads) balances memory usage against BPB by giving each query head its own key-value projection, potentially improving representational capacity compared to grouped-query attention with fewer KV heads.

## Configuration

| Parameter | Value |
|-----------|-------|
| `NUM_KV_HEADS` | `8` |
| attention_mode | GQA (effectively MHA since kv_heads == num_heads) |
| num_heads | 8 |
| model_params | 19,419,208 |
| wallclock_cap | 540s (1 GPU) |
| seed | 1337 |
| steps completed | 1537 / 20000 |
| step_avg | ~351 ms |

No recipe; single env-var override on the default baseline config.

## Results

| Metric | kv_heads=8 (this) | kv_heads=2 (exp001) |
|--------|-------------------|---------------------|
| model_params | 19,419,208 | 15,880,264 |
| steps completed | 1,537 | 1,680 |
| step_avg (ms) | 351 | 321 |
| val_bpb @ stop | 1.3125 | 1.3149 |
| int8+zlib val_bpb | **1.3136** | **1.3160** |
| int8+zlib artifact (bytes) | 16,010,927 | 14,608,277 |
| peak memory (MiB) | 11,270 | 9,814 |
| screen_ema_bpb | 1.2828 | — |

**Key training log lines (exp002):**
```
attention_mode:gqa num_heads:8 num_kv_heads:8
model_params:19419208
step:1537/20000 val_loss:2.2161 val_bpb:1.3125 train_time:540148ms step_avg:351.43ms
stopping_early: wallclock_cap train_time:540148ms step:1537/20000
Serialized model int8+zlib: 15963234 bytes
Total submission size int8+zlib: 16010927 bytes
final_int8_zlib_roundtrip_exact val_loss:2.21793496 val_bpb:1.31358602
```

**Observations:**
- kv_heads=8 gives a marginal **-0.0024 BPB** improvement over kv_heads=2 after int8 quantization.
- However, the model is **22% larger** (19.4M vs 15.9M params), trains **9% slower** per step (351 vs 321 ms), and completes **8.5% fewer steps** within the wallclock budget.
- The int8+zlib artifact is **16,010,927 bytes** — this exceeds the competition's 16,000,000-byte hard cap by ~11 KB. The submission would be disqualified without further compression or parameter reduction.
- The quant gap is negligible (0.0014% per metadata), so quantization fidelity is excellent.

## Verdict

**neutral** — kv_heads=8 (full MHA) produces a tiny BPB improvement over kv_heads=2 but the tradeoffs are unfavorable: the artifact barely exceeds the 16 MB cap, the model trains slower and completes fewer steps, and the marginal gain is not statistically meaningful from a single run. The default kv_heads=4 baseline remains the better operating point.

## Suggested follow-ups

- **kv_heads=4 vs kv_heads=6 sweep**: The default baseline uses 4 KV heads. A finer sweep between 4-6 could find a better tradeoff without blowing the artifact budget.
- **Combine with dimension reduction**: If kv_heads=8 helps capacity, pair it with a slight dim reduction (e.g., d_model=480) to bring the artifact back under 16 MB.
- **Multi-seed validation**: Run kv_heads=2, 4, 8 at 3 seeds each to get statistically significant deltas — the current 0.0024 difference is within noise.
- **8-GPU run**: The 1-GPU setup caps at ~1500 steps. An 8-GPU run would show whether the kv_heads=8 advantage grows with more training steps or saturates.
