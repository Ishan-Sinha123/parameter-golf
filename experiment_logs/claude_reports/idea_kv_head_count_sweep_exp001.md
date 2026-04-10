# kv_heads=2

## Hypothesis

Reducing GQA key/value head count from the baseline to `NUM_KV_HEADS=2` (with the default 8 query heads, i.e. 8:2 GQA) should balance KV-cache and KV-projection parameter cost against BPB — shrinking K/V projections without meaningfully hurting loss. Any bytes freed can later be redeployed into depth or MLP width in a stacked recipe.

## Configuration

| Env var | Value |
| --- | --- |
| `NUM_KV_HEADS` | `2` |

- Recipe: *no `recipe_id` — bare env override on the current screening baseline.*
- Attention mode confirmed in log: `attention_mode:gqa num_heads:8 num_kv_heads:2`.
- Model params: `15,880,264`.
- Tokenizer: `fineweb_1024_bpe.model` (sentencepiece, 1024 vocab).
- Schedule: `iterations:20000 warmup_steps:20 max_wallclock_seconds:540.000`.
- Seed: `1337`, `world_size: 1`, `grad_accum_steps: 8`.
- Source ref: *(empty)*, reproduction: no.

## Results

Key lines from `experiment_logs/idea_kv_head_count_sweep/idea_kv_head_count_sweep_exp001/train.log`:

```
model_params:15880264
attention_mode:gqa num_heads:8 num_kv_heads:2
step:0/20000    val_loss:6.9378 val_bpb:4.1090
step:1000/20000 val_loss:2.3143 val_bpb:1.3706 train_time:322337ms
step:1680/20000 val_loss:2.2201 val_bpb:1.3149 train_time:540126ms step_avg:321.50ms
stopping_early: wallclock_cap train_time:540126ms step:1680/20000
peak memory allocated: 9814 MiB reserved: 9896 MiB
Serialized model int8+zlib: 14560584 bytes (payload:17169696 raw_torch:17209405 payload_ratio:3.64x)
Total submission size int8+zlib: 14608277 bytes
final_int8_zlib_roundtrip val_loss:2.2221 val_bpb:1.3160 eval_time:10387ms
final_int8_zlib_roundtrip_exact val_loss:2.22205572 val_bpb:1.31602656
```

Caveat: the run hit the **540 s screening wallclock cap at step 1680 / 20000** (`stopping_early: wallclock_cap`, ≈8% of planned steps) — early-training values, not a converged result. No warnings or NaNs emitted.

Baseline val_bpb for delta calc: **1.10625353**

| Metric | Value | Δ vs baseline |
| --- | --- | --- |
| screen EMA val_bpb | 1.30301978 | **+0.19677** |
| gate int6 val_bpb | 1.31600000 | **+0.20975** |
| fp → int6 quant gap | −2.66e-05 | ≈ 0 (effectively lossless) |
| gate artifact size (MB) | 0.0 (not populated) | — |
| int8+zlib artifact bytes | 14,608,277 | under 16 MB cap |
| peak memory | 9,814 MiB | — |
| training steps reached | 1680 / 20000 | early stop |
| gate passed | ✅ `true` | — |
| promote_ema_bpb | `null` | not promoted |

## Verdict

**regression**

`NUM_KV_HEADS=2` came in ~0.197 nats above the 1.10625 reference baseline and was not promoted from screening. The gate "passed" only in the trivial sense that quantization is essentially lossless (`−2.66e-5` fp→int6 gap) and the int8+zlib artifact fits under 16 MB — there is no BPB win. Because the run was killed by the 540 s wallclock cap at step 1680 / 20000, the regression is confounded with severe undertraining and should not be read as definitive evidence that 2 KV heads is fundamentally bad.

## Suggested follow-ups

- Re-run with a **matched, uncapped 10-minute** wallclock so the schedule actually converges, then compare `NUM_KV_HEADS ∈ {1, 2, 4, 8}` on the same seed and recipe.
- Cross-check against `idea_kv_head_count_sweep_exp002` before drawing sweep-level conclusions — one screen point is not a curve.
- Compose `NUM_KV_HEADS=2` on top of the current SOTA chain (XSA / GPTQ / LeakyReLU² / TTT) across ≥3 seeds — composability at the frontier is what drives records, and a loss on the default baseline doesn't rule out a gain after stacking.
- If `NUM_KV_HEADS=2` keeps regressing at matched compute, try `NUM_KV_HEADS=1` (MQA) as the lower bound and redirect freed parameters into MLP 3× expansion or an extra layer.
- Investigate the `gate_artifact_mb = 0.0` reporting: the raw log shows a 14.6 MB int8+zlib artifact, so the gate metric appears not to be populated from the training log.
- Confirm whether the 540 s screening cap is intentional for this stage or should be lifted — the current budget isn't long enough to distinguish real BPB losses from undertraining.
