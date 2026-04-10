# MLP_MULT=4

## Hypothesis

MLP expansion 4× trades artifact size for convergence speed. The premise is
that a wider FFN should give each step more representational throughput,
hopefully recovering the wallclock cost via faster per-step loss drop and a
smaller quantization gap — but at the risk of blowing the 16 MB artifact cap
and starving the run of steps within the 480 s screening budget.

## Configuration

| Env override | Value |
|---|---|
| `MLP_MULT` | `4` |

- **Recipe:** none (single env-var override on the default baseline)
- **Source ref:** *(none provided)*
- **Reproduction:** no
- **Model params:** 26,497,096 (~26.5 M) — `model_params:26497096` (log L5)
- **Iterations planned:** 20,000, wallclock cap 480 s
- **Seed:** 1337
- **Attention:** GQA, 8 heads / 4 KV heads
- **Batch:** 524,288 tokens, seq_len 1024, grad_accum 8

## Results

Quoted key lines from
`experiment_logs/idea_mlp_expansion_sweep_2x_3x_4x/idea_mlp_expansion_sweep_2x_3x_4x_exp002/train.log`:

```
model_params:26497096
step:1000/20000 val_loss:2.2588 val_bpb:1.3378 train_time:400637ms
step:1199/20000 val_loss:2.2260 val_bpb:1.3184 train_time:480205ms
stopping_early: wallclock_cap train_time:480205ms step:1199/20000
peak memory allocated: 13017 MiB reserved: 13386 MiB
Serialized model: 104973719 bytes
Code size: 47693 bytes
Serialized model int8+zlib: 19775260 bytes (payload:26634528 payload_ratio:3.94x)
Total submission size int8+zlib: 19822953 bytes
final_int8_zlib_roundtrip val_loss:2.2280 val_bpb:1.3195 eval_time:13493ms
final_int8_zlib_roundtrip_exact val_loss:2.22798977 val_bpb:1.31954104
```

| Metric | Value | Δ vs baseline (1.0810) |
|---|---|---|
| screen_ema_bpb | 1.31026 | **+0.22926** |
| gate_int6_bpb (int8+zlib roundtrip) | 1.31954 | **+0.23854** |
| gate_quant_gap | −4.10e-05 | ≈0 (no quant gap) |
| gate_artifact_mb (metadata) | 0.0 | — (field appears unreported) |
| int8+zlib submission size (log) | ~19.82 MB | **> 16 MB cap** |
| final step reached | 1,199 / 20,000 | hit 480 s wallclock cap |
| step avg | ~401 ms/step | slow vs typical baseline |
| peak VRAM | 13,017 MiB | — |
| gate_passed (metadata) | True | — |

**Observations:**
1. **Did not converge.** Only 1,199/20,000 steps finished before the 480 s
   cap. Val_bpb was still 1.3378 at step 1000 and only reached 1.3184 by
   step 1199 — hundreds of milli-nats above the 1.0810 baseline.
2. **Quant gap is essentially zero** (int8+zlib is −4e-05 *better* than
   EMA). That is the one encouraging signal — but it's unverified at
   convergence.
3. **Artifact is over cap.** The int8+zlib submission is 19.82 MB, ~24 %
   above the 16,000,000-byte hard limit. The `gate_artifact_mb: 0.0`
   metadata field is misleading; the log is authoritative.
4. **No warnings or divergence.** Training loss decreased monotonically
   through the logged checkpoints.

## Verdict

**regression**

At the current baseline shape, MLP_MULT=4 blows both budgets: ~26.5 M params
→ 19.8 MB int8+zlib artifact (over the 16 MB cap) and ~400 ms/step → only
1,199 steps in 480 s, leaving val_bpb at +0.239 nats vs the 1.0810 SOTA
baseline. The hypothesis about convergence speed is effectively unfalsified
because the run never converged; the only positive signal is a near-zero
quant gap. Do not stack on top of current SOTA.

## Suggested follow-ups

- **Shrink-then-widen:** retry MLP_MULT=4 with fewer layers and/or smaller
  d_model so parameters land in the ~6–9 M range, artifact fits under 16 MB
  int8+zlib, and step time drops back to ~150 ms. This is the only way to
  actually test whether wide FFN helps per-step convergence.
- **Isolate the quant-gap claim:** matched-param ablation MLP_MULT=2 deeper
  vs MLP_MULT=4 shallower, to confirm whether wide FFN really reduces the
  int6/int8 quant gap. The directional signal here (−4e-05) is the one
  thing worth salvaging.
- **Fix the gate reporter:** `gate_artifact_mb=0.0` was stored while the
  log shows 19.82 MB. The gate should read the real int8+zlib size and
  refuse to mark `gate_passed=True` on runs that exceed the 16 MB cap.
- **GPTQ / int6 / ternary on a shrunk wide-FFN variant:** given the
  near-zero int8+zlib gap, a more aggressive quantizer on a smaller wide
  model may be where this direction actually pays off.
- **Deprioritize against current SOTA:** do not stack MLP_MULT=4 on
  `rec_20260410_..._sp8192_3layerrecur_parresid_q` (val_bpb=1.0810) until
  a budget-compliant variant beats MLP_MULT=3 at matched wallclock.
