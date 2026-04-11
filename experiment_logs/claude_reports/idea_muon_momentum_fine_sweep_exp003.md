# muon_mom=0.97

## Hypothesis

Within the fine Muon-momentum sweep, `MUON_MOMENTUM=0.97` may be optimal
at the screening wallclock. The intuition is that a slightly higher-than-
default momentum improves gradient smoothing against the limited number
of effective update steps reachable inside the 480 s single-GPU screening
window, while remaining low enough to track local curvature changes.

## Configuration

| Env var | Value |
| --- | --- |
| `MUON_MOMENTUM` | `0.97` |

- **Recipe id:** `null` — single env-override on the screening default;
  not stacked on the current SOTA chain.
- **Stage reached:** `gate` (passed, **not** promoted).
- **Model:** `model_params = 17,059,912`, attention `gqa` 8h / 4 kv,
  `tie_embeddings=True`.
- **Data / shape:** `fineweb10B_sp1024`, `train_seq_len=1024`,
  `train_batch_tokens=524288`, `grad_accum_steps=8`.
- **Optim LRs:** `embed_lr=0.05`, `head_lr=0.0`, `matrix_lr=0.04`,
  `scalar_lr=0.04`, `warmup_steps=20`, `seed=1337`.
- **Budget:** `iterations=20000`, `max_wallclock_seconds=480`,
  `world_size=1`.

## Results

The run stopped cleanly on the wallclock cap at step **1434 / 20000**
(~334.80 ms/step). No NaNs, no warnings; the step-2 `train_loss=16.74`
spike is the usual cold-Muon transient and loss descends cleanly
thereafter. Peak memory **10,303 MiB** allocated.

Key log lines:

```
model_params:17059912
attention_mode:gqa num_heads:8 num_kv_heads:4
train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:20 max_wallclock_seconds:480.000
step:1000/20000 val_loss:2.3013 val_bpb:1.3629 train_time:334849ms step_avg:334.85ms
step:1434/20000 val_loss:2.2317 val_bpb:1.3218 train_time:480102ms step_avg:334.80ms
stopping_early: wallclock_cap train_time:480102ms step:1434/20000
Serialized model int8+zlib: 14307775 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
final_int8_zlib_roundtrip val_loss:2.2333 val_bpb:1.3227 eval_time:10853ms
final_int8_zlib_roundtrip_exact val_loss:2.23331788 val_bpb:1.32269665
```

| Metric | Value | Δ vs baseline (1.081) |
| --- | --- | --- |
| Final val_bpb (step 1434, fp16) | 1.32180 | +0.24080 |
| `screen_ema_bpb` | 1.28966815 | **+0.20867** |
| `gate_int6_bpb` (int8+zlib roundtrip) | 1.32270 | +0.24170 |
| `gate_quant_gap` | 3.35e-06 | effectively lossless |
| Artifact (int8+zlib payload) | 14,307,775 B (~14.31 MB) | under the 16 MB cap |
| `gate_passed` | true | — |
| `promote_ema_bpb` / `promote_int6_bpb` | `null` / `null` | not promoted |

**Baseline caveat.** The 1.081 reference is a full 10-min / 8×H100
SOTA-class run. This experiment is a 17 M-param, sp1024, single-GPU,
480 s screen with only `MUON_MOMENTUM` overridden, so the +0.21 nats
absolute gap mostly reflects the SOTA-vs-screen structural delta. The
run's real value lies in its ranking against its own sweep cohort
(`idea_muon_momentum_fine_sweep_exp001/002/004`), not against the SOTA
baseline directly.

## Verdict

**neutral.** The gate passed, the quantization gap is effectively zero
(3.35e-06 nats), and the int8+zlib artifact fits comfortably under
16 MB. But this is a screening-harness single point whose absolute delta
vs SOTA is structural, and it was not promoted. A win/regression call
requires the sibling momentum points from the same sweep under the same
480 s cap.

## Suggested follow-ups

- Aggregate `screen_ema_bpb` and `gate_int6_bpb` across all
  `idea_muon_momentum_fine_sweep_exp00*` runs under a **matched 480 s
  cap** and pick the cohort leader; only promote that one.
- If `0.97` wins the cohort, re-run at the full 10-min 8×H100 budget
  across ≥3 seeds so the result can clear the 0.005-nat / p < 0.01
  record bar (seed noise at 17 M params is comparable to that bar).
- Test the winning momentum **stacked on the current SOTA chain**
  (Self-Gen GPTQ + all-layer XSA + 11L) rather than the screening
  default — the sp1024 optimum may not transfer to sp8192 / deeper
  depths.
- Probe interactions with `MUON_WD`, Parallel Muon, and `warmdown3500`,
  since recent SOTA chains change effective update scale and that can
  shift the momentum optimum.
- Only extend the sweep to `≥0.98` if the cohort winner lands at the top
  edge of the current range; otherwise stop fine-tuning momentum and
  redirect effort toward higher-impact levers (QK5, all-layer XSA,
  self-gen GPTQ calibration).
