# shallower-wider 7L x 640d

## Hypothesis

Fewer transformer layers mean fewer sequential sync points per training step,
so a 7-layer geometry should raise tokens/sec and buy back gradient updates
under the 10-minute wallclock cap. Widening to `d_model=640` with 10 heads
and 2 KV heads (GQA, 5× ratio) is meant to recover the capacity lost from
cutting depth. Net claim: shallower-wider beats deeper-narrower at this
parameter budget once the throughput bonus is folded in.

## Configuration

| Env var        | Value |
|----------------|-------|
| `NUM_LAYERS`   | 7     |
| `MODEL_DIM`    | 640   |
| `NUM_HEADS`    | 10    |
| `NUM_KV_HEADS` | 2     |

- **Recipe:** none (`recipe_id: null` — env-only override on default `train_gpt.py`)
- **Source ref:** — (empty)
- **Reproduction:** false

Resolved from `train.log`:

- `model_params:19025350` (~19.0 M params)
- `attention_mode:gqa num_heads:10 num_kv_heads:2`
- `tie_embeddings:True embed_lr:0.05 head_lr:0.0 matrix_lr:0.04 scalar_lr:0.04`
- `train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:20`
- `max_wallclock_seconds:480.000 world_size:1 grad_accum_steps:8` (single-GPU screen)
- `seed:1337`

## Results

Key lines from `train.log`:

```
step:0/20000    val_loss:6.9469 val_bpb:4.1143 train_time:0ms
step:2/20000    train_loss:19.6042 train_time:640ms step_avg:319.93ms
step:1000/20000 train_loss:2.3516 train_time:322556ms step_avg:322.56ms
step:1000/20000 val_loss:2.3194 val_bpb:1.3737 train_time:322557ms step_avg:322.56ms
step:1489/20000 val_loss:2.2444 val_bpb:1.3293 train_time:480124ms step_avg:322.45ms
stopping_early: wallclock_cap train_time:480124ms step:1489/20000
peak memory allocated: 9537 MiB reserved: 9628 MiB
Serialized model int8+zlib: 15133368 bytes (payload:19135512 raw_torch:19170649 payload_ratio:3.91x)
Total submission size int8+zlib: 15181061 bytes
final_int8_zlib_roundtrip val_loss:2.2474 val_bpb:1.3310 eval_time:10793ms
final_int8_zlib_roundtrip_exact val_loss:2.24736472 val_bpb:1.33101597
```

Baseline val_bpb for delta: **1.081** (current SOTA reference).

| Metric             | Value                                  | Δ vs baseline (1.081)            |
|--------------------|----------------------------------------|----------------------------------|
| screen_ema_bpb     | **1.29712**                            | **+0.21612 (worse)**             |
| gate_int6_bpb      | **1.33100**                            | **+0.25000 (worse)**             |
| gate_quant_gap     | −1.6e−05                               | ≈ 0 (clean int8+zlib roundtrip)  |
| gate_artifact_mb   | 0.0 reported (actual ≈ 15.18 MB)       | under 16 MB cap                  |
| gate_passed        | true (quant-gap/artifact only, not BPB)| —                                |
| wallclock          | 480.124 s — hit cap at step 1489/20000 | —                                |
| step_avg           | ~322.5 ms (flat across the run)        | —                                |
| peak GPU mem       | 9537 MiB                               | —                                |
| model_params       | 19,025,350                             | —                                |
| promote_ema_bpb    | null (not promoted)                    | —                                |
| promote_int6_bpb   | null                                   | —                                |

Notes:

- **Quant gap is essentially zero** — int8+zlib roundtrip lands within 2e−5
  of the screen EMA, and the 15.18 MB artifact fits comfortably under the
  16 MB cap. The shape compresses cleanly.
- **Single-GPU screen.** `world_size:1` means the "fewer sync points → more
  tokens/sec" claim was never actually exercised — there were no multi-GPU
  collectives to amortize. The throughput hypothesis is untested by this
  run.
- **Severely undertrained.** Only 1489 / 20000 iterations completed before
  the 480 s wallclock cap fired. SOTA baselines at 1.081 are trained for
  many more steps on 8×H100; this screen is not directly comparable.
- **No divergence or warnings.** Transient `train_loss:19.6042` spike at
  step 2 recovers to 9.61 → 6.44 → 6.13 by step 5 — normal warmup
  behaviour. `step_avg` is flat at ~322 ms throughout, no throughput cliff.

## Verdict

**regression** — absolute val_bpb (EMA 1.297, int6 1.331) is ~0.22–0.25 nats
worse than the 1.081 baseline. `gate_passed=true` reflects only the
quant-gap and artifact-size checks, not a BPB win. The single-GPU,
wallclock-truncated screen also undersells the hypothesis's core claim
(sync-point reduction), so this does not fully falsify shallower-wider —
it just shows that at ~19 M params and 1489 steps, 7L × 640d does not
approach SOTA BPB and should not be promoted.

## Suggested follow-ups

- **Drop 7L from the depth sweep** for now — the screen gap (>0.21 nats) is
  too large to justify additional seeds; leaderboard SOTA clusters at
  10–12 layers.
- **Matched-param depth control:** screen 10L/480d and 11L/448d at ~19 M
  params under the same gate to confirm depth dominance before abandoning
  the shape axis entirely.
- **Multi-GPU rerun before declaring it dead.** If the tokens/sec
  hypothesis is to be tested at all, re-run 7L × 640d on 8×H100 SXM at the
  full 10-min budget so the sync-point lever is actually exercised — the
  1-GPU gate masks it.
- **7L/640d + MLP 3× expansion** to recover capacity through a wider FFN
  rather than more layers, if the shallow-wide direction is kept alive.
- **7L/640d + XSA last-4** — cross sliding attention may partially
  compensate for reduced depth by extending the effective receptive field
  at low cost.
- **Stop env-only forks from vanilla `train_gpt.py`.** Future shape sweeps
  should stack on the current SOTA recipe (Self-Gen GPTQ + all-layer XSA +
  EMA, val_bpb ≈ 1.081) so deltas are comparable to the frontier rather
  than to a stripped baseline.
- **Fix the screening budget.** A 480 s 1-GPU gate starves shallow configs
  of steps. Consider gating on tokens/sec-adjusted projected BPB, not
  absolute truncated BPB, before declaring more shape regressions.
