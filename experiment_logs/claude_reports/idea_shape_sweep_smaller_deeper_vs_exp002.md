# shallower-wider 7L x 640d

## Hypothesis

Fewer transformer layers reduce sequential sync points per step, so a 7-layer
geometry should raise tokens/sec and buy back training steps under the
10-minute wallclock cap. Widening to `d_model=640` with 10 heads / 2 KV heads
(GQA) is meant to offset the capacity lost from cutting depth. Net hope:
shallower-wider beats deeper-narrower at this parameter budget once the
throughput bonus is folded in.

## Configuration

| Env var        | Value |
|----------------|-------|
| `NUM_LAYERS`   | 7     |
| `MODEL_DIM`    | 640   |
| `NUM_HEADS`    | 10    |
| `NUM_KV_HEADS` | 2     |

- **Recipe:** none (`recipe_id: null` — env-only override on default `train_gpt.py`)
- **Reproduction:** false
- **Source ref:** —

Resolved from `train.log`:

- `model_params:19025350` (~19.0 M)
- `attention_mode:gqa num_heads:10 num_kv_heads:2`
- `tie_embeddings:True embed_lr:0.05 head_lr:0.0 matrix_lr:0.04 scalar_lr:0.04`
- `train_batch_tokens:524288 train_seq_len:1024 iterations:20000 warmup_steps:20`
- `max_wallclock_seconds:480.000 world_size:1 grad_accum_steps:8` (single-GPU screen)
- `seed:1337`

## Results

Key lines from `train.log`:

```
step:0/20000    val_loss:6.9469 val_bpb:4.1143 train_time:0ms
step:1000/20000 val_loss:2.3194 val_bpb:1.3737 train_time:322557ms step_avg:322.56ms
step:1489/20000 val_loss:2.2444 val_bpb:1.3293 train_time:480124ms step_avg:322.45ms
stopping_early: wallclock_cap train_time:480124ms step:1489/20000
peak memory allocated: 9537 MiB reserved: 9628 MiB
Serialized model int8+zlib: 15133368 bytes (payload:19135512 raw_torch:19170649 payload_ratio:3.91x)
Total submission size int8+zlib: 15181061 bytes
final_int8_zlib_roundtrip val_loss:2.2474 val_bpb:1.3310 eval_time:10793ms
final_int8_zlib_roundtrip_exact val_loss:2.24736472 val_bpb:1.33101597
```

Baseline val_bpb for delta: **1.10625353**

| Metric             | Value       | Δ vs baseline              |
|--------------------|-------------|----------------------------|
| screen_ema_bpb     | **1.29712** | **+0.19087 (worse)**       |
| gate_int6_bpb      | **1.33100** | **+0.22475 (worse)**       |
| gate_quant_gap     | −1.6e−05    | ≈ 0 (clean roundtrip)      |
| gate_artifact_mb   | 0.0 reported (actual int8+zlib ≈ 15.18 MB, under 16 MB cap) | — |
| gate_passed        | true (quant-gap/artifact only, not BPB) | —          |
| wallclock          | 480.124 s — hit cap at step 1489 / 20000 | —         |
| peak GPU mem       | 9537 MiB    | —                          |
| model_params       | 19,025,350  | —                          |
| promote_ema_bpb    | null (not promoted) | —                  |
| promote_int6_bpb   | null        | —                          |

Notes:

- Quantization gap is essentially zero — the shape compresses cleanly into the
  16 MB cap at int8+zlib.
- Run was **single-GPU** (`world_size:1`) under a 480 s screening cap, so the
  "fewer sync points" claim was not actually exercised — multi-GPU collectives
  were absent in this environment.
- Only **1489 / 20000 steps** completed before the wallclock cap, leaving the
  config deeply undertrained relative to the 8×H100 SOTA regime that produced
  the 1.106 baseline.
- No warnings or divergence in the log. Transient spike at step 2
  (`train_loss:19.6042`) recovers by step 4 — normal warmup behavior.
- `step_avg` is stable at ~322 ms across the whole run — no throughput cliff.

## Verdict

**regression** — absolute val_bpb (EMA 1.297, int6 1.331) is ~0.19–0.22 nats
worse than the 1.10625 baseline. `gate_passed=true` reflects only the
quant-gap and artifact-size checks, not a BPB win. The single-GPU,
wallclock-truncated screen also undersells the hypothesis's core claim
(sync-point reduction), so the result does not fully falsify shallower-wider
— it just shows that at ~19 M params and 1489 steps, 7L × 640d does not
approach SOTA BPB and should not be promoted.

## Suggested follow-ups

- **Drop 7L from the depth sweep** for now — the screen gap (>0.19 nats) is
  too large to justify additional seeds; leaderboard SOTA clusters at 10–11
  layers.
- **Matched-param depth control:** screen 10L/480d and 11L/448d at ~19 M
  params under the same gate to confirm depth dominance before abandoning
  the shape axis.
- **Multi-GPU rerun before declaring it dead.** If the tokens/sec hypothesis
  is to be tested at all, re-run 7L × 640d on 8×H100 SXM at the full budget
  so the sync-point lever is actually exercised — the 1-GPU gate masked it.
- **7L/640d + MLP 3× expansion** to recover capacity through a wider FFN
  rather than more layers, if the shallow-wide direction is kept alive.
- **7L/640d + XSA last-4** — cross sliding attention may partially compensate
  for reduced depth by extending effective receptive field.
- **Stop env-only forks from vanilla `train_gpt.py`.** Future shape sweeps
  should stack on the current SOTA recipe (Self-Gen GPTQ + all-layer XSA +
  EMA) so the result is comparable to the frontier.
- **Fix the screening budget.** A 480 s 1-GPU gate starves shallow configs
  of steps. Consider gating on tokens/sec-adjusted projected BPB, not
  absolute truncated BPB, before declaring more shape regressions.
