# shallower-wider 7L x 640d

## Hypothesis

Reducing depth from 9 to 7 layers while increasing width from 512 to 640 dimensions trades representational depth for fewer synchronization points per forward pass. Fewer layers means fewer sequential ops, potentially yielding higher tokens/sec throughput that could offset the capacity loss from reduced depth — especially under a tight 10-minute wallclock budget where more training steps could compensate.

## Configuration

| Env var | Value |
|---|---|
| `NUM_LAYERS` | 7 |
| `MODEL_DIM` | 640 |
| `NUM_HEADS` | 10 |
| `NUM_KV_HEADS` | 2 |

- **Recipe:** None (`recipe_id` is `null` — single shape change on baseline)
- **Source ref:** —
- **Is reproduction:** false

## Results

Training log was **not captured** for this run. Only pipeline-reported gate metrics are available.

| Metric | Value | Delta vs baseline (1.2244) |
|---|---|---|
| screen_ema_bpb | **1.2971** | +0.0727 (worse) |
| gate_int6_bpb | **1.3310** | +0.1066 (worse) |
| gate_quant_gap | −0.00002 | — (negligible; effectively zero) |
| gate_artifact_mb | 0.0 | — (not captured) |
| gate_passed | true | — |
| promote_ema_bpb | — | — |
| promote_int6_bpb | — | — |

Notes:
- The quant gap of ~0 is excellent — the 7L/640d shape quantizes cleanly to int6.
- EMA BPB of 1.297 is +0.073 worse than the 9L/512d baseline (1.2244), a clear regression.
- Gate int6 BPB of 1.331 is +0.107 worse than baseline, confirming the capacity loss is real even after quantization.
- Artifact size was not recorded (0.0 MB), likely a logging gap rather than a true value.

## Verdict

**regression** — The 7L/640d shape is materially worse than the 9L/512d baseline at both fp16 EMA (−0.073 nats) and int6 gate (−0.107 nats). While the near-zero quantization gap is a positive signal, the capacity loss from removing 2 layers is not compensated by the width increase. This aligns with the leaderboard trend where top entries use 10–11 layers, not fewer. Depth appears more valuable than width at this parameter scale.

## Suggested follow-ups

- **Deeper-narrower sweep (11L/448d, 10L/480d):** The leaderboard SOTA clusters at 10–11 layers. Test whether adding depth beyond 9L improves BPB even at reduced width to stay within the parameter budget.
- **7L/640d + MLP expansion:** If the wider shape is retained for throughput reasons, try MLP 3× expansion to recover capacity through wider feedforward layers.
- **7L/640d + XSA:** Cross sliding attention could partially compensate for reduced depth by allowing more token interaction per layer.
- **Throughput measurement:** Capture tokens/sec for this shape vs baseline to quantify whether the throughput hypothesis held — if 7L is significantly faster, a longer training schedule or TTT budget could partially recover the gap.
