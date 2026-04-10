# seq512 half-batch

## Hypothesis

Halving both sequence length (1024 -> 512) and batch tokens (524K -> 262K) should yield roughly 2x more optimizer steps within the same wallclock budget, since each forward/backward pass is cheaper with shorter sequences. More gradient updates could improve convergence and final BPB even though fewer total tokens are processed per step.

## Configuration

| Parameter | Value |
|---|---|
| `TRAIN_SEQ_LEN` | `512` |
| `TRAIN_BATCH_TOKENS` | `262144` |
| Recipe | None (default baseline) |
| Model | 9L, 512 dim, 1024 vocab, tied embeddings, 4 KV heads |
| GPUs | 1 (screening run) |

No recipe was applied. Only the sequence length and batch token count were modified relative to the default baseline.

## Results

| Metric | Value | Notes |
|---|---|---|
| EMA val_bpb (screen) | **1.3151** | — |
| Int6 val_bpb (gate) | **1.3375** | — |
| Quant gap | **0.0000339** | Extremely small |
| Artifact size | 0.0 MB (not recorded) | — |
| Gate passed | Yes | — |
| Promoted | No | promote fields null |

Training log was not captured for this run.

### Context from companion experiment (exp002)

The exp002 report documents both runs side-by-side. Key metrics for exp001 from that report:

| Metric | exp001 (seq512) | Default (seq1024, warmup report) |
|---|---|---|
| Steps completed | ~2162 | ~1439 |
| Step avg | ~176 ms | ~334 ms |
| Tokens seen | ~566M | ~754M |
| EMA val_bpb | ~1.315 | ~1.289 |

The 1.5x increase in optimizer steps did not compensate for the 25% reduction in total tokens seen. Default seq1024 configuration outperforms seq512 by ~0.026 EMA BPB on similar 1-GPU screening runs.

## Verdict

**Regression.** Shorter sequences (512) with half-batch tokens produced more optimizer steps (~2162 vs ~1439) but worse BPB than the default seq1024 configuration (~1.315 vs ~1.289 EMA). Total token throughput matters more than update frequency at this model scale and training budget. The one bright spot is an exceptionally small quantization gap (0.00003), suggesting int6 compression works well with shorter-context weight distributions. The gate passed in absolute terms but the configuration is strictly inferior to the default.

## Suggested follow-ups

- **seq512 with full batch tokens (524K):** Keep shorter sequences but maintain the same token throughput per step. This would test whether shorter attention windows help or hurt when total tokens are held constant.
- **seq256 extreme:** Push the sequence-length/step-count tradeoff further to map out the Pareto frontier, even if diminishing returns are expected.
- **Mixed-length curriculum:** Start with seq512 for fast early convergence (more steps), then switch to seq1024 or seq2048 for the final phase to capture longer-range dependencies.
- **Combine with Muon optimizer:** The higher step count from seq512 might interact positively with Muon's update dynamics, which were tuned for ~1400-step budgets in other experiments.
- **seq512 on 8-GPU full budget:** The 1-GPU screening may understate the technique's potential; at 8-GPU scale the step count could exceed 10K, possibly changing the convergence dynamics.
