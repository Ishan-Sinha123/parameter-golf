# Autoresearch Experiment Results

## Run 1: apr2 (single GPU, old dataset — climbmix-400b, 8192 BPE vocab)

**Note:** These results use a different dataset/tokenizer than the competition. BPB values are NOT comparable to the leaderboard.

### Best val_bpb: 0.983906 (started at 1.019646)

| commit | val_bpb | memory_gb | status | description |
|--------|---------|-----------|--------|-------------|
| 228791f | 1.019646 | 44.0 | keep | baseline (single GPU, 8L depth, ReLU^2) |
| c3a8896 | 1.009727 | 52.1 | keep | SwiGLU activation (replaces ReLU^2) |
| ffdbee1 | 1.013347 | 40.5 | discard | 4L 768w shallower wider model (worse than 8L 512w) |
| f1dabb0 | 0.998656 | 58.5 | keep | 6L 768w wider model with SwiGLU |
| 645ca98 | 1.062756 | 77.0 | discard | 6L 960w too wide (fewer steps and OOM-near) |
| ac97f6a | 1.036504 | 67.7 | discard | 6L 896w slightly wider (fewer steps hurt) |
| 8ea0fdb | 0.985880 | 58.2 | keep | halve batch to 2^18 (more steps per time) |
| 24f40df | 0.986370 | 29.5 | discard | batch 2^17 device_bs=64 (marginal, more noise) |
| 94045df | 0.989684 | 54.2 | discard | GQA 3 KV heads (worse than full MHA) |
| c4cb780 | 0.987761 | 58.2 | discard | all full-context windows (SSSL better) |
| 366bf1c | 0.994723 | 58.3 | discard | HEAD_DIM=64 12 heads (worse attention quality) |
| a369f45 | 0.988091 | 58.2 | discard | Muon LR 0.06 (higher than optimal) |
| 585d82f | 0.989647 | 58.2 | discard | 5% warmup (wastes time budget) |
| 56f3c89 | 0.990118 | 58.2 | discard | warmdown 30% (50% better) |
| 14b960b | 0.990185 | 58.2 | discard | no weight decay (WD=0.2 better) |
| 0697577 | 0.993066 | 57.0 | discard | no value embeddings (VE helps +0.007) |
| 47af3ba | 0.991071 | 58.2 | discard | softcap 30 (15 is better) |
| 1711a9b | 0.986518 | 58.2 | discard | embedding LR 1.0 (0.6 slightly better) |
| c754610 | 0.988141 | 58.2 | discard | RMSNorm before MLP output projection |
| 20c78d8 | 0.985781 | 58.2 | keep | FINAL_LR_FRAC=0.1 (tiny improvement) |
| 726c4a5 | 0.987234 | 51.4 | discard | MLP 3x ratio (less expressive per step) |
| bc16455 | 0.987223 | 67.4 | discard | 7L 768w (too slow per step) |
| 0ba0a58 | 0.985077 | 58.2 | keep | unembedding LR 0.008 (from 0.004) |
| dbc4117 | 0.986437 | 58.2 | discard | unembedding LR 0.012 (0.008 better) |
| 4d171ed | 0.983906 | 58.2 | keep | Muon LR 0.03 (from 0.04) |
| 838774a | 0.984849 | 58.2 | discard | Muon LR 0.02 (too low) |
| c9e0b5b | 0.984849 | 58.2 | discard | embedding LR 0.8 (0.6 better) |
| 8a5782b | 0.984306 | 58.2 | discard | cosine warmdown (linear better) |
| 8ed67c0 | 0.986465 | 59.5 | discard | VE on all layers (slower; alternating better) |
| 116da58 | 0.984258 | 58.2 | discard | weight decay 0.3 (0.2 better) |
| 6554380 | 0.985455 | 58.2 | discard | Adam beta1=0.9 (0.8 better) |
| 5e8e6d1 | 0.984098 | 58.2 | discard | TTT 3 steps lr=1e-4 (slight overfit) |
| 7f6c4aa | 0.983955 | 58.2 | discard | TTT 1 SGD step lr=5e-5 (not better) |
| df1d733 | 0.985483 | 29.8 | discard | DEVICE_BS=64 (slower; 128 better) |
| 2e3292a | 1.003823 | 58.2 | discard | parallel attn+MLP block (much worse) |
| 7817537 | 0.984496 | 58.2 | discard | softcap 20 (15 optimal) |

### Key findings (directional, not BPB-comparable):
- SwiGLU >> ReLU² (confirmed)
- 6L/768w sweet spot for this setup (wider than 8L/512d baseline)
- Batch size 2^18 > 2^19 (more optimizer steps helps)
- Muon LR 0.03 > 0.04 (slightly lower optimal)
- Unembedding LR 0.008 > 0.004
- Value embeddings help (~0.007 bpb)
- SSSL window pattern > all-full-context
- Softcap 15 > 20 > 30
- TTT shows marginal signal but doesn't clearly improve in this setup

---

## Run 2: apr2c (competition dataset — fineweb10B, sp1024 vocab)

**Note:** Agent ran on single GPU despite multi-GPU support (fell back to uv run). Results ARE on the competition dataset but with 5 min budget vs competition's 10 min.

### Experiments (from git commits):
1. `06f6812` — Baseline: unmodified train.py
2. `2290f96` — Fix DEVICE_BATCH_SIZE=64 for 8-GPU DDP
3. `c5245d8` — Switch MLP from ReLU-squared to SwiGLU activation
4. `de528d4` — Try 6L same-width (dim=512)
5. `46a45ba` — Increase DEVICE_BATCH_SIZE=128 on single GPU
6. `76b55a9` — Try 12L narrow (dim=384, 3 heads)

(Agent did not persist val_bpb results to TSV for this run.)

---

## Best Combo Experiment: 7L/MLP4x + SwiGLU + LoRA r16 QVK TTT

**Setup:** 8×H100, 600s training, competition dataset (fineweb10B sp1024)
**Config:** NUM_LAYERS=7, MLP_MULT=4, MLP_ACTIVATION=swiglu, TTT_MODE=lora, TTT_LORA_RANK=16, TTT_LORA_TARGETS=qvk

| Stage | val_bpb |
|-------|---------|
| Training (step 6022) | 1.1909 |
| Post-EMA | 1.1901 |
| Int6 roundtrip | 1.1950 |
| Int6 sliding window | 1.2144 |
| **TTT LoRA r16 QVK** | **1.1711** |

**Submission size:** 16.67MB (over 16MB limit — needs more aggressive pruning)
**SOTA comparison:** 1.1147 (0.056 gap)

Full log: `experiment_logs_fullscale/best_combo_7L_swiglu_ttt_qvk_r16.log`
