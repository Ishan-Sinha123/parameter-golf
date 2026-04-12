# 16MB in 10 Minutes: Adaptive Recurrence + Ternary QAT

## Executive Summary

To build a hyper-efficient language model under a 16MB footprint with a 10-minute training time, the most promising approach combines a hybrid Gated DeltaNet (GDN)/Attention architecture with Mixture-of-Recursions (MoR) for maximum parameter reuse and adaptive computation [executive_summary[6]][1]. This architecture, validated at the sub-100M scale by Olmo Hybrid, should be trained using a selective language modeling objective to maximize efficiency within the short training window [executive_summary[2]][2] [executive_summary[21]][3]. 

The critical component for meeting the size constraint is extreme Quantization-Aware Training (QAT) using a state-of-the-art framework like HESTIA, which enables stable ternary (~1.58-bit) quantization [executive_summary[12]][4]. This aggressive strategy is justified by research showing low-bit quantization disproportionately benefits models trained with limited compute [executive_summary[8]][5]. Techniques like In-Place Test-Time Training can further boost performance during the brief training phase without increasing the final model size [executive_summary[22]][6].

## North Star and Constraints: 16MB artifact, 10-minute train on 8xH100

Hitting 16MB requires ≈2-bit weights at ≈60M parameters, so adaptive compute (MoR) and low-bit QAT (HESTIA) must deliver quality fast. Training a model from scratch on a large dataset is impossible in this timeframe; the goal is only plausible if interpreted as fine-tuning on a small, targeted dataset [hardware_and_training_time_analysis[0]][1].

## 2024–2026 Evidence Map: Convergent recipe for ≤100M LMs

A convergent recipe emerges—3:1 GDN/attention hybrid + MoR routing + BLT input + HESTIA QAT—while several pieces remain unverified at tiny scale.

| Work (year/venue) | Authors | Link | Novelty | Small-scale applicability (≤100M) | Use for 16MB/10-min |
|---|---|---|---|---|---|
| Mixture-of-Recursions (2025, NeurIPS) | Min Bae et al. | [mixture_of_recursions_findings.source_link[0]][1] | Token-level routing over a shared recursive block; KV-sharing for active tokens | Demonstrated near 118–135M; mechanism scales down via shared stack | Adaptive depth boosts capacity without new params; try 4–6 loops, avg 2–3 |
| In-Place Test-Time Training (2026, arXiv) | Tianle Cai et al. | [in_place_test_time_training_findings.source_link[0]][6] | Treat MLP final projections as fast weights; chunk-wise updates | Not yet reported for byte/tiny LMs; easy to add | Ephemeral capacity during train/infer; zero artifact cost |
| HESTIA (2026, arXiv) | G. Wang et al. | [advanced_quantization_findings.source_link[0]][4] | Hessian-guided, annealed differentiable QAT for ternary/2-bit | Designed for extreme low-bit; tensor-wise scheduling fits shared blocks | Core to reach 16MB; use per-tensor annealing, clip FC2 activations |
| Rho‑1 (2024, NeurIPS) | Tongzhou Wang et al. | [selective_language_modeling_findings.source_link[0]][7] | Selective LM via token reweighting by a reference model | Replace ref model with n‑gram/entropy to stay cheap | Speed learning in 10 minutes; cap weights to avoid instability |
| BLT (2025, ACL) | Artidoro Pagnoni et al. | [byte_latent_transformer_findings.source_link[0]][8] | Entropy-based byte patches; variable granularity | Community 14M impl exists; ≤100M not rigorously benchmarked | Cut effective length; validate overhead at tiny scale |
| Low-Bit Favors Undertrained LLMs (2025, ACL) | Xiaoyu Ouyang et al. | [low_bit_quantization_for_undertrained_models_findings.source_link[0]][5] | Undertrained models gain more from low-bit; FC2 activations are W4A4 bottleneck | Directly matches 10-minute undertraining | Justifies aggressive ternary/2-bit; treat FC2 outliers |
| SEPARATE (2024, OpenReview) | Hanzhen Zhao et al. | [extreme_compression_findings.source_link[0]][9] | Seed-regenerated random projections for compression | No LM weight storage deployments yet | Experimental path to <10MB via seed + low-rank adapters |
| Olmo Hybrid (2026, AI2) | AI2 | [small_scale_hybrid_architecture_findings.source_link[0]][2] | 3:1 GDN:attention interleaving; strong at 60M/100M | Explicit small-scale wins | Use as the shared block inside MoR |

## Architecture Strategy: 3:1 GDN/attention inside Mixture-of-Recursions

A 4-layer shared block (3 GDN + 1 attention) recursed with token-level routing maximizes effective depth without growing parameters. 
* **60M/100M validation at AI2:** Use their ablations to set defaults (3:1 ratio, attention head/hidden sizes tuned for small models) [small_scale_hybrid_architecture_findings.paper_title[0]][2].
* **MoR routing and KV-sharing:** Limit max loops to 6; target average 2–3 to bound latency; route by token entropy/hardness; share KVs only for active tokens to reduce memory [mixture_of_recursions_findings.paper_title[0]][1].
* **Stability details:** Per-iteration affine calibrations (scale/bias) inside shared block; gradient checkpoint the shared block; optional stop-gradient on early recursions to control backprop cost.

## Input Representation: Byte Latent Transformer patches compress sequence length

Entropy-based patching reallocates compute to complex spans; validate that patching overhead doesn’t erase tokens/sec on 8×H100.
* Start with the 14M community BLT logic; integrate with router to share entropy signals [byte_latent_transformer_findings.summary_of_novelty[3]][10].
* Compare vs. compact BPE (e.g., 8–16k vocab) to ensure throughput parity; prioritize whichever yields higher tokens/sec at comparable BPB.

## Training and Quantization Plan: QAT-first schedule to reach ternary/2-bit

A short, staged QAT with HESTIA plus selective LM can converge useful capability in minutes.
* **HESTIA schedule:** Initialize from small PTQ or FP16; anneal temperature per tensor; push MLP/attention projections to ternary/2-bit first; keep layernorm/bias in higher precision if needed [advanced_quantization_findings.paper_title[0]][4].
* **FC2 activation handling:** Activation clipping, per-channel scales, or mixed-precision at FC2; monitor activation outliers [advanced_quantization_findings.source_link[1]][11].
* **Selective LM:** Compute n‑gram/entropy weights online; cap per-token weight (e.g., 3× median); decay cap over steps to stabilize [selective_language_modeling_findings.paper_title[0]][7].
* **In-Place TTT:** Enable chunk-wise fast-weight updates on MLP final projections during the 10-minute window; evaluate with/without to isolate durable vs. ephemeral gains [in_place_test_time_training_findings.paper_title[1]][12].

## Feasibility on 8×H100: Throughput targets and bottlenecks

The window is razor-thin; control QAT overhead (1.5–2×) and recurrence depth to keep updates high enough.
* **Throughput tactics:** Short sequences; micro-batch with grad accumulation; freeze router after a brief warmup; cap average loops to 2–3; use selective LM to focus gradients on hard tokens.
* **Polishing approach:** PTQ or 4-bit warm start, then 2–3 minutes of HESTIA annealing to land at ternary/2-bit.
* **Parallelism:** Exploit tensor/model parallel for the shared block; minimize per-loop synchronization.

## Risk Register and Countermeasures: Predefine off-ramps

| Risk | Why it matters | Mitigation |
|---|---|---|
| Quantization noise accumulates across recursions | Unstudied combo of MoR + ternary [key_challenges_and_research_gaps[0]][1] | Per-iteration affine calibrations; per-tensor annealing; stop-grad early loops; raise precision only for the most brittle tensors |
| BLT overhead exceeds savings at ≤100M | Tokens/sec collapse despite fewer patches | Switch to compact BPE; or static byte tokens with entropy-weighted loss only |
| Selective LM instability with n‑gram scorer | Noisy weights hurt convergence | Cap weights; moving-average normalization; immediate fallback to uniform loss if loss spikes |
| QAT time blow-up (1.5–2×) breaks SLA | Too few updates in 10 minutes | PTQ warm start + short HESTIA polish; freeze router early; reduce max loops to 3–4 |
| Seed-regenerated bases underperform | Speculative for inference [key_challenges_and_research_gaps[4]][9] | Keep as separate branch; strict success gate vs. ternary baseline |

## Implementation Roadmap: Two-week, decision-focused sprint

Parallelize prototypes; kill what doesn’t move BPB/tokens-sec by Day 7.
* **Days 1–3:** Stand up 3:1 GDN/attention shared block; integrate MoR routing; baseline FP16 training on byte tokens.
* **Days 4–6:** Add BLT patcher; measure wall-clock tokens/sec; A/B vs. compact BPE; keep faster path.
* **Days 7–9:** Integrate HESTIA; run PTQ→QAT polish; enforce 60M params; verify <16MB at 2-bit.
* **Days 10–11:** Add selective LM (n‑gram/entropy); cap weights; evaluate convergence speed.
* **Days 12–14:** Enable In-Place TTT; measure gains; finalize knobs (avg loops, clip levels); freeze recipe.

## Measurement Framework: Decide with numbers, not narratives

Track artifact size, stability, and real tokens/sec jointly.
* **Core metrics:** Bits-per-byte (BPB)/perplexity (byte-level), tokens/sec (end-to-end), final artifact size (MB), QAT convergence time, with/without-TTT deltas.
* **Stability:** Loss spikes, activation histograms (FC2), router distribution (avg loops).
* **Stop rules:** Kill any variant that drops tokens/sec >15% without a ≥3% BPB gain.

## Paper Quick-Reference: What’s novel and why it helps 16MB/10-min

| Title | Authors | Date/Venue | Link | Novelty | How to apply here |
|---|---|---|---|---|---|
| Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation | Min Bae et al. | Jul 2025, NeurIPS [mixture_of_recursions_findings.publication_info[0]][1] | [mixture_of_recursions_findings.source_link[0]][1] | Token-level depth routing over a shared stack; KV-sharing | Use a 4-layer shared block recursed up to 6 steps; route harder tokens deeper; keep params small |
| In-Place Test-Time Training | Tianle Cai et al. | Apr 2026, arXiv:2604.06169 [in_place_test_time_training_findings.publication_info[0]][12] | [in_place_test_time_training_findings.source_link[0]][6] | Updates MLP final projections as fast weights per chunk | Turn on during 10-minute train and optionally inference; discard fast weights from artifact |
| HESTIA: A Hessian-Guided Differentiable QAT for Extremely Low-Bit LLMs | G. Wang et al. | Jan 2026, arXiv:2601.20745 [advanced_quantization_findings.publication_info[0]][4] | [advanced_quantization_findings.source_link[0]][4] | Temperature-annealed, Hessian-guided ternary/2-bit training | Push projections to ternary/2-bit; per-tensor annealing; keep layernorm/bias higher precision if needed |
| Rho‑1: Not All Tokens Are What You Need | Tongzhou Wang et al. | Apr 2024, NeurIPS [selective_language_modeling_findings.publication_info[0]][7] | [selective_language_modeling_findings.source_link[0]][7] | Loss reweighting toward important tokens | Replace reference LM with n‑gram/entropy scorer; cap weights for stability |
| Byte Latent Transformer: Patches Scale Better Than Tokens | Artidoro Pagnoni et al. | 2025, ACL [byte_latent_transformer_findings.publication_info[0]][8] | [byte_latent_transformer_findings.source_link[0]][8] | Entropy-based variable-length byte patches | Reduce effective sequence length; validate overhead vs. BPE at ≤100M |
| Low-Bit Quantization Favors Undertrained LLMs | Xiaoyu Ouyang et al. | 2025, ACL [low_bit_quantization_for_undertrained_models_findings.publication_info[0]][5] | [low_bit_quantization_for_undertrained_models_findings.source_link[0]][5] | Undertrained models benefit more from low-bit; FC2 outliers matter | Justifies aggressive ternary/2-bit under 10-minute constraint; clip or mix-precision FC2 |
| SEPARATE: A Simple Low-rank Projection for Gradient Compression | Hanzhen Zhao et al. | 2024, OpenReview [extreme_compression_findings.publication_info[0]][9] | [extreme_compression_findings.source_link[0]][9] | Seed-regenerated random projections; no storage of big matrices | Experimental path: seed-born base + low-rank adapters to beat 16MB if needed |
| Olmo Hybrid: From Theory to Practice | AI2 | Mar 2026 [small_scale_hybrid_architecture_findings.publication_info[0]][2] | [small_scale_hybrid_architecture_findings.source_link[0]][2] | 3:1 GDN:attention wins at 60M/100M | Make this the shared block in MoR; inherit small-model hyperparams |

## Notes on Open Gaps to Monitor

Three red flags could derail the plan if not validated early:
* No peer-reviewed evidence yet for MoR + ternary QAT stability; test per-iteration calibrations immediately [key_challenges_and_research_gaps[0]][1].
* BLT at ≤100M lacks rigorous speed-quality tradeoff reporting; measure tokens/sec first [key_challenges_and_research_gaps[7]][13].
* N‑gram selective LM efficacy is unproven; keep a fast rollback to uniform loss [key_challenges_and_research_gaps[10]][7].

## What to Do Now (Immediate Next Steps)

* Fix the parameter budget at ~60M and build the 3:1 GDN/attention shared block within MoR; cap loops at 6 (avg 2–3).
* Implement HESTIA with per-tensor annealing; add FC2 activation clipping and mixed-precision fallback.
* Prototype BLT patching; A/B against compact BPE on tokens/sec; pick the faster.
* Add n‑gram/entropy selective LM with a strict weight cap and a rollback switch.
* Enable In-Place TTT for the 10-minute run and measure with/without effects.
* Enforce the 16MB export test (ternary/2-bit) by Day 9; kill any variant failing the size gate.

## References

1. *[2507.10524] Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation*. https://arxiv.org/abs/2507.10524
2. *Olmo Hybrid: From Theory to Practice*. https://allenai.org/papers/olmo-hybrid
3. *NeurIPS Oral Not All Tokens Are What You Need for Pretraining*. https://neurips.cc/virtual/2024/oral/98004
4. *HESTIA: A Hessian-Guided Differentiable Quantization ...*. https://arxiv.org/abs/2601.20745
5. *Low-Bit Quantization Favors Undertrained LLMs*. https://aclanthology.org/2025.acl-long.1555/
6. *[2604.06169] In-Place Test-Time Training*. https://arxiv.org/abs/2604.06169
7. *[2404.07965] Rho-1: Not All Tokens Are What You Need - arXiv*. https://arxiv.org/abs/2404.07965
8. *Byte Latent Transformer: Patches Scale Better Than Tokens*. https://aclanthology.org/2025.acl-long.453/
9. *A Simple Low-rank Projection for Gradient Compression in Modern ...*. https://openreview.net/forum?id=8HuLgtjqOD
10. *ianbarber/ttblt: A simplified implementation of Byte Latent ...*. https://github.com/ianbarber/ttblt
11. *[2505.14302] Scaling Law for Quantization-Aware Training*. https://arxiv.org/abs/2505.14302
12. *In-Place Test-Time Training - arXiv*. https://arxiv.org/html/2604.06169v1
13. *Byte Latent Transformer (BLT)*. https://huggingface.co/docs/transformers/en/model_doc/blt