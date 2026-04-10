# Parameter Golf — Mission

## The competition

**OpenAI Model Craft Challenge: Parameter Golf.** Train the best language
model that fits in a **16 MB artifact** and trains in **under 10 minutes on
8×H100 SXM**, evaluated by **bits-per-byte (BPB) on the FineWeb validation
set** (tokenizer-agnostic). Runs from 2026-03-18 to 2026-04-30.

This is an **L(N) optimization** — minimize loss given a fixed parameter
budget, unconstrained by data, compute, or architecture.

## Hard rules (disqualification if broken)

- **Artifact cap: 16,000,000 bytes decimal** (NOT 16 MiB = 16,777,216).
  Artifact = `code_bytes + compressed_model_bytes`. All counted code must
  live in `train_gpt.py`. The artifact must be self-contained — no
  external downloads, no network calls during evaluation.
- **Training budget: 10 minutes wallclock on 8×H100 SXM.** Exceeding the
  budget disqualifies the run.
- **Evaluation budget: 10 minutes additional wallclock on 8×H100 SXM.**
  This is on top of the training budget.
- **From scratch.** No pretrained weights. No distillation from larger
  teachers. No external checkpoints.
- **No validation leakage.** You cannot access validation data during
  training. You cannot compress validation bits into the 16 MB via a
  "paid prefix." You cannot cheat on test loss.
- **Test-time training is allowed ONLY on tokens you've already evaluated
  on.** Those tokens are already graded. TTT on unseen val tokens is
  forbidden.
- **Evaluation is reproducible.** Submissions run deterministically in the
  RunPod environment. Non-reproducible runs get disqualified.

## Record submission bar

A new SOTA must beat the existing SOTA by **at least 0.005 nats BPB** with
**p < 0.01** statistical significance (typically an average over 3 runs).
Systems-only speedups that don't change ML have the significance waived.
Submissions go as PRs adding a folder to `/records/<track>/<slug>/` with:
- `README.md` explaining the submission
- `submission.json` with author, val_bpb, metadata
- train logs proving the win
- `train_gpt.py` + deps that actually compile and run

## Current leaderboard context (as of 2026-04-10)

Best score: **1.1147** (11L AR Self-Gen GPTQ + XSA, abaybektursun). Recent
top entries stack these techniques on top of each other through a chain of
PRs:
- PR #1019: Self-gen GPTQ calibration + all-layer XSA (1.1147)
- PR #549:  LeakyReLU² + Legal Score-First TTT + Parallel Muon (1.1194)
- PR #374:  11L EMA + GPTQ-lite + warmdown3500 (1.1228)
- PR #287:  Partial RoPE + LN Scale + EMA + XSA4 (1.1248)
- PR #198:  11L XSA4 + EMA + Int6 MLP3x (1.1271)

Lesson: the frontier is **feature stacking**. Every winning run inherits
the previous winner's config and adds one or two new ingredients. This is
exactly what our recipe registry + `compose_recipe` pipeline must model.

Baseline: **1.2244** (9 layers, 512 dim, 1024 vocab, tied embeddings,
4 KV heads).

## Primary objective

**Minimize val_bpb** under all the hard rules above. Everything else
(quant gap, training stability, wallclock efficiency) only matters as a
lever toward lower BPB.

## Secondary objectives (tiebreakers / strategic)

1. **Statistical significance.** A 0.003-nat improvement is not a record.
   Plan ablations in groups of ≥3 seeds so wins are provable.
2. **Composability.** Prefer techniques that stack cleanly with the
   current SOTA chain over monolithic rewrites. The leaderboard history
   shows that's how records actually happen.
3. **Quantization gap.** Submissions typically quantize aggressively
   (int6, int5, ternary, 1-bit). A small fp16→quant gap is worth points.
4. **Wallclock headroom.** Finishing under budget leaves room for TTT at
   eval time, which is one of the strongest unlocks observed.

## Active research directions

**Currently hot on the leaderboard:**
- Attention variants: XSA (cross sliding attention) on last 4 / all layers,
  partial RoPE, GQA, sliding window
- Quantization: GPTQ (lite and full), int5/int6 mixed, ternary, 1-bit,
  self-generated calibration data
- Normalization / inits: layerwise LN scale, orthogonal init, EMA weights
- Activation variants: LeakyReLU², SmearGate, SwiGLU
- Depth/width: 10L, 11L with MLP 3× expansion is a common config
- Data / tokens: BigramHash(10240), trigram hash, sliding window eval
- Test-time training: LoRA TTT, Legal Score-First TTT, bias-only TTT
- Optimization: Parallel Muon, Muon WD, warmdown3500, warmup tuning

**Open requests from the maintainers (high-novelty targets):**
- 1-bit quantization (impl exists in non-record track: 1.1239)
- Ternary quantization (impl exists: 1.1570)
- JEPA
- Text diffusion
- H-net tokenization
- Universal transformer / depth recurrence
- Megakernels
- State-space models
- E2E test-time training
- Super long context for eval or training
- Learning adapters on random linear maps

Prefer the maintainers' open requests when choosing what to implement
speculatively — novelty has value even if the first run doesn't beat SOTA.

## What to ignore

- Pretrained weights, external teachers, retrieval from external corpora.
- Validation-set peeking in any form.
- Parameter blow-ups that don't fit under 16 MB even after aggressive
  quantization.
- Pure engineering speedups with no ML change (fine, but not novel —
  won't move the rule-waived records significantly unless they enable
  more training steps within the 10-min budget).
- "Generic LLM training tips" research queries. Frame everything in
  terms of 46M–200M models, from scratch, 10-min training budget,
  int6+ quantization, FineWeb BPB.

## How this mission is used by the autoresearch system

- **PR evaluation (`assess_pr`).** Every open PR on the fork gets read by
  Claude against this mission. Output is structured JSON: what technique,
  which files touched, env-var vs code change, expected BPB delta,
  whether it's composable with the current best baseline, novelty score.
- **Technique implementation (`implement_technique`).** When Parallel deep
  research surfaces a technique not yet in our `train_gpt.py`, Claude
  implements it behind a new env var flag on a feature branch, preserving
  the default baseline behavior. Branch is `auto/technique/<slug>`.
  **Never pushes to main.**
- **Recipe composition (`compose_recipe`).** Given the current best
  baseline recipe + a new feature, produce a stacked recipe that layers
  the feature on top. Tags the recipe with the canonical feature set.
- **Experiment reports (`write_report`).** On completion, Claude writes a
  markdown report: hypothesis → config → results vs baseline → verdict →
  suggested follow-ups. Stored under `experiment_logs/` and embedded into
  LanceDB for semantic retrieval.
- **Record reproduction (`reproduce_record`).** For each high-value entry
  in `records/`, Claude reads the submission bundle and produces a recipe
  so we can rerun it on our own infra and verify reproducibility.
- **Web research queries.** Every Parallel query is prepended with this
  mission so deep research is framed in terms of the competition rules,
  not generic ML advice.
- **Monitor agent.** A Claude monitor task periodically inspects training
  logs of active runs and can enqueue kill commands when it detects
  anomalies that rule-based detection misses (unexpected loss patterns,
  silent corruption, divergence before the 50-step window).
