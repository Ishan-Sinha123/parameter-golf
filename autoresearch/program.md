# autoresearch

This is an experiment to have the LLM do its own research. The focus is on novel ideas for test-time training (TTT), tokenizer techniques, layer importance, and architecture variants.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr2`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **MANDATORY — Wait for ALL GPUs to be completely free**:

   **YOU MUST NOT LAUNCH ANY TRAINING UNTIL THIS CHECK PASSES. NO EXCEPTIONS.**

   The GPUs are shared with other experiments. Before running ANY training (including the baseline), you MUST verify the GPUs are idle:

   ```bash
   # Run this command. If it produces ANY output, GPUs are busy. DO NOT PROCEED.
   nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
   ```

   - If the command produces **any output at all** (any PIDs listed), the GPUs are IN USE. You MUST wait.
   - Poll by running the exact same command every 60 seconds: `sleep 60` then re-check.
   - Do NOT try to use a "free" GPU while others are busy. Do NOT use `CUDA_VISIBLE_DEVICES` to grab one GPU. ALL GPUs must be free before you start.
   - Do NOT skip this step. Do NOT assume GPUs are free. Do NOT proceed "just for the baseline." EVERY run requires free GPUs.
   - Only when the command returns **completely empty output** (zero lines, zero PIDs) may you proceed.

   This wait may take 10+ minutes. That is expected and fine. Be patient.

7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

The training script supports **single-GPU and multi-GPU** (DDP). It runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation).

**Launch command** (auto-detects GPU count):
```bash
GPU_COUNT=$(nvidia-smi -L | wc -l)
if [ "$GPU_COUNT" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$GPU_COUNT train.py > run.log 2>&1
else
    uv run train.py > run.log 2>&1
fi
```

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Known findings from prior research

These results have been validated in previous scaled experiments. Use them as a starting point — do not waste runs re-discovering these:

- **SwiGLU >> ReLU-squared**: Replacing `F.relu(x).square()` with a SwiGLU gated activation gave 0.56 bpb improvement AND 5% faster per step. Switch to SwiGLU early as your baseline.
- **Width beats depth**: Shallower models with wider MLPs (e.g. 4L with MLP 4x) outperform deeper narrower models (e.g. 8L with MLP 2x). The shallower model gets ~36% more training steps in the fixed time budget, more than compensating for lost depth.
- **Trigram hash embedding is redundant**: No improvement on top of large bigram embeddings. Don't bother unless you have a novel angle.
- **Do NOT try these (confirmed harmful or neutral)**: Residual gating (+0.13 bpb worse), LN inverse sqrt schedule (+0.04 worse), Gram Newton-Schulz orthogonalization (+0.61 worse), causal convolutions in attention (+0.01 worse / neutral).

## Research directions

You are exploring novel ideas. Here are four directions to investigate — these are open questions, not prescriptions. You should develop your own hypotheses and test them empirically.

### Direction A: Test-Time Training (TTT)

Can adapting the model at evaluation time improve val_bpb? Consider:
- Adding a per-document adaptation loop before scoring during eval
- LoRA adapters on attention and/or MLP layers
- Bias-only fine-tuning on validation prefixes
- Full fine-tuning of the last N layers
- Key unknowns: which layers to adapt, learning rate, number of gradient steps, chunk size, which parameter targets (Q/V projections? MLP? both?)

Note: TTT modifies the eval procedure, not training. Implement by adding a TTT wrapper around the `evaluate_bpb` call from `prepare.py`. The model trains identically — only eval changes.

### Direction B: Tokenizer & input representations

Can auxiliary input features improve the model beyond what BPE tokens provide? Consider:
- N-gram hash embeddings (bigram, character-level features)
- Learned byte-level features combined with token embeddings
- Factored or compressed embeddings
- You cannot change the tokenizer (it's in prepare.py), but you can add parallel embedding channels in train.py

### Direction C: Layer importance & pruning

Are all layers equally important? Investigate:
- Per-layer gradient norms during training
- Removing or shrinking specific layers
- Progressive layer dropout
- Non-uniform learning rates per layer
- Non-adjacent skip connections
- Varying layer widths (wider early layers, narrower late layers, or vice versa)

### Direction D: Architecture variants

What structural changes improve the model? Explore:
- Alternative activations beyond SwiGLU (GeLU, SiLU, Mish, etc.)
- Gated attention mechanisms
- Different normalization strategies
- Mixture of local/global attention patterns (varying the window pattern)
- GQA with fewer KV heads vs full MHA
- Different head dimensions
- U-Net style encoder-decoder skip weights

## Anti-cheating / no-lookahead constraint

**CRITICAL**: You must derive ALL conclusions from your own experiments. Do NOT use prior knowledge of "what works" from training data or external sources. Each idea is a hypothesis — implement it, run it, let val_bpb decide. If you believe something should work, state WHY in your commit message before running, then validate empirically. The value of this research is genuine discovery, not reproducing known results.

Specifically: Do not reference specific papers, known-good hyperparameters, or techniques as "known to work." Every architectural choice must be justified by experimental results in THIS setup. The only exception is the "Known findings" section above, which you may treat as given.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	switch to SwiGLU activation
c3d4e5f	1.005000	44.0	discard	add TTT with LoRA r=4 (no improvement)
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## Updating the research log

After every batch of experiments (roughly every 5-10 runs, or whenever you make a significant finding), update `RESEARCH_LOG.md` in the repo root with your findings. Include:
- What you tried and why
- The val_bpb result
- Whether you kept or discarded the change
- Any insights or surprises

Commit the research log update and push to the branch periodically (every ~5-10 experiments) so the human can track progress remotely:
```bash
git add RESEARCH_LOG.md
git commit -m "Update research log with latest findings"
git push origin autoresearch/<tag>
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr2`).

LOOP FOREVER:

1. **GPU check** (EVERY iteration, not just the first): Run `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`. If ANY output appears, sleep 60 and re-check. Do NOT proceed until output is empty.
2. Look at the git state: the current branch/commit we're on
3. Tune `train.py` with an experimental idea by directly hacking the code.
4. git commit
5. Run the experiment using the launch command above (redirect everything — do NOT use tee or let output flood your context)
6. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
7. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
8. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
9. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
10. If val_bpb is equal or worse, you git reset back to where you started
11. Every 5-10 experiments, update RESEARCH_LOG.md and push to the branch

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
