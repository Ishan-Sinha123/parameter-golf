# AutoResearch v2: Specification

> **Author**: Ishan Sinha | **Status**: Draft | **Last Updated**: April 2026
>
> A system for rapid, observable, self-healing ML experiment iteration under the Parameter Golf constraints.

---

## 1. Problem Statement

The current iteration loop is too slow and too fragile. Specific failure modes observed:

1. **Sequential bottleneck**: Track 3 ran 36 experiments serially. On 8×H100, this wastes 6–7 GPUs per experiment.
2. **Missing GPTQ gate**: The autoresearch loop recommended WD 0.2 (best training BPB) which was catastrophic post-quantization (+0.767 BPB gap). Any system that evaluates training loss alone will produce false positives on hyperparameters that affect weight distributions.
3. **No observability**: Experiments run as fire-and-forget shell commands. No live dashboards, no alerting on NaN/divergence, no way to compare runs mid-flight.
4. **No recovery**: A single OOM, NCCL timeout, or filesystem error kills the experiment and wastes the entire time slot.
5. **Manual bookkeeping**: Results are tracked in TSV files and scattered logs. No structured experiment registry, no automated git branching, no artifact versioning.

### 1.1 Design Goals

| Priority | Goal | Metric |
|----------|------|--------|
| P0 | 4× faster idea-to-signal throughput | Experiments evaluated per hour |
| P0 | No false positives from missing GPTQ | 100% of promoted experiments pass GPTQ gate |
| P1 | Live observability | Dashboard updates every 10s during active runs |
| P1 | Self-healing on transient failures | Auto-retry on OOM, NCCL timeout, filesystem errors |
| P2 | Structured experiment history | Every experiment queryable by config, result, lineage |
| P2 | One-command reproducibility | Any past experiment re-runnable from its git branch |

### 1.2 Non-Goals (for v2)

- Multi-node distributed training (single-node 8×H100 only)
- Automated paper/PR generation
- Bayesian optimization or neural architecture search (human-in-the-loop decisions on what to try)
- Integration with external experiment tracking platforms (W&B, MLflow) — we build lean and local

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CONTROL PLANE                           │
│                                                             │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Registry │  │ Scheduler │  │ Monitor  │  │ Dashboard │  │
│  │ (SQLite) │◄─┤  (Queue)  │──┤(Watchdog)│──┤  (Web UI) │  │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘  └───────────┘  │
│       │              │              │                        │
│       ▼              ▼              ▼                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  EXPERIMENT BUS                      │    │
│  │  (event log: started, step, eval, error, done)      │    │
│  └──────────┬──────────────────────────┬───────────────┘    │
│             │                          │                     │
└─────────────┼──────────────────────────┼─────────────────────┘
              │                          │
              ▼                          ▼
┌─────────────────────┐    ┌─────────────────────────┐
│    DATA PLANE        │    │      GPU PLANE           │
│                      │    │                          │
│  /workspace/         │    │  Slot 0: GPU 0,1         │
│    experiments/      │    │  Slot 1: GPU 2,3         │
│      <exp_id>/       │    │  Slot 2: GPU 4,5         │
│        config.yaml   │    │  Slot 3: GPU 6,7         │
│        train.log     │    │                          │
│        metrics.jsonl │    │  (or 2 slots × 4 GPU,    │
│        checkpoint/   │    │   or 1 slot × 8 GPU)     │
│        artifacts/    │    │                          │
│    db/registry.db    │    │                          │
│    git/ (branches)   │    │                          │
└──────────────────────┘    └──────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   TOOLING PLANE                              │
│          (integrated from JianYan11/parameter-golf)          │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │  Compute Est.    │  │ Terminal Playgnd. │  │  Agent Loop │ │
│  │ (h100_time_guess │  │ (generate_demo   │  │  (agent.md  │ │
│  │     .py)         │  │     .py)         │  │   workflow) │ │
│  │                  │  │                  │  │             │ │
│  │ Pre-flight time  │  │ Post-gate qual.  │  │ Records-    │ │
│  │ estimation for   │  │ sanity check on  │  │ mining &    │ │
│  │ QUEUE stage;     │  │ checkpoints;     │  │ hypothesis  │ │
│  │ PROMOTE budget   │  │ human-in-the-    │  │ generation  │ │
│  │ verification     │  │ loop before      │  │ for DEFINE  │ │
│  │                  │  │ PROMOTE          │  │ stage       │ │
│  └────────┬─────────┘  └────────┬─────────┘  └──────┬─────┘ │
│           │                     │                    │       │
│           ▼                     ▼                    ▼       │
│     Scheduler.queue()    GATE→PROMOTE gap     sweep/define   │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 Component Overview

| Component | Responsibility | Implementation |
|-----------|---------------|----------------|
| **Registry** | Experiment definitions, configs, results, lineage | SQLite + YAML |
| **Scheduler** | GPU allocation, experiment queuing, phase promotion | Python daemon |
| **Runner** | Execute training, GPTQ, evaluation stages | Subprocess + torchrun |
| **Monitor** | Health checks, NaN detection, auto-restart | Watchdog thread |
| **Dashboard** | Live metrics, comparison views, status board | Single-file HTML (served via `python -m http.server`) |
| **Logger** | Structured event log, git branching, artifact persistence | JSONL + git |
| **Retriever** | Pull relevant PRs, papers, Discord snippets | CLI tool (on-demand, not daemon) |
| **Compute Estimator** | Pre-flight wallclock estimation, PROMOTE budget verification | `scripts/h100_time_guess.py` (from JianYan11/parameter-golf) |
| **Terminal Playground** | Qualitative checkpoint inspection between GATE and PROMOTE | `scripts/generate_demo.py` (from JianYan11/parameter-golf) |
| **Agent Loop** | Records-mining, hypothesis generation, autonomous experiment iteration | `agent.md` workflow (from JianYan11/parameter-golf) |

---

## 3. Operational Doctrine

This section codifies the behavioral rules for both human experimenters and autonomous agents operating the system. These rules are adapted from `agent.md` §7.1–7.6 (JianYan11/parameter-golf) and are **load-bearing** — the rest of the spec assumes they are followed.

### 3.1 Session Setup (per session / "new day")

Every session begins with a structured setup ritual. This is not optional — skipping steps leads to fork divergence, stale data, or wasted runs on broken configs.

1. **Agree on a run tag** (e.g., `apr3` or `2026-04-03`). The branch `research/<tag>` must not already exist for a fresh session. Use `research/<tag>-gpu0` for parallel GPU tracks.
2. **Sync with upstream OpenAI repo.** This working copy is a fork; keep an explicit line to `openai/parameter-golf` so challenge rules and `train_gpt.py` do not silently diverge.
   - Ensure a remote exists: `git remote add upstream https://github.com/openai/parameter-golf.git`
   - `git fetch upstream` on `main`
   - Inventory drift: `git log --oneline main..upstream/main` and `git diff main...upstream/main -- train_gpt.py README.md`
   - **Decision point**: merge upstream into `main` before cutting `research/<tag>`, or record "we intentionally stay behind on X because Y."
3. **Create the branch**: `git fetch origin && git checkout main && git pull && git checkout -b research/<tag>`
4. **Read in-scope files** before editing — this is a checklist, not a suggestion:
   - `README.md` — challenge rules, FAQ, integrity constraints
   - `data/README.md` — data layout, tokenizer, export notes
   - `train_gpt.py` — `Hyperparameters`, model, training loop, eval, int8+zlib serialization
   - `agent.md` — this operational doctrine
5. **Verify data**: confirm `fineweb_train_*.bin`, `fineweb_val_*.bin`, and tokenizer exist at expected paths. If missing, run `python3 data/cached_challenge_fineweb.py --variant sp1024`.
6. **`results.tsv` needs no manual header** — `train_gpt.py` creates it automatically on the first successful run.
7. **Confirm with the human** that setup looks good, then enter the experiment loop.

### 3.2 Hard Guardrails: What You CAN and CANNOT Do

These rules apply to both human experimenters and autonomous agents. The official README FAQ always takes precedence over this spec.

**CAN (primary edit surface)**:
- Change `train_gpt.py` — architecture, optimizers, schedules, quantization logic, logging cadence — subject to artifact cap, integrity rules, and official time caps.

**CAN (supporting)**:
- Drive training via environment variables documented in `Hyperparameters` / §6.2.
- Add optional dependencies in the spirit of the README (e.g., FlashAttention).

**CANNOT**:
- Violate integrity / leaderboard rules: no unpaid validation "prefix" in the artifact; no sneaking validation into training; TTT only on validation tokens already evaluated; no network during evaluation.
- Bypass the artifact cap: code + int8+zlib ≤ 16,000,000 bytes.
- Bypass official training/eval time caps (600s each on 8×H100) when claiming a `records/` result.
- Redefine or game `val_bpb` (change eval to non-comparable semantics) without rigorous justification.
- Treat `data/` pipeline scripts as mutable during routine iteration (shard format, tokenizer contract, download scripts are read-only unless the human explicitly scopes a tokenizer/dataset experiment).

### 3.3 Simplicity Criterion

All else being equal, **simpler is better**. This is a binding design principle, not a suggestion.

- A small improvement in `val_bpb` that adds ugly complexity is **not worth keeping**. Weigh the complexity cost against the improvement magnitude.
- **Removing** code and getting equal or better `val_bpb` is a simplification win — **always keep**.
- A **0.001 better `val_bpb`** that adds 20 lines of hacky code? Probably not worth it.
- A **0.001 improvement from deleting** code? Definitely keep.
- An improvement of **~0 `val_bpb`** but much simpler code? Keep.

The simplicity criterion applies at PROMOTE time: when deciding whether to merge an experiment into the main line, consider both the BPB delta and the diff complexity. The dashboard's Experiment Detail view shows the config diff specifically to support this evaluation.

### 3.4 Hypothesis Discipline

Every experiment must have a **one-sentence, testable hypothesis** before it runs. This is enforced by the `hypothesis` field in the YAML config (§4.1 DEFINE stage, with hypothesis discipline per §3.4).

**Before or during each iteration**, use web search / paper sources (arxiv, blogs, docs) to find methods relevant to the current bottleneck. Form the hypothesis as a testable prediction, not a vague direction:

- Good: "Reducing MUON_LR from 0.04 to 0.03 will improve screen BPB by ~0.002 based on Track 4 A1 results"
- Bad: "Try different learning rates"

**Cite sources** in the `notes` field of the YAML config and in `EXPERIMENT_DESC` (which flows into `results.tsv`). This makes the idea trace auditable: any future experimenter can see where the hypothesis came from.

### 3.5 Agent-Mode Behavioral Rules

When the system runs in autonomous mode (`autoresearch.py agent-loop`, §10.6), the following rules govern agent behavior. These are adapted from `agent.md` §7.6 and are critical for overnight/unattended operation.

**Never stop to ask**: After session setup (§3.1) is confirmed and the human has picked the initial route, the agent does **not** pause to ask whether to continue. It keeps iterating until manually interrupted. The human may be asleep, away from the computer, and expects the agent to work indefinitely.

**Exception — route selection**: The 3-route triage (§4.1 DEFINE, agent-assisted) requires a human pick at session start and again when pivoting after a dead end. This is the **only** point where the agent pauses for input. In `--autonomous` mode, the agent picks the highest-evidence route automatically.

**Timeout policy**: Each SCREEN experiment should take ~3 minutes (180s training + overhead). If wall clock exceeds 2× the budget (~6 minutes) or the process hangs with no log output for 60s, kill it and treat as failure. The Monitor (§6.3) handles this automatically.

**Crash handling**:
- Trivial fix (typo, import error) → fix and re-run.
- Fundamentally broken idea → mark as REJECTED, log the crash, move on.
- `train_gpt.py` automatically logs `CRASH` status in `results.tsv` for uncaught exceptions.

**Keep/revert discipline**: If round-trip `val_bpb` improves (lower) AND `Total submission size int8+zlib` is under 16,000,000 AND the simplicity criterion (§3.3) is met → **keep** the commit. Otherwise → **revert** (`git checkout -- train_gpt.py` or reset). Do not stack noise — if BPB is flat or worse, revert rather than hoping it compounds.

**If stuck**: Re-search (§3.4), re-read `train_gpt.py` / README, re-scan `records/` for a new 3-route menu, combine prior near-misses, or try bolder architectural changes. The loop runs until the human interrupts.

---

## 4. Experiment Pipeline

Every experiment moves through a **seven-stage pipeline**. The key insight from Track 4 is that stages 1–3 must all complete before an experiment is considered "passed" — training loss alone is insufficient. The INSPECT stage (integrated from JianYan11/parameter-golf's `generate_demo.py`) adds a human-in-the-loop qualitative checkpoint before committing to the expensive PROMOTE run.

```
DEFINE → QUEUE → SCREEN → GATE → INSPECT → PROMOTE → SUBMIT
                   │         │       │          │
                   │         │       │          └─ Full-budget run (600s, 8 GPU)
                   │         │       │             + sliding-window eval
                   │         │       │             + artifact size check
                   │         │       │
                   │         │       └─ Qualitative coherence check
                   │         │          (human streams tokens from checkpoint,
                   │         │           optional — auto-skipped in batch mode)
                   │         │
                   │         └─ GPTQ int6 quantization
                   │            + int6 roundtrip BPB
                   │            + artifact size estimate
                   │            (REJECT if quant gap > threshold)
                   │
                   └─ Fast training (180s, 2 GPU)
                      + training BPB
                      + EMA BPB
                      (REJECT if training BPB > baseline + margin)
```

### 4.1 Stage Details

#### DEFINE

Experiment definitions can come from two sources:

**Manual YAML config** — the experimenter writes directly:

```yaml
# experiments/exp_042_muon_lr_sweep.yaml
id: exp_042
name: "Muon LR 0.035"
parent: exp_000_baseline  # lineage tracking
hypothesis: "LR 0.03 was best in Track 4; test intermediate value"
category: hyperparameter
priority: high

# Only specify overrides. Everything else inherits from parent.
env_overrides:
  MUON_LR: "0.035"

# Which stages to run (default: all)
stages: [screen, gate]

# Expected behavior
expected_direction: negative  # expect BPB to decrease
reject_if_worse_by: 0.05     # auto-reject if >0.05 worse than parent
```

**Agent-assisted definition** — using the `agent.md` records-mining workflow (integrated from JianYan11/parameter-golf). This is the recommended path for sessions where the experimenter wants to explore new directions rather than run pre-planned sweeps:

1. The agent scans `records/track_10min_16mb/*/submission.json`, README files, and `track_non_record_16mb/` for evidence-backed ideas.
2. For each interesting submission, the agent extracts `val_bpb`, `bytes_total`, and any ablation tables.
3. The agent proposes **exactly three routes** that differ in **mechanism** (e.g., architecture vs quant vs optimizer) — not three variations of the same knob.
4. For each route, the agent argues "why it might work here" with **data tied to `records/`**: submission name, numeric `val_bpb` gap vs baseline, artifact size headroom, and caveats (different tokenizer, int6+zstd vs int8+zlib, etc.).
5. The human picks route A/B/C. The agent generates the corresponding YAML config(s) and drops them in `queue/`.

This workflow replaces the free-form "what should we try next?" decision with a structured, evidence-backed funnel. The agent's hypothesis becomes the `hypothesis` field in the YAML config, and the cited records become the `notes` field for audit trail.

```bash
# Agent-assisted define (interactive):
python autoresearch.py agent-define --parent exp_000_baseline

# This triggers the records-mining → 3 routes → human pick → YAML generation flow.
# The agent uses the env_overrides vocabulary from train_gpt.py Hyperparameters
# (see agent.md §5 for the full list: MUON_LR, NUM_LAYERS, MLP_MULT, etc.)
```

#### QUEUE

The scheduler validates the config, assigns an experiment ID, creates a git branch (`exp/exp_042`), runs a **compute pre-flight check**, and adds it to the priority queue.

**Compute pre-flight** (integrated from `scripts/h100_time_guess.py`): Before queuing, the scheduler estimates whether the experiment will fit within its wallclock budget. If prior logs exist for the parent experiment or a similar config, the estimator parses `train_time:...ms` and `step:N/M` from those logs and uses the TFLOPS-ratio method to predict wallclock on the target hardware. If the estimate exceeds the stage budget (180s for SCREEN, 600s for PROMOTE) by more than 20%, the experiment is flagged with a warning in the dashboard but still queued — the estimate is napkin math, not a hard gate.

```python
# In the scheduler's queue_experiment() method:
from scripts.h100_time_guess import estimate_h100_time

def queue_experiment(config):
    # ... validate config, assign ID, create git branch ...

    # Pre-flight: estimate wallclock from parent's logs if available
    parent_log = f"experiments/{config.parent_id}/train.log"
    if os.path.exists(parent_log):
        est_seconds = estimate_h100_time(parent_log, stage=config.stages[0])
        budget = 180 if config.stages[0] == "screen" else 600
        if est_seconds > budget * 1.2:
            config.warnings.append(
                f"Compute estimate: {est_seconds:.0f}s vs {budget}s budget "
                f"(parent log suggests this config may exceed wallclock)"
            )
            emit_event(config.id, "warning", {"compute_estimate": est_seconds})

    # ... add to priority queue ...
```

Priority ordering:

1. **Critical**: Experiments that test interactions with GPTQ (WD, LR, anything touching weight magnitudes)
2. **High**: Experiments confirming findings from other tracks
3. **Normal**: New ideas
4. **Low**: Speculative / high-risk ideas

#### SCREEN (180s, 2 GPU)

Fast training run. The runner:
1. Checks out the experiment's git branch
2. Sets env vars from `env_overrides` + `CUDA_VISIBLE_DEVICES` for the assigned GPU slot
3. Runs `torchrun --nproc_per_node=2` with `MAX_WALLCLOCK_SECONDS=180`
4. Streams `metrics.jsonl` (one JSON object per training step)
5. On completion: records training BPB, EMA BPB, step count, ms/step

**Auto-reject criteria**:
- Training BPB > parent's screen BPB + `reject_if_worse_by`
- NaN/Inf in loss at any point
- OOM (after 1 retry with reduced batch size)

#### GATE (GPTQ + int6 roundtrip eval)

This is the **critical stage** that Track 3 was missing. The runner:
1. Loads the screen checkpoint
2. Runs GPTQ int6 quantization with AR self-gen calibration
3. Computes int6 roundtrip BPB
4. Estimates artifact size (int6 weights + LZMA compression)

**Auto-reject criteria**:
- Quantization gap > 2× parent's quantization gap (catches WD 0.2-type failures)
- Artifact size > 15.5 MB (leaves 0.5 MB headroom)
- Int6 roundtrip BPB > parent's int6 roundtrip BPB + margin

**This stage runs on 1 GPU** (GPTQ is single-GPU). It can run in parallel with other experiments' screen stages.

#### INSPECT (optional, human-in-the-loop)

Integrated from JianYan11/parameter-golf's `scripts/generate_demo.py`. This is a **qualitative coherence check** — the experimenter streams tokens from the screen checkpoint to visually verify that the model produces reasonable text. This catches failure modes that BPB alone can miss: degenerate repetition, mode collapse to a narrow vocabulary, or garbage output from subtle training bugs.

**When to use**: INSPECT is optional and intended for experiments that change architecture or activation functions — cases where a good BPB number could mask qualitative degradation. For pure hyperparameter sweeps (LR, WD, batch size), INSPECT can be skipped.

**How it works**:
```bash
# The dashboard surfaces a one-click "Inspect" button for GATE-passed experiments.
# Under the hood, it runs:
python scripts/generate_demo.py \
    --checkpoint experiments/$EXP_ID/screen_checkpoint.pt \
    --tokenizer ./data/tokenizers/fineweb_1024_bpe.model \
    --prompt "The most important thing about" \
    --max-new-tokens 128 \
    --temperature 0.9 \
    --plain  # machine-parseable output for dashboard embedding
```

**Behavior**:
- In **interactive mode** (human at terminal): opens a prompt loop for free-form exploration. The ASCII mascot banner and colored output make this a pleasant developer experience.
- In **batch mode** (automated pipeline): auto-skipped unless the config specifies `inspect: true`. When run in batch mode, the output is captured to `experiments/$EXP_ID/inspect_samples.txt` and surfaced in the dashboard's Experiment Detail view.
- **No GPU slot required**: runs on CPU by default (the screen checkpoint is small enough). Does not block other experiments.
- **Not a gate**: INSPECT never auto-rejects. It is purely informational for the human deciding whether to PROMOTE.

#### PROMOTE (600s, 8 GPU)

Full-budget training for experiments that pass both screen and gate:
1. Full `torchrun --nproc_per_node=8` with `MAX_WALLCLOCK_SECONDS=600`
2. GPTQ int6 quantization
3. LZMA compression + artifact size verification
4. Int6 roundtrip eval
5. Sliding-window eval (stride 64)

Only 1 promote run at a time (requires all 8 GPUs).

#### SUBMIT

Manual step. The experimenter reviews the promote results, writes a README, and creates the PR. Not automated.

### 4.2 Parallelism Strategy

The key throughput lever. Default GPU allocation for screening:

```
8 GPUs → 4 slots × 2 GPUs each

Slot 0: GPU 0,1  →  Experiment A (screen)
Slot 1: GPU 2,3  →  Experiment B (screen)
Slot 2: GPU 4,5  →  Experiment C (screen)
Slot 3: GPU 6,7  →  Experiment D (screen)

Wall-clock: 180s per batch of 4 experiments
Throughput: ~80 experiments/hour (screen only)
           ~20 experiments/hour (screen + gate)
```

Compare to Track 3: 36 experiments in ~3 hours = 12/hour. This is a **6× throughput improvement** for screening alone.

When a promote run is queued, the scheduler drains all screen slots and consolidates to 8 GPU.

### 4.3 The Scaling Problem: How to Interpret Screen Results

The user correctly identifies that scaling relationships are non-linear. Our Track 4 data quantifies this:

| Experiment | 180s Train BPB | 180s Int6 BPB | 1200s Train BPB | 1200s Int6 BPB |
|-----------|---------------|--------------|----------------|---------------|
| Baseline  | ~1.338        | ~1.705       | 1.138          | 1.141         |

The **absolute gap** between 180s and 1200s is ~0.56 BPB (training) and ~0.20 (post-GPTQ at full budget). But the **ranking** is preserved: in every case where we have both 180s and full-budget data, the experiment that wins at 180s also wins at full budget.

**Design decision**: The system does NOT attempt to predict final BPB from screen BPB. Instead:

1. **Screen phase**: Rank experiments relative to each other. Reject obvious failures.
2. **Gate phase**: Catch GPTQ-hostile configurations. This is a pass/fail gate, not a ranking.
3. **Promote phase**: Get the real number. Only 2–3 experiments per session should reach this stage.

For experiments where even ranking fidelity is uncertain (e.g., techniques that specifically benefit from longer training, like TTT), the config can specify `stages: [promote]` to skip directly to full-budget.

---

## 5. Data Model

### 5.1 Experiment Registry (SQLite)

```sql
CREATE TABLE experiments (
    id              TEXT PRIMARY KEY,       -- "exp_042"
    name            TEXT NOT NULL,
    parent_id       TEXT REFERENCES experiments(id),
    hypothesis      TEXT,
    category        TEXT,                   -- architecture, hyperparameter, evaluation, ttt
    priority        TEXT DEFAULT 'normal',  -- critical, high, normal, low
    status          TEXT DEFAULT 'defined', -- defined, queued, screening, gating, promoting, done, rejected, failed
    rejection_reason TEXT,

    -- Config
    env_overrides   TEXT,                   -- JSON dict
    git_branch      TEXT,
    script_path     TEXT DEFAULT 'train_gpt.py',

    -- Screen results
    screen_steps        INTEGER,
    screen_ms_per_step  REAL,
    screen_train_bpb    REAL,
    screen_ema_bpb      REAL,
    screen_gpu_count    INTEGER,
    screen_wallclock_s  REAL,

    -- Gate results
    gate_int6_bpb       REAL,
    gate_quant_gap      REAL,
    gate_artifact_mb    REAL,
    gate_passed         BOOLEAN,

    -- Promote results
    promote_train_bpb   REAL,
    promote_ema_bpb     REAL,
    promote_int6_bpb    REAL,
    promote_sw_bpb      REAL,
    promote_artifact_mb REAL,
    promote_steps       INTEGER,

    -- Metadata
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at      DATETIME,
    completed_at    DATETIME,
    notes           TEXT
);

CREATE TABLE events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id   TEXT REFERENCES experiments(id),
    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,
    event_type      TEXT,    -- step, eval, error, stage_change, rejection
    payload         TEXT     -- JSON: {step, loss, bpb, lr, ...}
);

CREATE INDEX idx_events_exp ON events(experiment_id, timestamp);
CREATE INDEX idx_experiments_status ON experiments(status);
```

#### 5.1.1 TSV Migration (Bootstrapping from Existing Results)

Both `train_gpt.py`'s built-in TSV logger and the Karpathy-style `autoresearch/` workflow produce `results.tsv` files with historical experiment data. To bootstrap the SQLite registry from prior work, the system provides a one-time migration tool:

```bash
# Import existing results.tsv into the registry as historical experiments
python autoresearch.py import-tsv autoresearch/results.tsv --tag "autoresearch_v1"

# This creates registry entries with:
#   - id: auto-generated from commit hash (e.g., "legacy_228791f")
#   - status: "done" (for keep) or "rejected" (for discard/crash)
#   - screen_train_bpb: from val_bpb column
#   - notes: from description column
#   - git_branch: looked up from commit hash if available
```

The TSV format (tab-separated: `commit`, `val_bpb`, `memory_gb`, `status`, `description`) is the same between `train_gpt.py`'s auto-logger and the `autoresearch/program.md` convention, so both sources can be imported with the same command. The `EXPERIMENT_DESC` env var (set by the runner, §6.2) ensures new experiments continue to produce TSV entries as a redundant backup alongside SQLite.

### 5.2 Metrics Stream (JSONL)

Each training step emits one line to `experiments/<id>/metrics.jsonl`:

```json
{"t": 1712345678.123, "step": 42, "train_loss": 3.456, "lr": 0.025, "ms_step": 115.2, "gpu_mem_gb": 38.4, "tokens_seen": 1234567}
```

Evaluation events:

```json
{"t": 1712345699.456, "event": "eval", "stage": "screen", "val_bpb": 1.279, "ema_bpb": 1.293}
{"t": 1712345720.789, "event": "gptq", "int6_bpb": 1.394, "quant_gap": 0.115, "artifact_mb": 7.30}
```

### 5.3 Filesystem Layout

```
/workspace/autoresearch_v2/
├── autoresearch.py          # Main daemon (scheduler + monitor + runner)
├── agent.md                 # Agent loop workflow (from JianYan11/parameter-golf)
├── dashboard.html           # Single-file live dashboard
├── retriever.py             # Research retrieval CLI
├── db/
│   └── registry.db          # SQLite experiment registry
├── experiments/
│   ├── exp_000_baseline/
│   │   ├── config.yaml
│   │   ├── metrics.jsonl
│   │   ├── train.log        # Raw stdout/stderr
│   │   ├── results.tsv      # Per-experiment TSV (from train_gpt.py auto-logger)
│   │   ├── screen_checkpoint.pt
│   │   ├── gate_results.json
│   │   ├── inspect_samples.txt  # Token samples from generate_demo.py (optional)
│   │   └── artifacts/       # Compressed model, submission files
│   ├── exp_042_muon_lr/
│   │   └── ...
│   └── ...
├── queue/                   # Pending experiment configs (YAML)
├── templates/
│   ├── architecture.yaml
│   ├── hyperparameter.yaml
│   └── ttt.yaml
└── scripts/
    ├── generate_demo.py     # Terminal token streaming (from JianYan11/parameter-golf)
    ├── h100_time_guess.py   # Compute time estimator (from JianYan11/parameter-golf)
    ├── setup_git_branches.sh
    └── sync_logs.sh
```

---

## 6. Component Specifications

### 6.1 Scheduler

The scheduler is the central coordinator. It runs as a single Python process and manages the experiment queue and GPU allocation.

**State machine per experiment**:

```
DEFINED ──queue──► QUEUED ──assign_gpus──► SCREENING ──pass──► GATING ──pass──► INSPECTING ──approve──► PROMOTING ──done──► DONE
                                              │                   │                 │                       │
                                              ▼                   ▼                 │                       ▼
                                           REJECTED            REJECTED          (auto-skip            FAILED
                                           (bad BPB)        (bad quant gap)    if inspect:false)    (infra error)
```

Note: INSPECTING is optional. If the config does not specify `inspect: true`, or the pipeline is running in batch mode, the state transitions directly from GATING → PROMOTING. The INSPECT stage never auto-rejects; it only pauses for human review.

**GPU slot manager**:

```python
class GPUSlotManager:
    """
    Manages GPU allocation across concurrent experiments.

    Default: 4 slots of 2 GPUs each for screening.
    Promotion: drains all slots, allocates all 8 GPUs.
    Gate: uses 1 GPU (can overlap with screen slots).
    """

    def __init__(self, total_gpus=8, screen_gpus_per_slot=2):
        self.total_gpus = total_gpus
        self.screen_gpus = screen_gpus_per_slot
        self.slots = {}  # slot_id -> experiment_id or None
        self.mode = "screening"  # or "promoting"

    def request_slot(self, experiment_id, stage):
        """Returns list of GPU indices, or None if no slot available."""
        ...

    def release_slot(self, experiment_id):
        """Free GPUs when experiment completes or fails."""
        ...

    def consolidate_for_promote(self):
        """Wait for all screen slots to drain, switch to promote mode."""
        ...
```

**Scheduling loop** (pseudocode):

```python
while True:
    # 1. Check for new configs in queue/ directory
    new_configs = scan_queue_directory()
    for config in new_configs:
        register_experiment(config)  # includes compute pre-flight via h100_time_guess

    # 2. Check for completed/failed experiments
    for exp in running_experiments():
        if exp.process.poll() is not None:
            handle_completion(exp)  # includes post-screen h100 time estimate

    # 3. Run INSPECT for gate-passed experiments (CPU-only, non-blocking)
    for exp in gate_passed_pending_inspect():
        if exp.config.get("inspect", False):
            run_inspect(exp)  # generate_demo.py → inspect_samples.txt
        else:
            advance_to_promote_queue(exp)  # auto-skip INSPECT

    # 4. Schedule next experiments
    if promote_queue and all_screen_slots_empty():
        schedule_promote(promote_queue.pop())
    else:
        for pending in priority_sorted(screen_queue):
            slot = gpu_manager.request_slot(pending.id, "screen")
            if slot:
                launch_screen(pending, slot)

    # 5. Health checks
    for exp in running_experiments():
        check_health(exp)  # NaN, OOM, stall detection

    sleep(5)
```

### 6.2 Runner

Wraps `torchrun` invocations with env-var injection, log capture, and metrics streaming. The env-var vocabulary is defined by `train_gpt.py`'s `Hyperparameters` class; the canonical reference for all supported overrides is in `agent.md` §5 (integrated from JianYan11/parameter-golf).

**Key behaviors**:
- Sets `CUDA_VISIBLE_DEVICES` based on assigned slot
- Redirects stdout/stderr to `train.log` (never `tee` — per agent.md §5, avoid flooding agent context or dashboard with raw output)
- Parses training output to extract metrics → writes `metrics.jsonl`
- On OOM: retries once with `DEVICE_BATCH_SIZE` halved
- On NCCL timeout: retries once after 10s delay
- On NaN loss: kills immediately, marks as rejected

**Env-var categories** (from `agent.md` §5, kept in sync with `Hyperparameters`):
- **Data / run**: `DATA_PATH`, `TOKENIZER_PATH`, `RUN_ID`, `SEED`
- **Budget / schedule**: `MAX_WALLCLOCK_SECONDS`, `ITERATIONS`, `WARMUP_STEPS`, `WARMDOWN_ITERS`, `TRAIN_BATCH_TOKENS`, `TRAIN_SEQ_LEN`
- **Logging / eval cadence**: `TRAIN_LOG_EVERY`, `VAL_LOSS_EVERY`, `VAL_BATCH_SIZE`
- **Model shape**: `VOCAB_SIZE`, `NUM_LAYERS`, `MODEL_DIM`, `NUM_HEADS`, `NUM_KV_HEADS`, `MLP_MULT`, `TIE_EMBEDDINGS`, `ROPE_BASE`, `LOGIT_SOFTCAP`
- **Optimizer**: `EMBED_LR`, `MATRIX_LR`, `HEAD_LR`, `MUON_MOMENTUM`, `GRAD_CLIP_NORM`
- **Results tracking**: `EXPERIMENT_DESC`, `RESULTS_TSV_PATH`, `DISABLE_RESULTS_TSV`

The runner also sets `EXPERIMENT_DESC` to the experiment's `name` field from the YAML config, so `train_gpt.py`'s built-in TSV logger captures the description automatically. This provides a redundant results trail alongside the SQLite registry.

**Screen runner**:
```bash
CUDA_VISIBLE_DEVICES=$GPUS \
MAX_WALLCLOCK_SECONDS=180 \
EXPERIMENT_DESC="$EXP_NAME" \
RESULTS_TSV_PATH=experiments/$EXP_ID/results.tsv \
$ENV_OVERRIDES \
torchrun --standalone --nproc_per_node=$NGPUS train_gpt.py \
    > experiments/$EXP_ID/train.log 2>&1
```

**Gate runner**:
```bash
# Runs GPTQ + int6 eval on the screen checkpoint
# Single GPU, ~2-3 minutes
CUDA_VISIBLE_DEVICES=$GPU \
python gptq_gate.py \
    --checkpoint experiments/$EXP_ID/screen_checkpoint.pt \
    --output experiments/$EXP_ID/gate_results.json
```

**Log parsing** (from `agent.md` §6): After each run, the runner extracts results using the same grep patterns the agent uses interactively:
```bash
grep 'final_int8_zlib_roundtrip' experiments/$EXP_ID/train.log  # primary BPB
grep 'Total submission size int8+zlib' experiments/$EXP_ID/train.log  # artifact size
```

**Post-run compute verification**: After SCREEN completes, the runner invokes `h100_time_guess.py check` on the train log to estimate whether the same config would fit within the PROMOTE budget on 8×H100. This estimate is stored in the registry and surfaced in the dashboard's Comparison Table.

```python
# In the runner's handle_screen_completion():
est_h100_seconds = h100_time_guess.check_log(
    f"experiments/{exp_id}/train.log",
    local_tflops=gpu_slot_manager.slot_tflops(slot_id)
)
db.update(exp_id, screen_h100_estimate_s=est_h100_seconds)
```

### 6.3 Monitor (Watchdog)

A thread within the scheduler that inspects running experiments every 5 seconds.

**Health checks**:

| Check | Signal | Action |
|-------|--------|--------|
| NaN/Inf loss | `train_loss` is NaN or > 100 | Kill process, mark REJECTED |
| Stall | No new log line for 60s | Kill process, retry once |
| OOM | `CUDA out of memory` in stderr | Kill, retry with smaller batch |
| NCCL timeout | `NCCL timeout` in stderr | Kill, retry after 10s |
| Divergence | Loss increasing for 50+ consecutive steps | Warn in dashboard (don't kill) |
| GPU utilization | `nvidia-smi` shows <10% util for 30s | Warn in dashboard |

**Auto-retry budget**: Each experiment gets at most 2 retries. After that, it's marked FAILED with the error log preserved.

### 6.4 Dashboard

A single HTML file served via `python -m http.server 8080`. Reads `metrics.jsonl` files and `registry.db` directly. No build step, no framework.

**Views**:

1. **Status Board**: All experiments with their current stage, status (color-coded), key metrics. Sortable by BPB, priority, timestamp.

2. **Live Training Curves**: For each running experiment, a real-time loss curve updated every 10s. Overlaid with the baseline curve for visual comparison. X-axis: wall-clock seconds (not steps), because the whole point is time-constrained.

3. **Comparison Table**: Side-by-side metrics for selected experiments. Columns: config diff, screen BPB, gate BPB (int6), quant gap, artifact size, steps, ms/step.

4. **Experiment Detail**: Full config, training log tail, metrics, git branch, parent lineage.

5. **GPU Utilization**: Per-GPU memory and compute utilization from `nvidia-smi`.

**Implementation approach**:
- Single `dashboard.html` with embedded CSS and JS
- Polls `/api/status` endpoint every 10s (served by a tiny Python HTTP handler in the scheduler)
- Uses `<canvas>` for training curves (no charting library dependency)
- Fallback: if no HTTP server, can be opened directly and reads JSONL files via `fetch('file://...')`

### 6.5 Logger

**Structured logging**: All events go to both the SQLite `events` table and experiment-specific `metrics.jsonl`.

**Git integration**:

```bash
# On experiment creation:
git checkout -b exp/$EXP_ID
git commit --allow-empty -m "exp($EXP_ID): $NAME - $HYPOTHESIS"

# On experiment completion:
git add experiments/$EXP_ID/
git commit -m "exp($EXP_ID): result=$STATUS screen_bpb=$BPB gate_int6=$INT6"

# On promotion:
git tag "promote/$EXP_ID" -m "Promoted: sw_bpb=$SW_BPB artifact=$MB"
```

Every experiment is a git branch. Every result is a commit. The full history is `git log --all --oneline`.

**Log persistence**: `sync_logs.sh` runs every 5 minutes via cron, rsyncing the entire `autoresearch_v2/` directory to a backup location (second disk, or `rsync` to a cheap storage VM if available).

### 6.6 Research Retriever

A CLI tool, not a daemon. Invoked manually when exploring new ideas. The retriever has two complementary modes: **external search** (PRs, papers) and **local records-mining** (integrated from `agent.md` §7.4–7.5).

**External search** (GitHub + arxiv):

```bash
# Search competition PRs for relevant techniques
python retriever.py search "mixed precision quantization int5 int6"

# Fetch and summarize a specific PR
python retriever.py pr 1105

# Search arxiv for relevant papers
python retriever.py arxiv "test-time training language models quantization"
```

**Local records-mining** (from `agent.md` §7.5 step 1): Scans `records/` in the parameter-golf repo for evidence-backed ideas. This is the same workflow used by the agent-assisted DEFINE stage (§4.1), but exposed as a standalone CLI for human use:

```bash
# Mine records/ for ideas relevant to a bottleneck
python retriever.py records --bottleneck "quantization gap too large"

# This scans records/track_10min_16mb/*/submission.json and README files,
# extracts val_bpb, bytes_total, and ablation tables, and returns
# structured summaries ranked by relevance to the bottleneck query.

# Propose 3 experiment routes based on records (same logic as agent-define)
python retriever.py propose --parent exp_000_baseline --bottleneck "architecture"
```

**Implementation**: External search uses GitHub API to search PR titles/descriptions in `openai/parameter-golf`, and arxiv API for paper search. Records-mining uses local file scanning with structured extraction of `submission.json` fields, README tables, and log greps. Both modes cache results locally in `db/retriever_cache.db`. Returns structured summaries (title, author, key technique, reported BPB improvement if available).

**Integration with agent.md workflow**: The retriever's `propose` subcommand implements the same 3-route proposal logic described in `agent.md` §7.5: it mines records for evidence, proposes three mechanistically-diverse routes, and cites specific submissions with numeric `val_bpb` gaps. The key difference from pure `agent.md` usage is that the retriever can also query external sources (GitHub PRs, arxiv) alongside local records, combining both into a unified ranking.

**Not automated**: The retriever does not automatically suggest experiments. It provides information for the human to make decisions. This is a deliberate design choice — automated experiment suggestion tends to explore a narrow region of the search space (local optima), while human intuition is better at identifying orthogonal directions. The `agent.md` workflow (§7.6) does support fully autonomous iteration, but even then it pauses for human route selection at pivot points.

---

## 7. Experiment Templates

Pre-built YAML templates for common experiment types, reducing boilerplate.

### 7.1 Architecture Template

```yaml
# templates/architecture.yaml
category: architecture
priority: high
stages: [screen, gate]
reject_if_worse_by: 0.03

env_overrides:
  NUM_LAYERS: "{{ layers }}"
  MLP_RATIO: "{{ mlp_ratio }}"
  # Adjust XSA and VE layer indices for new depth
  XSA_LAYERS: "{{ xsa_layers }}"
  VE_LAYERS: "{{ ve_layers }}"
```

### 7.2 Hyperparameter Template

```yaml
# templates/hyperparameter.yaml
category: hyperparameter
priority: normal
stages: [screen, gate]
reject_if_worse_by: 0.05

env_overrides:
  "{{ param_name }}": "{{ param_value }}"
```

### 7.3 Batch Ablation Generation

```bash
# Generate a sweep from the command line:
python autoresearch.py sweep \
    --template hyperparameter \
    --param MUON_LR \
    --values 0.02,0.025,0.03,0.035,0.04 \
    --parent exp_000_baseline

# This creates 5 YAML configs in queue/ and registers them all
```

---

## 8. Failure Modes and Mitigations

| Failure Mode | Detection | Mitigation | Recovery |
|-------------|-----------|------------|----------|
| OOM during training | `RuntimeError: CUDA out of memory` in stderr | Retry with `DEVICE_BATCH_SIZE` halved | Automatic (1 retry) |
| NCCL timeout | `NCCL timeout` in stderr | Retry after 10s delay | Automatic (1 retry) |
| NaN loss | `train_loss` is NaN or Inf | Kill immediately | Mark REJECTED, no retry |
| GPTQ OOM | OOM during quantization | Reduce calibration set size | Automatic (1 retry) |
| Disk full | `No space left on device` | Alert in dashboard | Manual: clean old checkpoints |
| GPU hardware error | `ECC error` or `Xid` in dmesg | Alert + exclude GPU from pool | Manual: restart node |
| Process stall | No log output for 60s | Kill and retry | Automatic (1 retry) |
| SQLite corruption | Write error to registry.db | WAL mode + periodic backup | Restore from backup |
| Scheduler crash | Scheduler process dies | systemd auto-restart | Automatic |

### 8.1 Checkpoint Strategy

- **Screen stage**: Save checkpoint at end of training (single file, ~100–400 MB depending on model size). Used by gate stage.
- **Promote stage**: Save checkpoint every 1000 steps. On failure/restart, resume from latest checkpoint.
- **Retention policy**: Keep screen checkpoints for 24 hours after completion. Keep promote checkpoints indefinitely. Delete rejected experiment checkpoints after gate results are recorded.

---

## 9. Implementation Plan

### Phase 0: External Tool Integration (Day 0, pre-requisite)
Vendor the three tools from JianYan11/parameter-golf into the autoresearch_v2 tree. This is a copy-and-adapt step, not a from-scratch build.

**Deliverables**:
- Copy `scripts/h100_time_guess.py` → `scripts/h100_time_guess.py`. Refactor `check_log()` and `machine_total_tflops()` into importable functions (currently they call `sys.exit()` on error — change to raising exceptions). Add `estimate_h100_time(log_path, stage)` wrapper that returns seconds.
- Copy `scripts/generate_demo.py` → `scripts/generate_demo.py`. Add `--plain --prompt --max-new-tokens` batch mode that writes to a file instead of stdout. Verify it works with `screen_checkpoint.pt` format (not just `final_model.pt`).
- Adapt `agent.md` §7.5 step 1 (records-mining logic) into `retriever.py records` and `retriever.py propose` subcommands. The scanning/extraction logic (parse `submission.json`, grep READMEs for `val_bpb` tables) becomes Python code; the 3-route proposal format becomes structured JSON output.
- Import existing `autoresearch/results.tsv` (37 experiments) into SQLite with `import-tsv` command.

**Validation**: `python scripts/h100_time_guess.py check experiment_logs/some_run.log` produces a valid estimate. `python scripts/generate_demo.py --checkpoint <any_checkpoint> --plain --prompt "The" --max-new-tokens 32` produces text output. `python retriever.py records --bottleneck "architecture"` returns structured results from `records/`.

### Phase 1: Core Loop (Day 1)
Build the minimum viable system that runs experiments faster than the current approach.

**Deliverables**:
- `autoresearch.py` with scheduler, runner, GPU slot manager
- YAML config format + experiment registration
- SQLite registry (schema only, no dashboard yet)
- Screen → Gate pipeline with auto-reject
- Basic `metrics.jsonl` logging
- Compute pre-flight check in QUEUE using `h100_time_guess.estimate_h100_time()`
- Post-screen compute verification using `h100_time_guess.check_log()`
- Runner sets `EXPERIMENT_DESC` and `RESULTS_TSV_PATH` for TSV redundancy

**Validation**: Run the Track 4 Category A ablations (A1–A4) as a batch. Should complete in ~10 minutes (180s screen + 180s gate × 4 experiments, 2 at a time) vs the original ~45 minutes sequential. Verify that compute estimates appear in the registry for each experiment.

### Phase 2: Observability (Day 2)
Add the dashboard and monitoring.

**Deliverables**:
- `dashboard.html` with status board + live training curves
- HTTP API endpoint in scheduler for dashboard polling
- Watchdog thread with health checks
- GPU utilization monitoring
- "Inspect" button in dashboard that invokes `generate_demo.py` on GATE-passed checkpoints
- Compute estimate column in Comparison Table (from `h100_time_guess` data in registry)

**Validation**: Run a batch of 8 experiments and observe them live in the dashboard. Verify that NaN detection kills a deliberately-broken experiment within 10s. Click "Inspect" on a passed experiment and see token samples in the Experiment Detail view.

### Phase 3: Git Integration + Persistence (Day 3)
Add reproducibility infrastructure.

**Deliverables**:
- Git branch per experiment, commits on completion
- `sync_logs.sh` cron job
- Sweep generation CLI
- Experiment templates
- `agent-define` command (records-mining → 3 routes → YAML generation)

**Validation**: Run a 5-point LR sweep, verify each has its own git branch, verify results are in SQLite, verify `sync_logs.sh` produces a valid backup. Run `agent-define --parent exp_000_baseline` and verify it produces 3 proposed routes with cited records.

### Phase 4: Research Retriever + Agent Loop (Day 4, optional)
Add the information retrieval tool and autonomous iteration mode.

**Deliverables**:
- GitHub PR search
- Arxiv search
- Local records-mining (`retriever.py records` and `retriever.py propose`)
- Local caching
- `autoresearch.py agent-loop` mode: implements the `agent.md` §7.5 loop on top of the scheduler — iterates autonomously (define → queue → screen → gate → keep/revert) until manually stopped, pausing only for human route selection at pivot points

---

## 10. Operational Runbook

### 10.1 Starting a Session

Follow the full session setup ritual in §3.1 first (tag, upstream sync, branch, file read, data verify). Then:

```bash
# 1. Bootstrap from prior work (first session only)
cd /workspace/autoresearch_v2
python autoresearch.py import-tsv autoresearch/results.tsv --tag "autoresearch_v1"

# 2. Check hardware and get time equivalence
python scripts/h100_time_guess.py
# → "ballpark: 8× ~1979 TFLOPS for 10 min ≈ 56.5 min on this rack (210 TFLOPS peak total)"

# 3. Create session branch (per §3.1)
git fetch origin && git checkout main && git pull
git checkout -b research/apr3

# 4. Start the scheduler daemon
python autoresearch.py daemon &

# 5. Open dashboard
# (from local machine, SSH tunnel)
ssh -L 8080:localhost:8080 user@runpod-host
# Then open http://localhost:8080 in browser

# 6a. Queue experiments (manual sweep)
python autoresearch.py sweep \
    --template architecture \
    --param NUM_LAYERS \
    --values 6,7,8,9 \
    --parent exp_000_baseline

# 6b. Or use agent-assisted definition (records-mining → 3 routes → pick)
python autoresearch.py agent-define --parent exp_000_baseline
# → Proposes 3 mechanistically-diverse routes with cited records
# → Human picks A/B/C → YAML configs generated in queue/

# 7. Monitor in dashboard, review results, queue follow-ups
```

### 10.2 Inspecting a Checkpoint (Qualitative Check)

```bash
# Interactive: stream tokens from a screen checkpoint
python scripts/generate_demo.py \
    --checkpoint experiments/exp_042/screen_checkpoint.pt \
    --tokenizer ./data/tokenizers/fineweb_1024_bpe.model

# Or single-shot for quick check:
python scripts/generate_demo.py \
    --checkpoint experiments/exp_042/screen_checkpoint.pt \
    --prompt "The most important discovery in" \
    --max-new-tokens 128 --plain

# The dashboard also has an "Inspect" button for GATE-passed experiments
# that runs this automatically and displays samples inline.
```

### 10.3 Promoting an Experiment

```bash
# After reviewing screen + gate + inspect results in dashboard:
python autoresearch.py promote exp_042

# This drains screen slots, runs full 600s training on 8 GPU,
# then runs full eval pipeline (GPTQ + sliding window)

# Pre-promote: check compute estimate from screen run
python scripts/h100_time_guess.py check experiments/exp_042/train.log
# → "guess at 8×H100 train time: 485.2 s   official cap: 600 s   PASS"
```

### 10.4 Emergency: Killing Everything

```bash
# Kill all running experiments and the scheduler
python autoresearch.py kill-all

# Or just kill one experiment
python autoresearch.py kill exp_042
```

### 10.5 Reviewing Results

```bash
# CLI summary of all experiments
python autoresearch.py status

# Export results to TSV (for external analysis or compatibility with agent.md workflow)
python autoresearch.py export results.tsv

# Show experiment lineage tree
python autoresearch.py tree

# Check if a specific experiment would fit within official budget
python scripts/h100_time_guess.py check experiments/exp_042/train.log
```

### 10.6 Autonomous Agent Loop

For overnight or unattended sessions, the agent loop (from `agent.md` §7.6) runs on top of the scheduler. It iterates autonomously through define → queue → screen → gate → keep/revert, pausing only for human route selection at session start or when pivoting after a dead end.

```bash
# Start the scheduler daemon first
python autoresearch.py daemon &

# Then start the agent loop (runs until manually interrupted)
python autoresearch.py agent-loop \
    --parent exp_000_baseline \
    --branch research/apr3

# The agent loop:
# 1. Mines records/ for 3 mechanistically-diverse routes
# 2. Waits for human to pick A/B/C (one-time at session start)
# 3. Generates YAML configs → drops in queue/
# 4. Monitors results as they come back from scheduler
# 5. Keeps experiments that improve BPB, reverts others
# 6. When the chosen route is exhausted, proposes 3 new routes (human pick again)
# 7. Repeats until Ctrl+C

# Fully autonomous mode (no human route selection — picks best route automatically):
python autoresearch.py agent-loop \
    --parent exp_000_baseline \
    --branch research/apr3 \
    --autonomous
```

The agent loop does NOT replace the scheduler — it is a client that submits experiments to the scheduler's queue. Multiple agent loops can run concurrently (e.g., one exploring architecture changes, another exploring hyperparameters) as long as they target different parent experiments.

---

## 11. Key Design Decisions and Rationale

### 11.1 Why 2 GPUs per screen slot, not 1?

The SOTA script uses `grad_accum_steps = 8 // world_size`. On 1 GPU, that's 8× accumulation — which means 8× fewer gradient updates per wall-clock second compared to 8 GPU. The effective batch size is preserved, but you get 8× fewer steps in 180s. This makes 1-GPU screening unreliable for ranking because the model is so undertrained that noise dominates.

With 2 GPUs, accumulation is 4×. Step count in 180s is roughly half of the 8-GPU baseline (~500 steps), which is enough for ranking fidelity based on Track 2 data (564 steps produced usable rankings).

### 11.2 Why not Bayesian optimization?

Three reasons:
1. The search space is heterogeneous (architecture choices, continuous hyperparameters, boolean flags, evaluation-time techniques). BO works best on continuous spaces.
2. The objective has a critical non-smooth dependency (GPTQ quantization) that BO's surrogate model can't capture.
3. The best ideas in the competition (BigramHash, XSA, TTT) came from human insight, not hyperparameter sweeps. The system should amplify human intuition, not replace it.

### 11.3 Why SQLite, not Postgres/Redis?

Single-node system with one writer (the scheduler). SQLite with WAL mode handles this perfectly, requires zero infrastructure, and survives process restarts. The entire database is a single file that gets backed up with `rsync`.

### 11.4 Why a single HTML file for the dashboard?

Zero dependencies. No `npm install`, no build step, no framework updates. The dashboard needs to work on a fresh RunPod instance with nothing installed except Python and a browser. It reads JSONL files directly and polls a tiny HTTP endpoint. If the HTTP server is down, you can still open the HTML file directly and manually refresh.

### 11.5 Why not just use Weights & Biases?

Three reasons:
1. W&B requires network access during training, which may not be available (some RunPod configs are network-restricted during training).
2. The GPTQ gate is custom logic that doesn't map cleanly to W&B's experiment tracking model.
3. Latency: W&B dashboard updates are on the order of 30–60s. We want 10s for interactive monitoring.

That said, nothing prevents adding a W&B sync later as a non-critical logging backend.

### 11.6 Why integrate JianYan11/parameter-golf tools instead of building from scratch?

Three tools from [JianYan11/parameter-golf](https://github.com/JianYan11/parameter-golf) solve real problems we'd otherwise have to build:

1. **`h100_time_guess.py`** — Compute estimation is surprisingly tricky to get right (GPU TFLOPS tables, log parsing, TFLOPS-ratio math). This script already handles the common GPU SKUs, has a clean CLI, and supports both pre-flight estimation and post-run verification. The refactoring cost (making `check_log()` importable) is ~30 minutes vs ~2 hours to build from scratch.

2. **`generate_demo.py`** — Qualitative checkpoint inspection fills a gap the spec originally missed. BPB is necessary but not sufficient for model quality assessment, especially for architecture changes. This script already handles all checkpoint formats (`final_model.pt`, `.int8.ptz`, DDP prefixes, nested dicts), supports CPU inference (no GPU slot needed), and has a polished interactive experience. Building this from scratch would be ~1 day of work for diminishing returns.

3. **`agent.md` workflow** — The records-mining and 3-route proposal pattern is a battle-tested approach to experiment ideation (37 experiments run in the autoresearch_v1 campaign). Rather than designing a new ideation workflow, we formalize this existing pattern into the DEFINE stage and the retriever's `propose` subcommand.

The integration principle is: **vendor the tool, adapt the interface, don't fork the logic**. Each tool keeps its core algorithm intact; we add thin wrappers (importable functions, batch mode flags, JSON output) so the scheduler and dashboard can call them programmatically. If the upstream tools improve, we can re-vendor with minimal merge conflicts.

### 11.7 Why a seven-stage pipeline instead of six?

The original spec had six stages (DEFINE → QUEUE → SCREEN → GATE → PROMOTE → SUBMIT). The INSPECT stage was added between GATE and PROMOTE based on the `generate_demo.py` integration. The rationale:

- PROMOTE is expensive (600s on all 8 GPUs, blocks all other experiments). A 30-second qualitative check that catches degenerate models before PROMOTE saves ~10 minutes of wasted GPU time per false positive.
- INSPECT is optional and lightweight (CPU-only, no GPU slot). It adds zero overhead to the automated pipeline when skipped.
- For architecture experiments — the most likely to produce qualitatively broken models despite decent BPB — INSPECT provides signal that no quantitative metric captures.