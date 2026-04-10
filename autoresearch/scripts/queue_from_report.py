#!/usr/bin/env python3
"""Queue Bucket-1 experiments derived from the Adaptive-Recurrence+Ternary-QAT report.

Every hypothesis in here maps to env vars that actually exist in
train_gpt.py (verified against os.environ.get() calls at lines 41-87).
Anything from the report that requires new code is NOT in this file —
those belong in a separate design doc, not the experiment queue.

Usage:
    python -m autoresearch.scripts.queue_from_report

It will:
  1. Create a single parent idea holding the full report as notes
     (status=parked, so the scheduler ignores it).
  2. Create 7 child ideas, one per hypothesis cluster, auto-approved.
  3. Create ~15 experiments across those ideas with env_overrides and a
     structured notes template containing success + kill criteria.

Run this from the repo root. It talks directly to the same SQLite
registry the worker is writing to (SQLite WAL supports concurrent
writers), so you don't need to stop the worker.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from autoresearch.config import AutoResearchConfig
from autoresearch.db.registry import Registry
from autoresearch.db.models import IdeaSource, ExperimentCategory, ExperimentStatus
from autoresearch.ideas.tracker import IdeaTracker


# ── Note template ────────────────────────────────────────────────────

NOTE_TEMPLATE = """## Hypothesis
{hypothesis}

## Mechanism
{mechanism}

## Success criterion
{success}

## Kill criterion
{kill}

## Confounds
{confounds}

## Rollback
{rollback}
"""

DEFAULT_SUCCESS = (
    "Screen EMA BPB improves vs current-best baseline by >= 0.003 "
    "AND wallclock/step does not regress > 5%."
)
DEFAULT_KILL = (
    "EMA BPB > baseline + 0.01 after 120s of wallclock, "
    "OR NaN/inf loss, "
    "OR tokens/sec drops > 15% without >= 3% BPB gain."
)


# ── Experiment spec ──────────────────────────────────────────────────

@dataclass
class ExpSpec:
    name: str
    env: dict                 # env_overrides
    hypothesis: str           # one-liner for the experiment row
    mechanism: str            # why we think it works
    confounds: str            # what to rule out
    rollback: dict            # env_overrides that revert to baseline
    category: ExperimentCategory = ExperimentCategory.HYPERPARAMETER
    priority: int = 2
    success: str = DEFAULT_SUCCESS
    kill: str = DEFAULT_KILL


@dataclass
class IdeaSpec:
    title: str
    hypothesis: str           # idea-level one-liner
    tags: list
    priority: int
    experiments: list         # list[ExpSpec]
    category: ExperimentCategory = ExperimentCategory.HYPERPARAMETER


# ── The Bucket-1 plan ────────────────────────────────────────────────

IDEAS: list[IdeaSpec] = [

    # ── (1) Shape: smaller-deeper ─────────────────────────────────────
    IdeaSpec(
        title="Shape sweep: smaller-deeper vs wider-shallower",
        hypothesis=(
            "At 10-min budget the model is undertrained. Reallocating "
            "params toward more layers (depth) with slightly smaller "
            "width may converge faster in wallclock."
        ),
        tags=["architecture", "shape", "undertraining"],
        priority=3,
        category=ExperimentCategory.ARCHITECTURE,
        experiments=[
            ExpSpec(
                name="deeper-narrower 11L x 384d",
                env={"NUM_LAYERS": "11", "MODEL_DIM": "384"},
                hypothesis="More depth at lower width = more effective compute per token.",
                mechanism="Report cites Olmo Hybrid finding 3:1 GDN/attn wins at small scale; closest env-only approximation is increasing depth.",
                confounds="Total params change — must compare artifact MB too.",
                rollback={"NUM_LAYERS": "9", "MODEL_DIM": "512"},
                category=ExperimentCategory.ARCHITECTURE,
                priority=3,
            ),
            ExpSpec(
                name="shallower-wider 7L x 640d",
                env={"NUM_LAYERS": "7", "MODEL_DIM": "640", "NUM_HEADS": "10"},
                hypothesis="Fewer layers = fewer sync points, more tokens/sec, may offset capacity loss.",
                mechanism="Bounded wallclock favors throughput; report mentions minimizing per-loop sync.",
                confounds="NUM_HEADS must divide MODEL_DIM; head_dim changes.",
                rollback={"NUM_LAYERS": "9", "MODEL_DIM": "512", "NUM_HEADS": "8"},
                category=ExperimentCategory.ARCHITECTURE,
                priority=3,
            ),
        ],
    ),

    # ── (2) MLP expansion sweep ───────────────────────────────────────
    IdeaSpec(
        title="MLP expansion sweep (2x/3x/4x)",
        hypothesis=(
            "Report argues FC2 is the QAT bottleneck and that MLP "
            "capacity matters disproportionately at small scale. "
            "Sweeping MLP_MULT at fixed shape isolates the tradeoff."
        ),
        tags=["architecture", "mlp"],
        priority=2,
        category=ExperimentCategory.ARCHITECTURE,
        experiments=[
            ExpSpec(
                name=f"MLP_MULT={m}",
                env={"MLP_MULT": m},
                hypothesis=f"MLP expansion {m}x trades artifact size for convergence speed.",
                mechanism="Wider MLP = more parameters per layer; report says undertrained LLMs tolerate more params.",
                confounds="Artifact MB scales roughly linearly with MLP_MULT; must check int6_bpb and artifact_mb jointly.",
                rollback={"MLP_MULT": "2"},
                category=ExperimentCategory.ARCHITECTURE,
                priority=2,
            )
            for m in ("3", "4")
        ],
    ),

    # ── (3) Sequence length / batch tokens ────────────────────────────
    IdeaSpec(
        title="Shorter sequences, more updates per wallclock",
        hypothesis=(
            "Report: 'Short sequences; micro-batch with grad accumulation' "
            "for throughput. Halving seq_len + halving batch_tokens keeps "
            "tokens-seen constant but doubles update count."
        ),
        tags=["throughput", "schedule"],
        priority=3,
        experiments=[
            ExpSpec(
                name="seq512 half-batch",
                env={"TRAIN_SEQ_LEN": "512", "TRAIN_BATCH_TOKENS": "262144"},
                hypothesis="2x more optimizer steps in the same wallclock window.",
                mechanism="Attention is O(L^2); shorter L = faster step; more steps = better convergence under 10-min cap.",
                confounds="Loss may be higher with shorter context; compare at matched total tokens seen.",
                rollback={"TRAIN_SEQ_LEN": "1024", "TRAIN_BATCH_TOKENS": "524288"},
                priority=3,
            ),
            ExpSpec(
                name="seq2048 double-batch",
                env={"TRAIN_SEQ_LEN": "2048", "TRAIN_BATCH_TOKENS": "1048576"},
                hypothesis="Longer context may give better BPB if per-step cost is offset by fewer steps needed.",
                mechanism="Tests the opposite direction to confirm the tradeoff is monotonic.",
                confounds="May OOM or crush tokens/sec.",
                rollback={"TRAIN_SEQ_LEN": "1024", "TRAIN_BATCH_TOKENS": "524288"},
                priority=2,
            ),
        ],
    ),

    # ── (4) Schedule tightening for 10-min budget ─────────────────────
    IdeaSpec(
        title="Warmup/warmdown compression for 10-min budget",
        hypothesis=(
            "Default warmup=20, warmdown=1200 was tuned for longer runs. "
            "At 10-min wallclock budget, proportionally shorter schedules "
            "should leave more steps at peak LR."
        ),
        tags=["schedule", "lr"],
        priority=2,
        experiments=[
            ExpSpec(
                name="tight-schedule",
                env={"WARMUP_STEPS": "10", "WARMDOWN_ITERS": "600"},
                hypothesis="Spend more of the budget at peak LR → faster descent.",
                mechanism="Report: 'freeze router after a brief warmup' suggests compressing warmup is safe.",
                confounds="Shorter warmup can destabilize Muon; watch for loss spikes in first 100 steps.",
                rollback={"WARMUP_STEPS": "20", "WARMDOWN_ITERS": "1200"},
                priority=2,
            ),
        ],
    ),

    # ── (5) Muon momentum sweep ───────────────────────────────────────
    IdeaSpec(
        title="Muon momentum fine sweep",
        hypothesis=(
            "PR#1521 claims 0.97 is better than default 0.95. Do a proper "
            "fine sweep to confirm and find the optimum."
        ),
        tags=["optimizer", "muon"],
        priority=2,
        experiments=[
            ExpSpec(
                name=f"muon_mom={m}",
                env={"MUON_MOMENTUM": m},
                hypothesis=f"Momentum {m} may be optimal at this wallclock.",
                mechanism="Higher momentum = more smoothing, lower = more reactivity. Short runs often prefer lower.",
                confounds="Interacts with MUON_MOMENTUM_WARMUP_STEPS; keep that fixed for this sweep.",
                rollback={"MUON_MOMENTUM": "0.95"},
                priority=2,
            )
            for m in ("0.93", "0.96", "0.97", "0.98")
        ],
    ),

    # ── (6) KV-head sweep (proxy for report's KV sharing) ─────────────
    IdeaSpec(
        title="KV-head count sweep",
        hypothesis=(
            "Report advocates KV-sharing inside MoR. The closest env-only "
            "analogue is reducing NUM_KV_HEADS (GQA ratio). Tighter GQA "
            "saves memory at some BPB cost — find the knee."
        ),
        tags=["architecture", "attention", "gqa"],
        priority=2,
        category=ExperimentCategory.ARCHITECTURE,
        experiments=[
            ExpSpec(
                name=f"kv_heads={k}",
                env={"NUM_KV_HEADS": k},
                hypothesis=f"{k} KV heads balances memory vs BPB.",
                mechanism="Fewer KV heads = smaller KV cache = more room for batch/seq; report ties this to throughput.",
                confounds="NUM_KV_HEADS must divide NUM_HEADS (8 by default).",
                rollback={"NUM_KV_HEADS": "4"},
                category=ExperimentCategory.ARCHITECTURE,
                priority=2,
            )
            for k in ("2", "8")
            # 4 is default; 1 wouldn't divide; 8 = full MHA
        ],
    ),

    # ── (7) LR scale sweep ────────────────────────────────────────────
    IdeaSpec(
        title="Matrix LR sweep at tight budget",
        hypothesis=(
            "With shorter effective training, LR may need to scale. "
            "Sweep MATRIX_LR around default 0.04."
        ),
        tags=["optimizer", "lr"],
        priority=2,
        experiments=[
            ExpSpec(
                name=f"matrix_lr={lr}",
                env={"MATRIX_LR": lr},
                hypothesis=f"Matrix LR {lr} may better fit compressed budget.",
                mechanism="At 10-min budget, slightly higher LR can reach lower loss faster — at cost of stability.",
                confounds="Interacts with schedule; pair each with fixed warmup.",
                rollback={"MATRIX_LR": "0.04"},
                priority=2,
            )
            for lr in ("0.03", "0.05", "0.06")
        ],
    ),
]


# ── The parent "roadmap" idea ────────────────────────────────────────

ROADMAP_REPORT = """# 16MB in 10 Minutes: Adaptive Recurrence + Ternary QAT (Full Report)

This idea is PARKED. It holds the long-form research report from the
Parallel Web Systems deep research run. The autoresearch system cannot
run any experiment from this report as-is because every high-impact
technique (MoR, HESTIA, BLT, GDN/Attention hybrid, SEPARATE) requires
new code, not an env-var sweep.

The Bucket-1 subset that IS env-sweepable has been spun out into its
own child ideas (see `parent_idea` pointer in the registry).

## To convert any section of this report into a real experiment:
1. Add a concrete env var + code path in train_gpt.py
2. Design a single screen-stage measurement with a kill rule
3. Create a new child idea under this one

## High-level recommendations (for manual planning only, NOT the queue)
- MoR routing over a shared 4-layer block (3 GDN + 1 attention)
- HESTIA ternary QAT for the 16MB artifact constraint
- BLT byte patching to compress effective sequence length
- In-Place TTT on MLP final projections
- Selective LM (Rho-1 style) with n-gram scorer

These are multi-week research projects, not autoresearch tasks.
"""


# ── Driver ───────────────────────────────────────────────────────────

def main():
    cfg_path = Path("autoresearch/config.yaml")
    config = AutoResearchConfig.from_yaml(cfg_path)
    registry = Registry(config.abs_db_path)
    tracker = IdeaTracker(config, registry)

    # Create roadmap parent idea and park it
    roadmap = tracker.create_idea(
        title="Adaptive Recurrence + Ternary QAT (roadmap)",
        hypothesis=(
            "Multi-week research plan derived from Parallel Web Systems "
            "deep research. Parked — env-sweepable pieces are spun out "
            "as child ideas; new-code pieces need manual scoping."
        ),
        source=IdeaSource.WEB_RESEARCH,
        source_ref="parallel_web_systems/adaptive_recurrence_ternary_qat",
        priority=1,
        tags=["roadmap", "multi-week", "mor", "hestia", "blt", "ttt"],
        notes=ROADMAP_REPORT,
    )
    tracker.park_idea(roadmap.id, "Roadmap only — see child ideas for runnable experiments")
    print(f"[parent] {roadmap.id}  status=parked")

    total_exps = 0
    for spec in IDEAS:
        idea = tracker.create_idea(
            title=spec.title,
            hypothesis=spec.hypothesis,
            source=IdeaSource.WEB_RESEARCH,
            source_ref="parallel_web_systems/adaptive_recurrence_ternary_qat",
            priority=spec.priority,
            tags=spec.tags,
            notes=f"Derived from Bucket-1 extraction of the roadmap report ({roadmap.id}).",
            parent_idea=roadmap.id,
        )
        tracker.approve_idea(idea.id, "Auto-approved by queue_from_report.py")
        print(f"[idea] {idea.id}  p={spec.priority}  '{spec.title}'")

        for exp_spec in spec.experiments:
            notes = NOTE_TEMPLATE.format(
                hypothesis=exp_spec.hypothesis,
                mechanism=exp_spec.mechanism,
                success=exp_spec.success,
                kill=exp_spec.kill,
                confounds=exp_spec.confounds,
                rollback=f"env_overrides = {exp_spec.rollback}",
            )
            exp = tracker.create_experiment(
                idea_id=idea.id,
                name=exp_spec.name,
                env_overrides=exp_spec.env,
                category=exp_spec.category,
                hypothesis=exp_spec.hypothesis,
                stages=["screen", "gate"],
                priority=exp_spec.priority,
                notes=notes,
            )
            # Promote DEFINED → QUEUED so the scheduler actually picks it up.
            registry.update_experiment_status(exp.id, ExperimentStatus.QUEUED)
            total_exps += 1
            print(f"    [exp] {exp.id}  env={exp_spec.env}")

    print()
    print(f"Done. Created {len(IDEAS)} ideas + 1 roadmap parent, {total_exps} experiments.")
    print("The scheduler will pick them up on its next tick (~10s).")


if __name__ == "__main__":
    sys.exit(main() or 0)
