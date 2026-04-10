"""Idea tracker: manages research ideas and their experiment logs.

Ideas are first-class entities. Each idea has:
- A hypothesis, source, and critical evaluation
- One or more experiments testing variants of the idea
- A dedicated log directory: logs/ideas/<idea_id>/
- A conclusion once all experiments complete
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import AutoResearchConfig
from ..db.models import (
    Idea, IdeaStatus, IdeaSource,
    Experiment, ExperimentStatus, ExperimentCategory,
)
from ..db.registry import Registry

log = logging.getLogger(__name__)


class IdeaTracker:
    """Manages the lifecycle of research ideas."""

    def __init__(self, config: AutoResearchConfig, registry: Registry):
        self.config = config
        self.registry = registry
        self.ideas_dir = config.abs_ideas_dir
        self.ideas_dir.mkdir(parents=True, exist_ok=True)

    # ── Idea CRUD ──────────────────────────────────────────────────────

    def create_idea(self, title: str, hypothesis: str,
                     source: IdeaSource = IdeaSource.HUMAN,
                     source_ref: str = "",
                     priority: int = 2,
                     tags: list[str] | None = None,
                     notes: str = "",
                     parent_idea: str | None = None) -> Idea:
        """Create a new research idea.

        Ideas are auto-approved on creation so the scheduler picks them up
        and queues experiments without a human-in-the-loop approval gate.
        Rejection is still manual — the user can reject an idea at any
        time from the dashboard, but the default is "run it".
        """
        idea_id = self._generate_id(title)
        idea = Idea(
            id=idea_id, title=title, hypothesis=hypothesis,
            source=source, source_ref=source_ref,
            status=IdeaStatus.APPROVED, priority=priority,
            parent_idea=parent_idea,
            tags=tags or [], notes=notes,
        )
        self.registry.insert_idea(idea)

        # Create idea log directory
        idea_dir = self.ideas_dir / idea_id
        idea_dir.mkdir(parents=True, exist_ok=True)
        self._write_idea_readme(idea)

        log.info("Created idea: %s - %s", idea_id, title)
        return idea

    def evaluate_idea(self, idea_id: str, evaluation: str):
        """Add critical evaluation to an idea (rules, scalability, novelty)."""
        self.registry.update_idea_evaluation(idea_id, evaluation)
        # Append to idea log
        self._append_log(idea_id, "evaluation", evaluation)

    def approve_idea(self, idea_id: str, notes: str = ""):
        """Approve an idea for experimentation."""
        self.registry.update_idea_status(idea_id, IdeaStatus.APPROVED, notes)

    def activate_idea(self, idea_id: str):
        """Mark an idea as having active experiments."""
        self.registry.update_idea_status(idea_id, IdeaStatus.ACTIVE)

    def complete_idea(self, idea_id: str, conclusion: str):
        """Mark an idea as completed with a conclusion."""
        self.registry.update_idea_status(idea_id, IdeaStatus.COMPLETED, conclusion)
        self._append_log(idea_id, "conclusion", conclusion)

    def park_idea(self, idea_id: str, reason: str = ""):
        """Park an idea for later."""
        self.registry.update_idea_status(idea_id, IdeaStatus.PARKED, reason)

    def reject_idea(self, idea_id: str, reason: str):
        """Reject an idea."""
        self.registry.update_idea_status(idea_id, IdeaStatus.REJECTED, reason)
        self._append_log(idea_id, "rejected", reason)

    def set_priority(self, idea_id: str, priority: int):
        """Update idea priority (1=low, 4=critical)."""
        from ..db.models import EventType
        with self.registry._conn() as conn:
            conn.execute(
                "UPDATE ideas SET priority=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (priority, idea_id),
            )
        self.registry.emit_event(
            EventType.USER_PRIORITIZE, "idea", idea_id,
            {"priority": priority},
        )

    # ── Experiment creation under ideas ────────────────────────────────

    def create_experiment(self, idea_id: str, name: str,
                           env_overrides: dict,
                           category: ExperimentCategory = ExperimentCategory.OTHER,
                           hypothesis: str = "",
                           parent_exp: str | None = None,
                           stages: list[str] | None = None,
                           priority: int | None = None,
                           notes: str = "") -> Experiment:
        """Create an experiment under an idea."""
        idea = self.registry.get_idea(idea_id)
        if not idea:
            raise ValueError(f"Idea not found: {idea_id}")

        exp_count = len(idea.experiment_ids)
        exp_id = f"{idea_id}_exp{exp_count + 1:03d}"

        exp = Experiment(
            id=exp_id, name=name, idea_id=idea_id,
            hypothesis=hypothesis or idea.hypothesis,
            category=category,
            priority=priority if priority is not None else idea.priority,
            env_overrides=env_overrides,
            parent_id=parent_exp,
            stages=stages or ["screen", "gate"],
            notes=notes,
        )
        self.registry.insert_experiment(exp)

        # Create experiment log dir under idea
        exp_dir = self.ideas_dir / idea_id / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Activate idea if it was just approved
        if idea.status == IdeaStatus.APPROVED:
            self.activate_idea(idea_id)

        self._append_log(idea_id, "experiment_created",
                          f"{exp_id}: {name} | env={json.dumps(env_overrides)}")
        return exp

    # ── Queries ────────────────────────────────────────────────────────

    def get_idea_with_experiments(self, idea_id: str) -> Optional[dict]:
        """Get full idea context with all experiments."""
        idea = self.registry.get_idea(idea_id)
        if not idea:
            return None
        experiments = self.registry.list_experiments(idea_id=idea_id)
        log_path = self.ideas_dir / idea_id / "log.jsonl"
        log_entries = []
        if log_path.exists():
            for line in log_path.read_text().strip().split("\n"):
                if line.strip():
                    try:
                        log_entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return {
            "idea": idea,
            "experiments": experiments,
            "log": log_entries,
        }

    def get_idea_summary_table(self) -> list[dict]:
        """Get a summary of all ideas with their best experiment results."""
        ideas = self.registry.list_ideas()
        summaries = []
        for idea in ideas:
            experiments = self.registry.list_experiments(idea_id=idea.id)
            best_screen_bpb = None
            best_gate_bpb = None
            best_promote_bpb = None
            last_finished = None
            total_exps = len(experiments)
            running = sum(1 for e in experiments if e.status in
                          (ExperimentStatus.SCREENING, ExperimentStatus.GATING,
                           ExperimentStatus.PROMOTING, ExperimentStatus.DEPLOYING))
            done = sum(1 for e in experiments if e.status == ExperimentStatus.DONE)
            for e in experiments:
                if e.screen_ema_bpb and (best_screen_bpb is None or e.screen_ema_bpb < best_screen_bpb):
                    best_screen_bpb = e.screen_ema_bpb
                if e.gate_int6_bpb and (best_gate_bpb is None or e.gate_int6_bpb < best_gate_bpb):
                    best_gate_bpb = e.gate_int6_bpb
                if e.promote_ema_bpb and (best_promote_bpb is None or e.promote_ema_bpb < best_promote_bpb):
                    best_promote_bpb = e.promote_ema_bpb
                if e.completed_at and (last_finished is None or str(e.completed_at) > last_finished):
                    last_finished = str(e.completed_at)

            summaries.append({
                "id": idea.id,
                "title": idea.title,
                "status": idea.status.value,
                "priority": idea.priority,
                "source": idea.source.value,
                "tags": idea.tags,
                "total_experiments": total_exps,
                "running_experiments": running,
                "completed_experiments": done,
                "best_screen_bpb": best_screen_bpb,
                "best_gate_bpb": best_gate_bpb,
                "best_promote_bpb": best_promote_bpb,
                "created_at": str(idea.created_at) if idea.created_at else None,
                "updated_at": str(idea.updated_at) if idea.updated_at else None,
                "last_experiment_finished_at": last_finished,
            })
        return summaries

    # ── Logging ────────────────────────────────────────────────────────

    def _append_log(self, idea_id: str, event: str, message: str):
        """Append to the idea's structured log."""
        log_path = self.ideas_dir / idea_id / "log.jsonl"
        entry = {
            "t": time.time(),
            "ts": datetime.utcnow().isoformat(),
            "event": event,
            "message": message,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_experiment_event(self, idea_id: str, experiment_id: str,
                              event: str, data: dict):
        """Log an experiment event under its idea."""
        exp_log = self.ideas_dir / idea_id / experiment_id / "events.jsonl"
        exp_log.parent.mkdir(parents=True, exist_ok=True)
        entry = {"t": time.time(), "event": event, **data}
        with open(exp_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _write_idea_readme(self, idea: Idea):
        """Write a human-readable README for the idea directory."""
        readme = self.ideas_dir / idea.id / "README.md"
        readme.write_text(
            f"# {idea.title}\n\n"
            f"**ID**: {idea.id}\n"
            f"**Hypothesis**: {idea.hypothesis}\n"
            f"**Source**: {idea.source.value}"
            f"{' (' + idea.source_ref + ')' if idea.source_ref else ''}\n"
            f"**Priority**: {idea.priority}\n"
            f"**Tags**: {', '.join(idea.tags) if idea.tags else 'none'}\n\n"
            f"## Notes\n{idea.notes}\n"
        )

    def _generate_id(self, title: str) -> str:
        """Generate a unique idea ID from title."""
        import re
        slug = re.sub(r'[^a-z0-9]+', '_', title.lower().strip())[:30].strip('_')
        # Check for collisions
        existing = {i.id for i in self.registry.list_ideas()}
        candidate = f"idea_{slug}"
        if candidate not in existing:
            return candidate
        for i in range(2, 100):
            c = f"idea_{slug}_{i}"
            if c not in existing:
                return c
        return f"idea_{slug}_{int(time.time())}"
