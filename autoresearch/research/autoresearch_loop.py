"""Continuous autoresearch loop: assesses experiment results, proposes follow-ups.

This is the "brain" that runs alongside the scheduler. While the scheduler
executes jobs on GPU nodes, this loop:

1. Monitors completed experiments for interesting results
2. Evaluates whether ideas panned out or need pivoting
3. Generates follow-up experiments (sweeps, ablations, new directions)
4. Proposes new ideas from research agent findings
5. Auto-completes ideas whose experiments have converged

It runs on the Azure VM and never touches GPUs directly.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import AutoResearchConfig
from ..db.models import (
    IdeaStatus, IdeaSource, ExperimentStatus, ExperimentCategory,
)
from ..db.registry import Registry
from ..db.knowledge import KnowledgeBase
from ..db.recipes import RecipeStore
from ..ideas.tracker import IdeaTracker
from ..research.agent import ResearchAgent
from ..claude import ClaudeRunner, build_task

log = logging.getLogger(__name__)


class AutoResearchLoop:
    """Continuous loop that assesses results and proposes new experiments."""

    def __init__(self, config: AutoResearchConfig, registry: Registry,
                 ideas: IdeaTracker, research: ResearchAgent,
                 knowledge: KnowledgeBase,
                 claude: Optional[ClaudeRunner] = None):
        self.config = config
        self.registry = registry
        self.ideas = ideas
        self.research = research
        self.kb = knowledge
        self.recipes = RecipeStore(registry, config.abs_recipes_dir)
        self.claude = claude  # may be None if claude_enabled=false
        self._stop_event = threading.Event()
        self._assessed_experiments: set[str] = set()  # exp_ids we've already assessed
        self._reported_experiments: set[str] = set()  # exp_ids we've kicked off write_report for

    def run(self):
        """Main loop (blocking). Runs every tick alongside the scheduler."""
        log.info("AutoResearch loop starting")

        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as e:
                log.exception("AutoResearch loop error: %s", e)

            # Run assessments every 30s (less frequent than scheduler)
            self._stop_event.wait(30)

        log.info("AutoResearch loop stopped")

    def stop(self):
        self._stop_event.set()

    def _tick(self):
        """One assessment cycle."""
        # 1. Assess newly completed experiments
        self._assess_completed_experiments()

        # 2. Check if any active ideas should be completed or pivoted
        self._evaluate_active_ideas()

        # 3. Auto-queue experiments for approved ideas that have no experiments yet
        self._auto_queue_approved_ideas()

        # 4. Generate follow-up experiments from successful screens
        self._generate_followups()

    # ── Assess Completed Experiments ───────────────────────────────────

    def _assess_completed_experiments(self):
        """Look at experiments that just finished and draw conclusions."""
        for status in (ExperimentStatus.DONE, ExperimentStatus.REJECTED, ExperimentStatus.FAILED):
            experiments = self.registry.list_experiments(status=status)
            for exp in experiments:
                if exp.id in self._assessed_experiments:
                    continue
                self._assessed_experiments.add(exp.id)
                self._assess_experiment(exp)

    def _assess_experiment(self, exp):
        """Assess a single completed experiment and store in knowledge base."""
        idea = self.registry.get_idea(exp.idea_id)
        if not idea:
            return

        verdict = self._verdict(exp)
        results = {
            "screen_ema_bpb": exp.screen_ema_bpb,
            "screen_train_bpb": exp.screen_train_bpb,
            "gate_int6_bpb": exp.gate_int6_bpb,
            "gate_quant_gap": exp.gate_quant_gap,
            "gate_passed": exp.gate_passed,
            "promote_ema_bpb": exp.promote_ema_bpb,
        }

        # Update the recipe's best-observed metrics so the leaderboard and
        # current_best_baseline pointer both stay in sync with reality.
        recipe_id = getattr(exp, "recipe_id", None)
        if recipe_id and exp.status == ExperimentStatus.DONE:
            val = exp.promote_ema_bpb or exp.screen_ema_bpb or exp.promote_train_bpb
            int6 = exp.promote_int6_bpb or exp.gate_int6_bpb
            art = exp.promote_artifact_mb or exp.gate_artifact_mb
            if val is not None or int6 is not None:
                try:
                    self.recipes.update_best_metrics(
                        recipe_id, exp.id,
                        val_bpb=val, int6_bpb=int6, artifact_mb=art,
                    )
                except Exception as e:
                    log.warning("recipe metric update failed for %s: %s",
                                exp.id, e)

        # Store in knowledge base for future queries
        self.kb.store_experiment_result(
            experiment_id=exp.id,
            name=exp.name,
            hypothesis=exp.hypothesis,
            env_overrides=exp.env_overrides or {},
            results={k: v for k, v in results.items() if v is not None},
            verdict=verdict,
            tags=idea.tags if idea.tags else [],
        )

        # Kick off a Claude `write_report` task on DONE experiments so the
        # experiment_logs/claude_reports/ markdown + verdict populate the
        # leaderboard without blocking the assessment loop.
        if (exp.status == ExperimentStatus.DONE
                and self.claude is not None
                and self.config.claude_auto_report
                and exp.id not in self._reported_experiments):
            self._reported_experiments.add(exp.id)
            try:
                baseline = None
                current_best = self.recipes.current_best()
                if current_best and current_best.best_val_bpb:
                    baseline = current_best.best_val_bpb
                log_path = ""
                # The scheduler syncs training logs to experiment_logs/<id>.log
                candidate = Path(self.config.workspace_dir).parent / "experiment_logs" / f"{exp.id}.log"
                if candidate.exists():
                    log_path = str(candidate)
                spec = build_task(
                    "write_report",
                    config=self.config, registry=self.registry,
                    experiment=exp, log_path=log_path,
                    recipe_id=getattr(exp, "recipe_id", "") or "",
                    baseline_bpb=baseline,
                )
                self.claude.run_async(spec)
            except Exception as e:
                log.warning("write_report spawn failed for %s: %s", exp.id, e)

        if exp.status == ExperimentStatus.DONE:
            metrics = []
            if exp.screen_ema_bpb:
                metrics.append(f"screen_bpb={exp.screen_ema_bpb:.4f}")
            if exp.gate_int6_bpb:
                metrics.append(f"gate_bpb={exp.gate_int6_bpb:.4f}")
            if exp.gate_quant_gap is not None:
                metrics.append(f"quant_gap={exp.gate_quant_gap:.4f}")
            if exp.promote_ema_bpb:
                metrics.append(f"promote_bpb={exp.promote_ema_bpb:.4f}")

            self.ideas.log_experiment_event(
                exp.idea_id, exp.id, "assessment",
                {"status": "done", "metrics": ", ".join(metrics),
                 "verdict": verdict},
            )
            log.info("Assessed %s: %s (%s)", exp.id, verdict,
                     ", ".join(metrics))

        elif exp.status == ExperimentStatus.REJECTED:
            self.ideas.log_experiment_event(
                exp.idea_id, exp.id, "assessment",
                {"status": "rejected", "reason": exp.rejection_reason},
            )
            log.info("Assessed %s: REJECTED (%s)", exp.id, exp.rejection_reason)

        elif exp.status == ExperimentStatus.FAILED:
            self.ideas.log_experiment_event(
                exp.idea_id, exp.id, "assessment",
                {"status": "failed", "reason": exp.rejection_reason},
            )

    def _verdict(self, exp) -> str:
        """Quick verdict on an experiment's results."""
        if exp.gate_passed is False:
            return "gate_failed"
        if exp.screen_ema_bpb and exp.screen_ema_bpb < 1.30:
            return "promising"
        if exp.screen_ema_bpb and exp.screen_ema_bpb < 1.35:
            return "competitive"
        return "marginal"

    # ── Evaluate Active Ideas ──────────────────────────────────────────

    def _evaluate_active_ideas(self):
        """Check if active ideas should be completed, parked, or extended."""
        active_ideas = self.registry.list_ideas(IdeaStatus.ACTIVE)

        for idea in active_ideas:
            experiments = self.registry.list_experiments(idea_id=idea.id)
            if not experiments:
                continue

            # Count states
            total = len(experiments)
            done = sum(1 for e in experiments if e.status == ExperimentStatus.DONE)
            rejected = sum(1 for e in experiments if e.status == ExperimentStatus.REJECTED)
            failed = sum(1 for e in experiments if e.status == ExperimentStatus.FAILED)
            running = sum(1 for e in experiments if e.status in (
                ExperimentStatus.SCREENING, ExperimentStatus.GATING,
                ExperimentStatus.PROMOTING, ExperimentStatus.DEPLOYING))
            queued = sum(1 for e in experiments if e.status == ExperimentStatus.QUEUED)

            finished = done + rejected + failed

            # All experiments done → evaluate idea
            if finished == total and running == 0 and queued == 0:
                self._conclude_idea(idea, experiments)

            # All rejected/failed → idea is a dead end
            elif rejected + failed == total:
                self.ideas.reject_idea(
                    idea.id,
                    f"All {total} experiments rejected/failed. Dead end.",
                )
                log.info("Idea %s: all experiments failed, rejecting", idea.id)

    def _conclude_idea(self, idea, experiments):
        """Draw a conclusion for a completed idea."""
        done_exps = [e for e in experiments if e.status == ExperimentStatus.DONE]
        if not done_exps:
            self.ideas.reject_idea(idea.id, "No successful experiments")
            return

        # Find best result
        best = min(done_exps,
                   key=lambda e: e.screen_ema_bpb if e.screen_ema_bpb else float('inf'))

        conclusion = (
            f"Completed {len(experiments)} experiments. "
            f"Best: {best.name} with screen_bpb={best.screen_ema_bpb:.4f}"
        )
        if best.gate_int6_bpb:
            conclusion += f", gate_bpb={best.gate_int6_bpb:.4f}"
        if best.gate_passed:
            conclusion += " (gate PASSED → candidate for promotion)"
        else:
            conclusion += " (gate failed or not run)"

        self.ideas.complete_idea(idea.id, conclusion)
        log.info("Idea %s completed: %s", idea.id, conclusion)

    # ── Auto-Queue Approved Ideas ──────────────────────────────────────

    def _auto_queue_approved_ideas(self):
        """For approved ideas with no experiments, create initial experiments."""
        approved = self.registry.list_ideas(IdeaStatus.APPROVED)
        for idea in approved:
            if idea.experiment_ids:
                continue  # Already has experiments

            # Check if idea has env_overrides hints in notes (JSON block)
            env = self._extract_env_from_notes(idea.notes)
            if env:
                exp = self.ideas.create_experiment(
                    idea_id=idea.id,
                    name=f"{idea.title[:30]} - initial",
                    env_overrides=env,
                    category=self._guess_category(idea.tags),
                    hypothesis=idea.hypothesis,
                    stages=["screen", "gate"],
                )
                self.registry.update_experiment_status(exp.id, ExperimentStatus.QUEUED)
                log.info("Auto-queued initial experiment for idea %s: %s",
                         idea.id, exp.id)

    # ── Follow-up Generation ───────────────────────────────────────────

    def _generate_followups(self):
        """Generate follow-up experiments from promising screen results."""
        done_exps = self.registry.list_experiments(status=ExperimentStatus.DONE)

        for exp in done_exps:
            if exp.id in self._assessed_experiments:
                continue
            if not exp.gate_passed:
                continue
            if not exp.screen_ema_bpb or exp.screen_ema_bpb > 1.35:
                continue

            # This experiment passed the gate and has good BPB → consider promotion
            idea = self.registry.get_idea(exp.idea_id)
            if not idea or idea.status in (IdeaStatus.COMPLETED, IdeaStatus.REJECTED):
                continue

            # Check if there's already a promote-stage experiment
            all_exps = self.registry.list_experiments(idea_id=idea.id)
            has_promote = any(
                "promote" in (e.stages or [])
                for e in all_exps
                if e.status in (ExperimentStatus.QUEUED, ExperimentStatus.PROMOTING,
                                ExperimentStatus.DONE)
            )
            if has_promote:
                continue

            # Log recommendation (don't auto-promote — that's a human decision)
            self.ideas.log_experiment_event(
                exp.idea_id, exp.id, "recommendation",
                {"action": "consider_promotion",
                 "screen_bpb": exp.screen_ema_bpb,
                 "gate_bpb": exp.gate_int6_bpb,
                 "message": f"{exp.name} passed gate with bpb={exp.gate_int6_bpb:.4f}. Consider promoting."},
            )

    # ── Helpers ────────────────────────────────────────────────────────

    def _extract_env_from_notes(self, notes: str) -> dict:
        """Try to extract env_overrides from idea notes (looks for JSON block)."""
        import re
        m = re.search(r'\{[^}]*"[A-Z_]+"[^}]*\}', notes)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {}

    def _guess_category(self, tags: list) -> ExperimentCategory:
        tag_str = " ".join(tags).lower()
        if "architecture" in tag_str or "mlp" in tag_str or "activation" in tag_str:
            return ExperimentCategory.ARCHITECTURE
        if "hyper" in tag_str or "lr" in tag_str or "optimizer" in tag_str:
            return ExperimentCategory.HYPERPARAMETER
        if "quant" in tag_str:
            return ExperimentCategory.QUANTIZATION
        if "ttt" in tag_str:
            return ExperimentCategory.TTT
        return ExperimentCategory.OTHER

    # ── Status ─────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        return {
            "assessed_experiments": len(self._assessed_experiments),
            "running": not self._stop_event.is_set(),
        }
