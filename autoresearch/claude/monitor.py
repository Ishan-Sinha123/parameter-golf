"""Claude-based anomaly detector for running experiments.

Complements the rule-based scheduler.monitor (NaN/OOM/stall detectors).
This one runs Claude Code with the `monitor_run` task type against each
active experiment on a slower cadence, reading the tail of the training
log and giving a structured verdict: healthy / suspicious / kill / unknown.

Repeated "kill" verdicts on the same experiment are escalated to a real
kill via the command queue. Single "suspicious" verdicts just get logged
as events so a human can eyeball them in the dashboard.

Cadence, lookback window, and escalation thresholds are all config-driven
so the user can tune how aggressive this is.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from ..config import AutoResearchConfig
from ..db.models import ExperimentStatus
from ..db.registry import Registry
from .runner import ClaudeRunner, ClaudeTaskResult
from .tasks import build_task

log = logging.getLogger(__name__)


# Experiment statuses that are "live" and worth monitoring
_ACTIVE_STATUSES = (
    ExperimentStatus.SCREENING,
    ExperimentStatus.GATING,
    ExperimentStatus.PROMOTING,
    ExperimentStatus.DEPLOYING,
)


class ClaudeMonitor:
    """Periodic Claude-based anomaly detector for running experiments."""

    def __init__(self, config: AutoResearchConfig, registry: Registry,
                 claude: ClaudeRunner,
                 *,
                 interval_s: float = 180.0,
                 kill_after_repeats: int = 2):
        self.config = config
        self.registry = registry
        self.claude = claude
        self.interval_s = interval_s
        self.kill_after_repeats = kill_after_repeats
        self._stop = threading.Event()
        # exp_id -> count of consecutive "kill" verdicts
        self._kill_streak: dict[str, int] = defaultdict(int)
        # exp_id -> timestamp of last monitor run (avoid double-fire)
        self._last_checked: dict[str, float] = {}

    def run(self):
        log.info("Claude monitor starting (interval=%ds, kill_after=%d)",
                 self.interval_s, self.kill_after_repeats)
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                log.exception("ClaudeMonitor tick error: %s", e)
            self._stop.wait(self.interval_s)
        log.info("Claude monitor stopped")

    def stop(self):
        self._stop.set()

    # ── Core ─────────────────────────────────────────────────────────

    def _tick(self):
        # Gather live experiments across all active statuses
        live = []
        for st in _ACTIVE_STATUSES:
            live.extend(self.registry.list_experiments(status=st))

        now = time.time()
        for exp in live:
            # Skip if we checked this one very recently (cross-tick dedup)
            last = self._last_checked.get(exp.id, 0)
            if now - last < self.interval_s * 0.9:
                continue
            self._last_checked[exp.id] = now

            log_path = self._resolve_log_path(exp.id)
            if not log_path:
                # Nothing to inspect yet; skip
                continue

            elapsed_s = 0.0
            if exp.started_at:
                try:
                    # started_at is stored as a string; cheap parse
                    from datetime import datetime
                    started = datetime.fromisoformat(str(exp.started_at))
                    elapsed_s = (datetime.utcnow() - started).total_seconds()
                except Exception:
                    pass

            last_metric = {
                "screen_ema_bpb": exp.screen_ema_bpb,
                "screen_train_bpb": exp.screen_train_bpb,
                "screen_ms_per_step": exp.screen_ms_per_step,
                "gate_int6_bpb": exp.gate_int6_bpb,
                "status": exp.status.value,
            }

            try:
                spec = build_task(
                    "monitor_run",
                    config=self.config, registry=self.registry,
                    experiment_id=exp.id, log_path=str(log_path),
                    elapsed_s=elapsed_s, last_metric=last_metric,
                )
                # Fire async so one slow Claude call doesn't block the loop
                self.claude.run_async(
                    spec,
                    on_complete=lambda r, eid=exp.id: self._handle_verdict(eid, r),
                )
            except Exception as e:
                log.warning("monitor_run spawn failed for %s: %s", exp.id, e)

    def _handle_verdict(self, exp_id: str, result: ClaudeTaskResult):
        if not result.success or not result.parsed:
            log.debug("monitor_run %s returned no verdict", exp_id)
            return
        verdict = result.parsed.get("verdict", "unknown")
        reason = result.parsed.get("reason", "")
        log.info("monitor_run %s: verdict=%s reason=%s",
                 exp_id, verdict, reason[:200])

        # Emit an event regardless so the dashboard shows it
        try:
            from ..db.models import EventType
            self.registry.emit_event(
                EventType.EXP_STEP, "experiment", exp_id,
                {"source": "claude_monitor", "verdict": verdict,
                 "reason": reason[:500]},
            )
        except Exception:
            pass

        if verdict == "kill":
            self._kill_streak[exp_id] += 1
            if self._kill_streak[exp_id] >= self.kill_after_repeats:
                log.warning(
                    "ClaudeMonitor escalating %s to real kill "
                    "after %d consecutive kill verdicts: %s",
                    exp_id, self._kill_streak[exp_id], reason[:200],
                )
                try:
                    self.registry.enqueue_command(
                        "kill_experiment",
                        target_id=exp_id,
                        payload={"mode": "graceful",
                                 "grace_period_s": 10,
                                 "reason": f"claude_monitor: {reason[:200]}"},
                        issued_by="claude_monitor",
                    )
                except Exception as e:
                    log.exception("failed to enqueue kill: %s", e)
        else:
            # Reset streak on any non-kill verdict
            self._kill_streak[exp_id] = 0

    def _resolve_log_path(self, exp_id: str) -> Optional[Path]:
        """Locate the training log for an experiment.

        The scheduler syncs GPU-side logs into
        <repo_root>/experiment_logs/<exp_id>.log — we use that as the
        canonical path and fall back to None if the sync hasn't started.
        """
        workspace = Path(self.config.workspace_dir).parent
        candidate = workspace / "experiment_logs" / f"{exp_id}.log"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
        return None
