"""Multi-node experiment scheduler with user steering.

The scheduler runs on the Azure VM and dispatches jobs
to GPU nodes via the ClusterManager. It supports:
- Git-based deployment: commit+push → git pull on GPU → run
- Every experiment linked to a commit SHA for reproducibility
- Priority-based scheduling with user override
- Continuous log streaming from GPU nodes → experiment_logs/
- Live metrics parsing — DB updated as training progresses
- Health checks every 10s (NaN, OOM, stall, divergence)
- Stop/reprioritize any experiment at any time
- Promote consolidation (drain slots → full-node run)
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import AutoResearchConfig
from ..cluster.manager import ClusterManager
from ..tracing import tracer
from ..db.models import (
    Experiment, ExperimentStatus, EventType,
)
from ..db.registry import Registry
from ..ideas.tracker import IdeaTracker

log = logging.getLogger(__name__)

# Regex for parsing training output
_STEP_RE = re.compile(
    r"step\s+(\d+)/\d+.*?loss[:\s]+([\d.]+).*?lr[:\s]+([\d.e-]+).*?([\d.]+)\s*ms"
)
_TRAIN_BPB_RE = re.compile(r"(?:train_bpb|final_train_bpb)[:\s]+([\d.]+)")
_EMA_BPB_RE = re.compile(r"(?:ema_bpb|final_ema_bpb)[:\s]+([\d.]+)")
# train_gpt.py actually emits val_bpb periodically and final_int8_zlib_roundtrip val_bpb
# at the end. Use these as fallbacks for ema_bpb / int6_bpb which no longer exist.
_VAL_BPB_RE = re.compile(r"(?<!roundtrip\s)val_bpb[:\s]+([\d.]+)")
_FINAL_INT8_BPB_RE = re.compile(r"final_int8_zlib_roundtrip\s+val_loss:[\d.]+\s+val_bpb[:\s]+([\d.]+)")
_INT6_BPB_RE = re.compile(r"(?:int6_bpb|final_int6_bpb)[:\s]+([\d.]+)")
_QUANT_GAP_RE = re.compile(r"quant_gap[:\s]+([\d.]+)")
_ARTIFACT_MB_RE = re.compile(r"artifact.*?([\d.]+)\s*MB")


@dataclass
class _ActiveRun:
    experiment: Experiment
    stage: str
    host: str
    gpu_indices: list[int]
    started_at: float
    commit_sha: str = ""
    # Log streaming state
    log_byte_offset: int = 0
    local_log_path: Optional[Path] = None
    last_log_stream: float = 0
    # Health check state
    last_health_check: float = 0
    stall_since: Optional[float] = None
    last_step: int = 0
    last_loss: float = 0.0
    recent_losses: list[float] = field(default_factory=list)


class Scheduler:
    """Main scheduling loop with git deployment and continuous monitoring."""

    def __init__(self, config: AutoResearchConfig, registry: Registry,
                 cluster: ClusterManager, ideas: IdeaTracker):
        self.config = config
        self.registry = registry
        self.cluster = cluster
        self.ideas = ideas

        self._lock = threading.Lock()
        self._active_runs: dict[str, _ActiveRun] = {}  # exp_id -> run
        self._stop_event = threading.Event()
        self._promote_queue: list[str] = []  # exp_ids waiting for promote

        self.queue_dir = config.abs_queue_dir
        self.queue_dir.mkdir(parents=True, exist_ok=True)

        # experiment_logs/ at project root for all synced logs
        self._experiment_logs_dir = Path(config.workspace_dir).parent / "experiment_logs"
        self._experiment_logs_dir.mkdir(parents=True, exist_ok=True)

        # Detect local repo dir for git operations
        self._repo_dir = Path(config.deploy_repo_dir) if config.deploy_repo_dir else (
            Path(config.workspace_dir).parent
        )

        # Recipe store — used to roll experiment results onto recipe
        # best_val_bpb so the leaderboard reflects completed runs, and to
        # resolve baseline script_path + env_overrides at launch time.
        from ..db.recipes import RecipeStore
        self._recipes = RecipeStore(registry, config.abs_recipes_dir)

    # ── Main Loop ──────────────────────────────────────────────────────

    def run(self):
        """Start the scheduling loop (blocking)."""
        log.info("Scheduler starting, tick=%ss, health_check=%ss, log_stream=%ss",
                 self.config.tick_interval_s,
                 self.config.health_check_interval_s,
                 self.config.log_stream_interval_s)

        # Initial cluster discovery
        self.cluster.discover_all()
        summary = self.cluster.get_cluster_summary()
        log.info("Cluster: %d nodes online, %d GPUs total, %d free",
                 summary["online_nodes"], summary["total_gpus"], summary["free_gpus"])

        # Recover orphan runs left over from a prior scheduler process.
        try:
            self._recover_orphan_runs()
        except Exception as e:
            log.exception("orphan recovery failed: %s", e)

        tick_count = 0
        while not self._stop_event.is_set():
            try:
                self._tick(tick_count)
            except Exception as e:
                log.exception("Scheduler tick error: %s", e)

            tick_count += 1
            # Re-probe cluster every 6 ticks (~60s)
            if tick_count % 6 == 0:
                self.cluster.discover_all()

            self._stop_event.wait(self.config.tick_interval_s)

        log.info("Scheduler stopped")

    def stop(self):
        """Request scheduler shutdown."""
        self._stop_event.set()

    def _tick(self, tick_count: int):
        """One scheduling cycle."""
        # 0. Drain the command queue (dashboard → worker RPC)
        self._drain_commands()

        # 1. Scan queue directory for new YAML configs
        self._scan_queue()

        # 2. Stream logs from active runs → local experiment_logs/
        self._stream_logs()

        # 3. Check completions of active runs
        self._check_completions()

        # 4. Health-check active runs (NaN, OOM, stall, divergence)
        self._health_check()

        # 5. Schedule pending experiments
        self._schedule_pending()

    # ── Command Queue (dashboard → worker RPC) ─────────────────────────

    def _drain_commands(self):
        """Execute any pending commands enqueued by the dashboard process.

        Supported commands:
        - kill_experiment        payload: {"mode": "graceful"|"force",
                                            "reason": str,
                                            "grace_period_s": int}
        - prioritize_experiment  payload: {"priority": int}
        - promote_experiment     payload: {}
        - stop_all               payload: {}
        - poll_now               payload: {}  (re-poll PRs/papers)
        """
        cmds = self.registry.claim_pending_commands(limit=20)
        for cmd in cmds:
            try:
                self._execute_command(cmd)
            except Exception as e:
                log.exception("Command %s failed: %s", cmd["id"], e)
                self.registry.complete_command(
                    cmd["id"], result=f"error: {e}", failed=True)

    def _execute_command(self, cmd: dict):
        name = cmd["command"]
        target = cmd.get("target_id") or ""
        try:
            payload = json.loads(cmd.get("payload") or "{}")
        except Exception:
            payload = {}
        log.info("Executing command %s: %s target=%s", cmd["id"], name, target)

        if name == "kill_experiment":
            mode = payload.get("mode", "graceful")
            reason = payload.get("reason", "user request via dashboard")
            grace = int(payload.get("grace_period_s", 10))
            self.stop_experiment(target, reason=reason, mode=mode,
                                  grace_period_s=grace)
            self.registry.complete_command(
                cmd["id"], result=json.dumps({"killed": target, "mode": mode}))
        elif name == "prioritize_experiment":
            self.prioritize_experiment(target, int(payload.get("priority", 2)))
            self.registry.complete_command(cmd["id"], result="ok")
        elif name == "promote_experiment":
            self.promote_experiment(target)
            self.registry.complete_command(cmd["id"], result="ok")
        elif name == "stop_all":
            self.stop_all()
            self.registry.complete_command(cmd["id"], result="ok")
        else:
            self.registry.complete_command(
                cmd["id"], result=f"unknown command: {name}", failed=True)

    # ── Git Deployment ─────────────────────────────────────────────────

    def _get_current_commit(self) -> str:
        """Get the current HEAD commit SHA of the local repo."""
        try:
            r = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=str(self._repo_dir),
            )
            return r.stdout.strip() if r.returncode == 0 else ""
        except Exception:
            return ""

    def _get_deploy_branch(self) -> str:
        """Get the branch to deploy from."""
        if self.config.deploy_branch:
            return self.config.deploy_branch
        # Auto-detect current branch
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=str(self._repo_dir),
            )
            return r.stdout.strip() if r.returncode == 0 else "main"
        except Exception:
            return "main"

    def _ensure_pushed(self) -> str:
        """Ensure local changes are committed and pushed. Returns commit SHA.

        If deploy_auto_commit is True and there are uncommitted changes,
        auto-commits them with a descriptive message.
        """
        commit_sha = self._get_current_commit()
        branch = self._get_deploy_branch()

        if self.config.deploy_auto_commit:
            # Check for uncommitted changes
            try:
                r = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True, text=True, timeout=5,
                    cwd=str(self._repo_dir),
                )
                if r.stdout.strip():
                    # There are changes — auto-commit
                    subprocess.run(
                        ["git", "add", "-A"],
                        capture_output=True, timeout=10,
                        cwd=str(self._repo_dir),
                    )
                    subprocess.run(
                        ["git", "commit", "-m",
                         f"autoresearch: auto-commit before experiment deploy"],
                        capture_output=True, timeout=10,
                        cwd=str(self._repo_dir),
                    )
                    commit_sha = self._get_current_commit()
                    log.info("Auto-committed changes: %s", commit_sha[:8])
            except Exception as e:
                log.warning("Auto-commit failed: %s", e)

        # Push to remote
        try:
            r = subprocess.run(
                ["git", "push", "origin", branch],
                capture_output=True, text=True, timeout=30,
                cwd=str(self._repo_dir),
            )
            if r.returncode == 0:
                log.info("Pushed to origin/%s (%s)", branch, commit_sha[:8])
            else:
                log.warning("git push failed: %s", r.stderr.strip()[:200])
        except Exception as e:
            log.warning("git push failed: %s", e)

        return commit_sha

    # ── Log Streaming ──────────────────────────────────────────────────

    def _stream_logs(self):
        """Pull new log lines from all active runs and write locally.

        Runs every log_stream_interval_s. For each active run:
        1. Fetch new bytes from remote train.log since last offset
        2. Append to local experiment_logs/<idea>/<exp>/train.log
        3. Parse any new step metrics and update DB in real-time
        """
        now = time.time()
        with self._lock:
            active = list(self._active_runs.items())

        for exp_id, run in active:
            if now - run.last_log_stream < self.config.log_stream_interval_s:
                continue
            run.last_log_stream = now

            # Fetch new log content
            new_text, new_offset = self.cluster.fetch_log_incremental(
                exp_id, run.log_byte_offset)
            if not new_text:
                continue

            run.log_byte_offset = new_offset

            # Write to local experiment_logs/
            if self.config.log_sync_to_experiment_logs and run.local_log_path:
                try:
                    with open(run.local_log_path, "a") as f:
                        f.write(new_text)
                except Exception as e:
                    log.debug("Log write error for %s: %s", exp_id, e)

            # Parse live metrics from new lines
            self._parse_live_metrics(exp_id, run, new_text)

    def _parse_live_metrics(self, exp_id: str, run: _ActiveRun, new_text: str):
        """Parse step metrics from new log lines and update DB in real-time."""
        for line in new_text.split("\n"):
            m = _STEP_RE.search(line)
            if m:
                step = int(m.group(1))
                loss = float(m.group(2))
                ms_per_step = float(m.group(4))

                if step > run.last_step:
                    run.last_step = step
                    run.last_loss = loss
                    run.recent_losses.append(loss)
                    if len(run.recent_losses) > 100:
                        run.recent_losses = run.recent_losses[-100:]

                    # Emit step event (throttled — every 50 steps)
                    if step % 50 == 0:
                        self.registry.emit_event(
                            EventType.EXP_STEP, "experiment", exp_id,
                            {"step": step, "loss": loss,
                             "ms_per_step": ms_per_step,
                             "stage": run.stage},
                        )

    # ── Queue Scanning ─────────────────────────────────────────────────

    def _scan_queue(self):
        """Look for new YAML experiment configs in queue/."""
        for yaml_path in sorted(self.queue_dir.glob("*.yaml")):
            try:
                with open(yaml_path) as f:
                    cfg = yaml.safe_load(f)
                if not cfg or not cfg.get("idea_id"):
                    log.warning("Skipping %s: missing idea_id", yaml_path.name)
                    continue

                exp_id = cfg.get("id", yaml_path.stem)
                if self.registry.get_experiment(exp_id):
                    yaml_path.rename(yaml_path.with_suffix(".yaml.done"))
                    continue

                from ..db.models import ExperimentCategory
                exp = self.ideas.create_experiment(
                    idea_id=cfg["idea_id"],
                    name=cfg.get("name", exp_id),
                    env_overrides=cfg.get("env_overrides", {}),
                    category=ExperimentCategory(cfg.get("category", "other")),
                    hypothesis=cfg.get("hypothesis", ""),
                    parent_exp=cfg.get("parent_id"),
                    stages=cfg.get("stages", ["screen", "gate"]),
                    priority=cfg.get("priority", 2),
                    notes=cfg.get("notes", ""),
                )
                self.registry.update_experiment_status(exp.id, ExperimentStatus.QUEUED)
                self.registry.emit_event(EventType.EXP_QUEUED, "experiment", exp.id,
                                          {"from_file": yaml_path.name})
                yaml_path.rename(yaml_path.with_suffix(".yaml.done"))
                log.info("Queued experiment %s from %s", exp.id, yaml_path.name)
            except Exception as e:
                log.error("Error processing %s: %s", yaml_path.name, e)

    # ── Orphan Recovery ────────────────────────────────────────────────

    def _recover_orphan_runs(self):
        """Finalize experiments left in SCREENING/GATING/PROMOTING across
        a scheduler restart. In-memory _active_runs is empty at this
        point, so these runs would otherwise sit forever. Strategy: pull
        the remote train.log, parse metrics if present, and call the
        stage-completion handler. If no metrics can be parsed, mark
        FAILED so the loop makes forward progress."""
        active_statuses = (
            ExperimentStatus.SCREENING,
            ExperimentStatus.GATING,
            ExperimentStatus.PROMOTING,
            ExperimentStatus.DEPLOYING,
        )
        orphans = []
        for st in active_statuses:
            orphans.extend(self.registry.list_experiments(status=st))
        if not orphans:
            return
        log.warning("Recovering %d orphan experiment run(s)", len(orphans))
        for exp in orphans:
            try:
                stage = {
                    ExperimentStatus.SCREENING: "screen",
                    ExperimentStatus.GATING: "gate",
                    ExperimentStatus.PROMOTING: "promote",
                    ExperimentStatus.DEPLOYING: "screen",
                }.get(exp.status, "screen")
                # Build a minimal _ActiveRun stub so _handle_* helpers
                # can read stage/host/started_at.
                started_at = time.time()
                if exp.started_at:
                    try:
                        started_at = datetime.fromisoformat(
                            str(exp.started_at).replace(" ", "T")
                        ).timestamp()
                    except Exception:
                        pass
                stub = _ActiveRun(
                    experiment=exp,
                    stage=stage,
                    host=exp.node_host or "",
                    gpu_indices=exp.gpu_indices or [],
                    started_at=started_at,
                    commit_sha=exp.commit_sha or "",
                    local_log_path=None,
                    recent_losses=[],
                    last_step=0,
                )
                # Try to sync & parse the log.
                log_text = ""
                try:
                    self.cluster.sync_experiment_results(
                        exp.id,
                        self.config.abs_ideas_dir / exp.idea_id / exp.id,
                    )
                    log_text = self.cluster.get_log_tail(exp.id, lines=500) or ""
                except Exception as e:
                    log.warning("orphan %s: log fetch failed: %s", exp.id, e)
                metrics = self._parse_metrics(log_text) if log_text else {}
                if metrics:
                    log.info("orphan %s: recovered metrics %s",
                             exp.id, metrics)
                    try:
                        self.cluster.release_gpus(exp.id)
                    except Exception:
                        pass
                    if stage == "screen":
                        self._handle_screen_completion(exp.id, stub, metrics)
                    elif stage == "gate":
                        self._handle_gate_completion(exp.id, stub, metrics)
                    elif stage == "promote":
                        self._handle_promote_completion(exp.id, stub, metrics)
                else:
                    log.warning("orphan %s: no metrics, marking FAILED", exp.id)
                    try:
                        self.cluster.release_gpus(exp.id)
                    except Exception:
                        pass
                    self.registry.update_experiment_status(
                        exp.id, ExperimentStatus.FAILED,
                        "orphaned by scheduler restart, no metrics in log")
                    self.registry.emit_event(
                        EventType.EXP_FAILED, "experiment", exp.id,
                        {"reason": "orphaned_restart"})
            except Exception as e:
                log.exception("orphan recovery of %s failed: %s", exp.id, e)

    # ── Completion Checking ────────────────────────────────────────────

    def _check_completions(self):
        """Check if any active runs have finished."""
        with self._lock:
            active = list(self._active_runs.items())

        for exp_id, run in active:
            if not self.cluster.check_job_alive(exp_id):
                self._handle_completion(exp_id, run)

    def _handle_completion(self, exp_id: str, run: _ActiveRun):
        """Process a completed experiment run."""
        log.info("Run completed: %s (stage=%s, commit=%s)",
                 exp_id, run.stage, run.commit_sha[:8] if run.commit_sha else "unknown")

        # Final log sync — get everything
        local_dir = self.config.abs_ideas_dir / run.experiment.idea_id / exp_id
        self.cluster.sync_experiment_results(exp_id, local_dir)

        # Also sync full log to experiment_logs/
        if run.local_log_path:
            self.cluster.sync_log_file(exp_id, run.local_log_path)

        # Parse final results from log
        log_text = self.cluster.get_log_tail(exp_id, lines=500)
        metrics = self._parse_metrics(log_text)

        if run.stage == "screen":
            self._handle_screen_completion(exp_id, run, metrics)
        elif run.stage == "gate":
            self._handle_gate_completion(exp_id, run, metrics)
        elif run.stage == "promote":
            self._handle_promote_completion(exp_id, run, metrics)

        # Release GPUs
        self.cluster.release_gpus(exp_id)
        with self._lock:
            self._active_runs.pop(exp_id, None)

        # Record retroactive stage span covering deploy→completion. This
        # runs on the scheduler completion thread, not the launch thread,
        # so we use tracer.record() to write an already-closed row.
        exp_after = self.registry.get_experiment(exp_id)
        status_str = "ok"
        error = ""
        if exp_after is not None:
            s = str(exp_after.status.value) if hasattr(
                exp_after.status, "value") else str(exp_after.status)
            if s in ("failed", "rejected"):
                status_str = "error"
                error = exp_after.rejection_reason or ""
        tracer.record(
            kind=f"stage_{run.stage}",
            name=f"{run.stage}:{exp_id}",
            entity=("experiment", exp_id),
            started_at=run.started_at,
            ended_at=time.time(),
            status=status_str,
            attrs={
                "stage": run.stage,
                "host": run.host,
                "gpus": run.gpu_indices,
                "commit": (run.commit_sha or "")[:12],
                "idea_id": run.experiment.idea_id,
                **{k: v for k, v in (metrics or {}).items() if v is not None},
            },
            error=error,
        )

    def _handle_screen_completion(self, exp_id: str, run: _ActiveRun, metrics: dict):
        train_bpb = metrics.get("train_bpb")
        ema_bpb = metrics.get("ema_bpb")

        if train_bpb is None and ema_bpb is None:
            self.registry.update_experiment_status(
                exp_id, ExperimentStatus.FAILED, "No metrics in screen output")
            self.registry.emit_event(EventType.EXP_FAILED, "experiment", exp_id,
                                      {"stage": "screen", "reason": "no_metrics"})
            return

        self.registry.update_screen_results(
            exp_id,
            train_bpb=train_bpb,
            ema_bpb=ema_bpb,
            wallclock_s=time.time() - run.started_at,
            gpu_count=len(run.gpu_indices),
        )

        # Roll val_bpb proxy (ema_bpb) up to the recipe so it lands on the
        # leaderboard even before gate runs.
        self._roll_to_recipe(exp_id, val_bpb=ema_bpb)

        exp = self.registry.get_experiment(exp_id)
        if exp and "gate" in exp.stages:
            self.registry.update_experiment_status(exp_id, ExperimentStatus.QUEUED)
        else:
            self.registry.update_experiment_status(exp_id, ExperimentStatus.DONE)

        self.registry.emit_event(EventType.EXP_COMPLETED, "experiment", exp_id,
                                  {"stage": "screen", "commit": run.commit_sha, **metrics})
        self.ideas.log_experiment_event(
            run.experiment.idea_id, exp_id, "screen_done",
            {"commit": run.commit_sha, **metrics})

    def _handle_gate_completion(self, exp_id: str, run: _ActiveRun, metrics: dict):
        int6_bpb = metrics.get("int6_bpb")
        quant_gap = metrics.get("quant_gap")

        if int6_bpb is not None:
            gate_passed = quant_gap is not None and quant_gap < 0.15
            self.registry.update_gate_results(
                exp_id, int6_bpb=int6_bpb,
                quant_gap=quant_gap or 0,
                artifact_mb=metrics.get("artifact_mb", 0),
                gate_passed=gate_passed,
            )
            self._roll_to_recipe(
                exp_id, int6_bpb=int6_bpb,
                artifact_mb=metrics.get("artifact_mb"),
            )
            if gate_passed:
                exp = self.registry.get_experiment(exp_id)
                if exp and "promote" in exp.stages:
                    self._promote_queue.append(exp_id)
                    self.registry.update_experiment_status(exp_id, ExperimentStatus.QUEUED)
                else:
                    self.registry.update_experiment_status(exp_id, ExperimentStatus.DONE)
            else:
                self.registry.update_experiment_status(
                    exp_id, ExperimentStatus.REJECTED,
                    f"Gate failed: quant_gap={quant_gap}")
        else:
            self.registry.update_experiment_status(
                exp_id, ExperimentStatus.FAILED, "Gate produced no int6_bpb")

        self.registry.emit_event(EventType.EXP_COMPLETED, "experiment", exp_id,
                                  {"stage": "gate", "commit": run.commit_sha, **metrics})

    def _handle_promote_completion(self, exp_id: str, run: _ActiveRun, metrics: dict):
        self.registry.update_promote_results(
            exp_id,
            train_bpb=metrics.get("train_bpb"),
            ema_bpb=metrics.get("ema_bpb"),
            int6_bpb=metrics.get("int6_bpb"),
            sw_bpb=metrics.get("sw_bpb"),
            artifact_mb=metrics.get("artifact_mb"),
        )
        self._roll_to_recipe(
            exp_id,
            val_bpb=metrics.get("ema_bpb"),
            int6_bpb=metrics.get("int6_bpb"),
            artifact_mb=metrics.get("artifact_mb"),
        )
        self.registry.update_experiment_status(exp_id, ExperimentStatus.DONE)
        self.registry.emit_event(EventType.EXP_COMPLETED, "experiment", exp_id,
                                  {"stage": "promote", "commit": run.commit_sha, **metrics})

    # ── Health Checks (every 10s) ──────────────────────────────────────

    def _health_check(self):
        """Check active runs for stalls, NaN, OOM, divergence."""
        now = time.time()
        with self._lock:
            active = list(self._active_runs.items())

        for exp_id, run in active:
            if now - run.last_health_check < self.config.health_check_interval_s:
                continue
            run.last_health_check = now

            log_text = self.cluster.get_log_tail(exp_id, lines=50)

            # ── Stall detection ──
            if not log_text:
                if run.stall_since is None:
                    run.stall_since = now
                elif now - run.stall_since > self.config.stall_timeout_s:
                    log.warning("Stall detected for %s (no output for %ds), killing",
                                exp_id, int(now - run.stall_since))
                    self._kill_experiment(exp_id, "stall timeout")
                continue

            run.stall_since = None  # Got output, reset stall timer

            # ── NaN / Inf detection ──
            if "nan" in log_text.lower() or "inf" in log_text.lower():
                for line in log_text.split("\n"):
                    if re.search(r"loss[:\s]+(nan|inf)", line, re.IGNORECASE):
                        log.warning("NaN/Inf loss for %s, killing", exp_id)
                        self._kill_experiment(exp_id, "NaN/Inf loss")
                        break

            # ── OOM detection ──
            if "CUDA out of memory" in log_text or "OutOfMemoryError" in log_text:
                log.warning("OOM for %s, killing", exp_id)
                self._kill_experiment(exp_id, "OOM")
                continue

            # ── NCCL error detection ──
            nccl_patterns = ["NCCL timeout", "NCCL error", "ncclInternalError"]
            for pattern in nccl_patterns:
                if pattern in log_text:
                    log.warning("NCCL error for %s: %s, killing", exp_id, pattern)
                    self._kill_experiment(exp_id, f"NCCL: {pattern}")
                    break

            # ── Divergence detection (loss increasing for 50+ steps) ──
            if len(run.recent_losses) >= 50:
                window = run.recent_losses[-50:]
                if all(window[i] < window[i+1] for i in range(len(window)-1)):
                    log.warning("Divergence detected for %s (loss increasing 50 steps), killing",
                                exp_id)
                    self._kill_experiment(exp_id, "divergence (loss increasing)")

            # ── Wallclock timeout ──
            elapsed = now - run.started_at
            wallclock = {
                "screen": self.config.screen_wallclock_s,
                "gate": self.config.gate_wallclock_s,
                "promote": self.config.promote_wallclock_s,
            }.get(run.stage, 180)
            # Allow 20% grace period beyond wallclock
            if elapsed > wallclock * 1.2:
                log.warning("Wallclock timeout for %s (%ds > %ds), killing",
                            exp_id, int(elapsed), wallclock)
                self._kill_experiment(exp_id, f"wallclock timeout ({int(elapsed)}s)")

    # ── Scheduling ─────────────────────────────────────────────────────

    def _schedule_pending(self):
        """Launch queued experiments on available GPUs."""
        queued = self.registry.list_experiments(status=ExperimentStatus.QUEUED)
        if not queued:
            return

        # Check for promote jobs first (they need a full node)
        for exp_id in list(self._promote_queue):
            exp = self.registry.get_experiment(exp_id)
            if not exp or exp.status != ExperimentStatus.QUEUED:
                self._promote_queue.remove(exp_id)
                continue
            alloc = self.cluster.allocate_gpus(exp_id, 8)
            if alloc:
                self._promote_queue.remove(exp_id)
                host, indices = alloc
                self._launch_experiment(exp, "promote", host, indices)
                return  # Only one promote at a time

        # Schedule screen/gate jobs
        for exp in sorted(queued, key=lambda e: -e.priority):
            if exp.id in self._promote_queue:
                continue
            with self._lock:
                if exp.id in self._active_runs:
                    continue

            stage, gpus_needed = self._next_stage(exp)
            if not stage:
                continue

            alloc = self.cluster.allocate_gpus(exp.id, gpus_needed)
            if alloc:
                host, indices = alloc
                self._launch_experiment(exp, stage, host, indices)

    def _ensure_experiment_recipe(self, exp: Experiment) -> Optional[str]:
        """Bind `exp.recipe_id` to the current_best_baseline if missing.

        Every experiment should reference a recipe so (a) it runs on top
        of the current SOTA baseline script, and (b) its results roll up
        into the leaderboard. Experiments created via the dashboard or
        research agent often have no recipe — this lazily attaches the
        current best on first launch.
        """
        if exp.recipe_id:
            return exp.recipe_id
        try:
            best = self._recipes.current_best()
            if best is not None:
                self.registry.set_experiment_recipe(exp.id, best.id)
                exp.recipe_id = best.id
                return best.id
        except Exception as e:
            log.warning("ensure_experiment_recipe failed for %s: %s", exp.id, e)
        return None

    def _resolve_script_path(self, exp: Experiment) -> str:
        """Resolve the training script for an experiment.

        Preference: exp.recipe_id → recipe.script_path, else
        current_best_baseline, else exp.script_path, else root train_gpt.py.
        """
        try:
            self._ensure_experiment_recipe(exp)
            if exp.recipe_id:
                r = self._recipes.get(exp.recipe_id)
                if r and r.script_path:
                    return r.script_path
            best = self._recipes.current_best()
            if best and best.script_path:
                return best.script_path
        except Exception as e:
            log.warning("recipe script_path resolution failed for %s: %s",
                        exp.id, e)
        return exp.script_path or "train_gpt.py"

    def _resolve_env_overrides(self, exp: Experiment) -> dict:
        """Merge the resolved baseline recipe env with the experiment's env.

        Recipe env is the floor; experiment overrides are layered on top so
        an idea tuning LEARNING_RATE doesn't drop the SOTA recipe's
        NUM_LAYERS / MODEL_DIM / etc.
        """
        merged: dict = {}
        try:
            self._ensure_experiment_recipe(exp)
            source = None
            if exp.recipe_id:
                source = self._recipes.get(exp.recipe_id)
            if source is None:
                source = self._recipes.current_best()
            if source and source.env_overrides:
                merged.update(source.env_overrides)
        except Exception as e:
            log.warning("recipe env resolution failed for %s: %s", exp.id, e)
        if exp.env_overrides:
            merged.update(exp.env_overrides)
        return merged

    def _roll_to_recipe(
        self, exp_id: str,
        val_bpb: Optional[float] = None,
        int6_bpb: Optional[float] = None,
        artifact_mb: Optional[float] = None,
    ):
        """Push an experiment's metrics onto its recipe's best-so-far.

        Keeps the leaderboard (which sources from recipes) in sync with
        completed experiments. No-op if the experiment has no recipe_id.
        """
        try:
            exp = self.registry.get_experiment(exp_id)
            if not exp or not exp.recipe_id:
                return
            self._recipes.update_best_metrics(
                recipe_id=exp.recipe_id,
                experiment_id=exp_id,
                val_bpb=val_bpb,
                int6_bpb=int6_bpb,
                artifact_mb=artifact_mb,
            )
        except Exception as e:
            log.warning("_roll_to_recipe failed for %s: %s", exp_id, e)

    def _next_stage(self, exp: Experiment) -> tuple[Optional[str], int]:
        """Determine next stage and GPU count for an experiment.

        All stages default to 8 GPUs — on a single 8-GPU node experiments
        serialize, but each one trains under conditions comparable to the
        SOTA record (which was tuned on 8×H100 × 600s). Running gate on
        1 GPU only reached ~1608/20000 steps and gave a fundamentally
        broken signal.
        """
        if exp.screen_ema_bpb is None and "screen" in exp.stages:
            return "screen", self.config.screen_gpus_per_job
        if exp.gate_int6_bpb is None and "gate" in exp.stages and exp.screen_ema_bpb is not None:
            return "gate", self.config.gate_gpus_per_job
        if exp.promote_ema_bpb is None and "promote" in exp.stages and exp.gate_passed:
            return "promote", self.config.promote_gpus_per_job
        return None, 0

    def _launch_experiment(self, exp: Experiment, stage: str,
                            host: str, gpu_indices: list[int]):
        with tracer.span("experiment_launch",
                         name=f"{stage}:{exp.id}",
                         entity=("experiment", exp.id)) as _s:
            _s.set("stage", stage)
            _s.set("host", host)
            _s.set("gpus", gpu_indices)
            _s.set("idea_id", exp.idea_id)
            if exp.recipe_id:
                _s.set("recipe_id", exp.recipe_id)
            self._launch_experiment_impl(exp, stage, host, gpu_indices, _s)

    def _launch_experiment_impl(self, exp: Experiment, stage: str,
                                host: str, gpu_indices: list[int],
                                _span):
        """Deploy and start an experiment on a node.

        Flow:
        1. Commit & push current code (if auto_commit enabled)
        2. git pull on GPU node to get the exact code
        3. Launch torchrun with env overrides
        4. Record commit SHA for reproducibility
        5. Set up local log file for streaming
        """
        log.info("Launching %s stage=%s on %s GPUs %s",
                 exp.id, stage, host, gpu_indices)

        self.registry.assign_experiment_node(exp.id, host, gpu_indices)
        self.registry.update_experiment_status(exp.id, ExperimentStatus.DEPLOYING)

        # Step 1: Ensure code is committed and pushed
        commit_sha = self._ensure_pushed()

        wallclock = {
            "screen": self.config.screen_wallclock_s,
            "gate": self.config.gate_wallclock_s,
            "promote": self.config.promote_wallclock_s,
        }.get(stage, 180)

        # Step 2-3: Deploy via git pull + launch
        branch = self._get_deploy_branch()
        job_dir = self.config.abs_ideas_dir / exp.idea_id / exp.id

        # Resolve the training script: prefer the recipe's baseline file
        # (sota_fork writes decoded SOTAs to autoresearch/baselines/<id>/),
        # falling back to the experiment's own script_path and finally
        # the vanilla root train_gpt.py.
        script_path = self._resolve_script_path(exp)
        resolved_env = self._resolve_env_overrides(exp)

        success = self.cluster.deploy_experiment(
            exp.id, job_dir=job_dir, env_overrides=resolved_env,
            script=script_path, wallclock_s=wallclock,
            branch=branch,
        )

        if success:
            status = {
                "screen": ExperimentStatus.SCREENING,
                "gate": ExperimentStatus.GATING,
                "promote": ExperimentStatus.PROMOTING,
            }.get(stage, ExperimentStatus.SCREENING)

            self.registry.update_experiment_status(exp.id, status)

            # Step 4: Set up local log directory
            local_log_dir = self._experiment_logs_dir / exp.idea_id / exp.id
            local_log_dir.mkdir(parents=True, exist_ok=True)
            local_log_path = local_log_dir / "train.log"

            # Write experiment metadata alongside the log
            meta = {
                "experiment_id": exp.id,
                "idea_id": exp.idea_id,
                "stage": stage,
                "host": host,
                "gpus": gpu_indices,
                "commit_sha": commit_sha,
                "branch": branch,
                "script_path": script_path,
                "recipe_id": exp.recipe_id,
                "env_overrides": resolved_env,
                "wallclock_s": wallclock,
                "started_at": datetime.utcnow().isoformat(),
            }
            with open(local_log_dir / "experiment.json", "w") as f:
                json.dump(meta, f, indent=2)

            self.registry.emit_event(
                EventType.EXP_DEPLOYED, "experiment", exp.id,
                {"stage": stage, "host": host, "gpus": gpu_indices,
                 "commit": commit_sha, "branch": branch},
            )

            with self._lock:
                self._active_runs[exp.id] = _ActiveRun(
                    experiment=exp, stage=stage,
                    host=host, gpu_indices=gpu_indices,
                    started_at=time.time(),
                    commit_sha=commit_sha,
                    local_log_path=local_log_path,
                )

            log.info("Deployed %s: commit=%s branch=%s", exp.id,
                     commit_sha[:8] if commit_sha else "?", branch)
            _span.set("commit", commit_sha[:12] if commit_sha else "")
            _span.set("script_path", script_path)
        else:
            log.error("Failed to deploy %s", exp.id)
            self.cluster.release_gpus(exp.id)
            self.registry.update_experiment_status(
                exp.id, ExperimentStatus.FAILED, "Deploy failed")
            _span.fail("cluster.deploy_experiment returned False")

    # ── User Controls ──────────────────────────────────────────────────

    def stop_experiment(self, exp_id: str, reason: str = "user request",
                         mode: str = "graceful", grace_period_s: int = 10):
        """Stop a running experiment.

        mode: 'graceful' (default) sends SIGTERM and waits up to
              grace_period_s before SIGKILL. 'force' SIGKILLs immediately.
        """
        self._kill_experiment(exp_id, reason, mode=mode,
                               grace_period_s=grace_period_s)
        self.registry.emit_event(
            EventType.USER_STOP, "experiment", exp_id,
            {"reason": reason, "mode": mode})

    def prioritize_experiment(self, exp_id: str, priority: int):
        """Change experiment priority."""
        with self.registry._conn() as conn:
            conn.execute(
                "UPDATE experiments SET priority=? WHERE id=?",
                (priority, exp_id),
            )
        self.registry.emit_event(
            EventType.USER_PRIORITIZE, "experiment", exp_id,
            {"priority": priority},
        )

    def promote_experiment(self, exp_id: str):
        """Queue an experiment for full-budget promotion."""
        exp = self.registry.get_experiment(exp_id)
        if not exp:
            raise ValueError(f"Experiment not found: {exp_id}")
        if exp_id not in self._promote_queue:
            self._promote_queue.append(exp_id)
        self.registry.update_experiment_status(exp_id, ExperimentStatus.QUEUED)
        log.info("Queued %s for promotion", exp_id)

    def stop_all(self):
        """Stop all running experiments and shut down."""
        with self._lock:
            active_ids = list(self._active_runs.keys())
        for exp_id in active_ids:
            self._kill_experiment(exp_id, "stop_all")
        self.stop()

    # ── Internal ───────────────────────────────────────────────────────

    def _kill_experiment(self, exp_id: str, reason: str,
                          mode: str = "graceful", grace_period_s: int = 10):
        """Kill a running experiment and do final log sync."""
        # Final log sync before killing (best effort — don't block the kill)
        with self._lock:
            run = self._active_runs.get(exp_id)
        if run and run.local_log_path:
            try:
                self.cluster.sync_log_file(exp_id, run.local_log_path)
            except Exception as e:
                log.warning("Pre-kill log sync failed for %s: %s", exp_id, e)

        self.cluster.kill_experiment(exp_id, mode=mode,
                                      grace_period_s=grace_period_s)
        self.cluster.release_gpus(exp_id)
        with self._lock:
            self._active_runs.pop(exp_id, None)
        self.registry.update_experiment_status(
            exp_id, ExperimentStatus.STOPPED, f"{reason} ({mode})")

    def _parse_metrics(self, log_text: str) -> dict:
        """Extract final metrics from training log output."""
        metrics = {}
        for m in _TRAIN_BPB_RE.finditer(log_text):
            metrics["train_bpb"] = float(m.group(1))
        for m in _EMA_BPB_RE.finditer(log_text):
            metrics["ema_bpb"] = float(m.group(1))
        # Fallback: if no ema_bpb/train_bpb, use the last periodic val_bpb as
        # ema_bpb so screen completion passes. Current train_gpt.py only emits val_bpb.
        if "ema_bpb" not in metrics:
            last_val = None
            for m in _VAL_BPB_RE.finditer(log_text):
                last_val = float(m.group(1))
            if last_val is not None:
                metrics["ema_bpb"] = last_val
                metrics.setdefault("train_bpb", last_val)
        for m in _INT6_BPB_RE.finditer(log_text):
            metrics["int6_bpb"] = float(m.group(1))
        # Fallback: treat final_int8_zlib_roundtrip val_bpb as int6_bpb equivalent.
        if "int6_bpb" not in metrics:
            for m in _FINAL_INT8_BPB_RE.finditer(log_text):
                metrics["int6_bpb"] = float(m.group(1))
        for m in _QUANT_GAP_RE.finditer(log_text):
            metrics["quant_gap"] = float(m.group(1))
        # Fallback: quant_gap = int6/int8-quantized bpb minus unquantized ema/train bpb.
        # train_gpt.py emits both but doesn't compute the delta itself.
        if "quant_gap" not in metrics:
            ref = metrics.get("ema_bpb") or metrics.get("train_bpb")
            if ref is not None and metrics.get("int6_bpb") is not None:
                metrics["quant_gap"] = metrics["int6_bpb"] - ref
        for m in _ARTIFACT_MB_RE.finditer(log_text):
            metrics["artifact_mb"] = float(m.group(1))
        last_step = 0
        for m in _STEP_RE.finditer(log_text):
            s = int(m.group(1))
            if s > last_step:
                last_step = s
        if last_step:
            metrics["steps"] = last_step
        return metrics

    # ── Status ─────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Full scheduler status for dashboard."""
        with self._lock:
            active = {
                eid: {
                    "stage": r.stage,
                    "host": r.host,
                    "gpus": r.gpu_indices,
                    "elapsed_s": time.time() - r.started_at,
                    "commit": r.commit_sha[:8] if r.commit_sha else "",
                    "last_step": r.last_step,
                    "last_loss": r.last_loss,
                }
                for eid, r in self._active_runs.items()
            }
        return {
            "running": len(active),
            "promote_queue": list(self._promote_queue),
            "active_runs": active,
            "cluster": self.cluster.get_cluster_summary(),
            "stats": self.registry.get_stats(),
        }
