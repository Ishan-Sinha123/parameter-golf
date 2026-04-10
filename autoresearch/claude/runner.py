"""Spawn Claude Code subprocesses for autonomous agent tasks.

The runner is the only place that actually shells out to `claude`. It
wraps Claude Code's headless `-p` mode with the flags we always want:

- `--model opus` + `--effort max` (per user config)
- `--permission-mode bypassPermissions` so Claude can `git`, `gh`, etc.
  without prompting (the Azure VM this runs on is the sandbox — the
  worker process shouldn't require human approval clicks)
- `--tools <allowlist>` per task so we never accidentally give a
  monitor-only task the ability to push code
- `--append-system-prompt` carrying the Parameter Golf mission, so
  every Claude invocation is framed by the competition rules
- Optional `--worktree` when the task needs an isolated git branch
  (e.g. implement_technique forking to auto/technique/<slug>)
- `--json-schema` when the task has a structured result contract

Concurrency is enforced per-process via a semaphore but also logged to
the SQLite `claude_tasks` table so cross-process observers (the
dashboard, the monitor) can see what's running.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from ..config import AutoResearchConfig
from ..db.registry import Registry

log = logging.getLogger(__name__)


# ── Task specification ───────────────────────────────────────────────


@dataclass
class ClaudeTaskSpec:
    """Everything the runner needs to spawn one Claude subprocess.

    Tasks fill this in; the runner treats it as opaque data and only
    knows how to translate it into CLI flags.
    """
    # Human-facing identity
    task_type: str                  # "write_report", "assess_pr", ...
    target_id: str                  # e.g. experiment id, PR number
    prompt: str                     # full user prompt to Claude

    # Execution
    cwd: str                        # working directory for the subprocess
    tools: list[str] = field(default_factory=lambda: ["Read", "Grep", "Glob"])
    worktree: bool = False          # create an isolated git worktree
    worktree_name: str = ""         # optional explicit worktree name
    append_system_prompt: str = ""  # prepended to every prompt (mission)
    json_schema: Optional[dict] = None  # structured output contract
    timeout_s: int = 1800           # hard kill after this many seconds
    extra_args: list[str] = field(default_factory=list)

    # Metadata — stored on the claude_tasks row for the dashboard
    notes: str = ""


@dataclass
class ClaudeTaskResult:
    """What came back from a Claude subprocess."""
    task_id: str                    # our internal id (uuid)
    task_type: str
    target_id: str
    exit_code: int
    duration_s: float
    stdout: str = ""
    stderr: str = ""
    parsed: Optional[dict] = None   # json-parsed stdout if schema enforced
    worktree_path: Optional[str] = None
    branch: Optional[str] = None
    error: str = ""

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.error


# ── Registry-side schema for tracking claude runs ───────────────────
#
# We keep this local to the runner module on purpose: it's the one
# subsystem that mutates this table. Dashboard + monitor can still
# query it read-only via registry._conn().


_CLAUDE_TASKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS claude_tasks (
    id              TEXT PRIMARY KEY,
    task_type       TEXT NOT NULL,
    target_id       TEXT DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|failed|killed
    started_at      DATETIME,
    completed_at    DATETIME,
    duration_s      REAL,
    exit_code       INTEGER,
    worktree_path   TEXT DEFAULT '',
    branch          TEXT DEFAULT '',
    stdout_path     TEXT DEFAULT '',
    stderr_path     TEXT DEFAULT '',
    parsed_json     TEXT DEFAULT '',
    error           TEXT DEFAULT '',
    notes           TEXT DEFAULT '',
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_claude_tasks_status
    ON claude_tasks(status, created_at);
CREATE INDEX IF NOT EXISTS idx_claude_tasks_target
    ON claude_tasks(target_id);
"""


def ensure_claude_tasks_schema(registry: Registry):
    """Make sure the claude_tasks table exists (idempotent)."""
    with registry._conn() as conn:
        conn.executescript(_CLAUDE_TASKS_SCHEMA)


# ── Runner ───────────────────────────────────────────────────────────


class ClaudeRunner:
    """Spawns Claude Code subprocesses with strict guardrails.

    Thread-safe and designed to be shared by the scheduler, the
    autoresearch loop, the PR ingester, and the monitor agent.
    """

    def __init__(self, config: AutoResearchConfig, registry: Registry,
                 *,
                 claude_bin: str = "claude",
                 model: str = "opus",
                 effort: str = "max",
                 max_concurrent: int = 3,
                 artifacts_dir: Optional[Path] = None):
        self.config = config
        self.registry = registry
        self.claude_bin = claude_bin
        self.model = model
        self.effort = effort
        self.max_concurrent = max_concurrent
        self.artifacts_dir = Path(artifacts_dir or
            Path(config.workspace_dir) / "logs" / "claude_tasks")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Bounded concurrency across the process. Cross-process limiting
        # happens by convention: only the worker spawns Claude tasks,
        # and the dashboard enqueues commands that the worker picks up.
        self._sem = threading.BoundedSemaphore(max_concurrent)
        self._running: dict[str, subprocess.Popen] = {}
        self._running_lock = threading.Lock()

        ensure_claude_tasks_schema(registry)

    # ── Public API ────────────────────────────────────────────────────

    def run(self, spec: ClaudeTaskSpec,
             on_complete: Optional[Callable[[ClaudeTaskResult], None]] = None,
             ) -> ClaudeTaskResult:
        """Run a task to completion (blocking, respects the semaphore)."""
        task_id = self._new_task_id(spec.task_type)
        self._insert_task_row(task_id, spec)

        with self._sem:
            self._mark_started(task_id)
            try:
                result = self._execute(task_id, spec)
            except Exception as e:
                log.exception("Claude task %s crashed: %s", task_id, e)
                result = ClaudeTaskResult(
                    task_id=task_id, task_type=spec.task_type,
                    target_id=spec.target_id, exit_code=-1,
                    duration_s=0.0, error=str(e),
                )
            self._mark_completed(task_id, result)

        if on_complete:
            try:
                on_complete(result)
            except Exception as e:
                log.exception("on_complete callback failed: %s", e)
        return result

    def run_async(self, spec: ClaudeTaskSpec,
                  on_complete: Optional[Callable[[ClaudeTaskResult], None]] = None,
                  ) -> threading.Thread:
        """Fire-and-forget a task on a background thread."""
        t = threading.Thread(
            target=self.run, args=(spec, on_complete),
            daemon=True, name=f"claude-{spec.task_type}",
        )
        t.start()
        return t

    def kill(self, task_id: str) -> bool:
        """Best-effort kill of a running claude subprocess."""
        with self._running_lock:
            proc = self._running.get(task_id)
        if not proc or proc.poll() is not None:
            return False
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            with contextlib.suppress(Exception):
                proc.terminate()
        return True

    # ── Execution core ────────────────────────────────────────────────

    def _execute(self, task_id: str, spec: ClaudeTaskSpec) -> ClaudeTaskResult:
        t0 = time.time()

        argv = self._build_argv(spec)
        env = os.environ.copy()
        # Make sure the worker's python env doesn't leak unexpected vars
        # into the child claude process
        env.pop("PYTHONPATH", None)

        stdout_path = self.artifacts_dir / f"{task_id}.stdout"
        stderr_path = self.artifacts_dir / f"{task_id}.stderr"

        log.info("Launching Claude task %s: type=%s target=%s tools=%s worktree=%s",
                 task_id, spec.task_type, spec.target_id,
                 ",".join(spec.tools), spec.worktree)
        log.debug("argv=%s cwd=%s", argv, spec.cwd)

        proc = subprocess.Popen(
            argv,
            cwd=spec.cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,  # own process group so we can SIGTERM the tree
        )
        with self._running_lock:
            self._running[task_id] = proc

        try:
            # We pass the prompt on stdin instead of argv to avoid hitting
            # shell argv length limits for long prompts and to keep quoting
            # simple.
            stdout, stderr = proc.communicate(
                input=spec.prompt, timeout=spec.timeout_s,
            )
            exit_code = proc.returncode
            error = ""
        except subprocess.TimeoutExpired:
            log.warning("Claude task %s timed out after %ds, killing", task_id,
                        spec.timeout_s)
            with contextlib.suppress(Exception):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            stdout, stderr = proc.communicate()
            exit_code = -9
            error = f"timeout after {spec.timeout_s}s"
        finally:
            with self._running_lock:
                self._running.pop(task_id, None)

        duration = time.time() - t0

        # Persist full logs to artifact files for debugging
        try:
            stdout_path.write_text(stdout or "")
            stderr_path.write_text(stderr or "")
        except Exception as e:
            log.warning("failed to write claude artifacts: %s", e)

        parsed = None
        if spec.json_schema and stdout:
            parsed = self._try_parse_json(stdout)

        # Detect branch/worktree from stdout if the task created one.
        # Claude Code prints "Created worktree at <path> on branch <name>"
        # when --worktree is passed. We parse it loosely.
        worktree_path, branch = None, None
        if spec.worktree:
            worktree_path, branch = self._detect_worktree(stdout or "")

        result = ClaudeTaskResult(
            task_id=task_id, task_type=spec.task_type,
            target_id=spec.target_id, exit_code=exit_code,
            duration_s=duration, stdout=stdout or "", stderr=stderr or "",
            parsed=parsed, worktree_path=worktree_path, branch=branch,
            error=error,
        )
        log.info("Claude task %s finished: exit=%d dur=%.1fs %s",
                 task_id, exit_code, duration,
                 "OK" if result.success else f"FAIL({error or 'nonzero'})")
        return result

    def _build_argv(self, spec: ClaudeTaskSpec) -> list[str]:
        """Turn a spec into the actual `claude ...` command line."""
        argv = [
            self.claude_bin, "-p",
            "--model", self.model,
            "--effort", self.effort,
            "--permission-mode", "bypassPermissions",
            "--allow-dangerously-skip-permissions",
            "--output-format", "json",
        ]

        if spec.tools:
            argv.extend(["--tools", ",".join(spec.tools)])

        if spec.append_system_prompt:
            argv.extend(["--append-system-prompt", spec.append_system_prompt])

        if spec.json_schema:
            argv.extend(["--json-schema", json.dumps(spec.json_schema)])

        if spec.worktree:
            if spec.worktree_name:
                argv.extend(["--worktree", spec.worktree_name])
            else:
                argv.append("--worktree")

        argv.extend(spec.extra_args)
        return argv

    # ── Parsing helpers ───────────────────────────────────────────────

    def _try_parse_json(self, text: str) -> Optional[dict]:
        """Claude Code with --output-format=json emits {...} on stdout.

        Precedence for the model's schema-structured output:
          1. ``structured_output`` — populated when --json-schema is used
             and the model emitted a valid structured reply via tool use
             (text ``result`` will be empty in this case).
          2. ``result`` (string) — normal prose response, may itself be
             a JSON blob we want to parse.
          3. ``result`` (dict) — already parsed dict.
          4. outer envelope as a last-resort fallback.
        """
        try:
            outer = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(outer, dict):
            return None
        structured = outer.get("structured_output")
        if isinstance(structured, dict) and structured:
            return structured
        result = outer.get("result")
        if isinstance(result, str) and result.strip():
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"_raw": result}
        if isinstance(result, dict):
            return result
        return outer

    def _detect_worktree(self, stdout: str) -> tuple[Optional[str], Optional[str]]:
        """Best-effort extraction of worktree path + branch from claude output.

        We don't hard-require this — the task prompt can ask Claude to
        echo the branch name in its JSON result for reliability.
        """
        path, branch = None, None
        for line in stdout.splitlines():
            low = line.lower()
            if "worktree" in low and "/" in line:
                for tok in line.split():
                    if tok.startswith("/"):
                        path = tok.rstrip(".,")
                        break
            if "branch" in low and path and branch is None:
                parts = line.split()
                for i, tok in enumerate(parts):
                    if tok.lower().startswith("branch") and i + 1 < len(parts):
                        branch = parts[i + 1].rstrip(".,")
                        break
        return path, branch

    # ── DB bookkeeping ────────────────────────────────────────────────

    def _new_task_id(self, task_type: str) -> str:
        return f"ct_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{task_type}_{uuid.uuid4().hex[:6]}"

    def _insert_task_row(self, task_id: str, spec: ClaudeTaskSpec):
        with self.registry._conn() as conn:
            conn.execute(
                """INSERT INTO claude_tasks
                   (id, task_type, target_id, status, notes)
                   VALUES (?, ?, ?, 'pending', ?)""",
                (task_id, spec.task_type, spec.target_id, spec.notes or ""),
            )

    def _mark_started(self, task_id: str):
        with self.registry._conn() as conn:
            conn.execute(
                """UPDATE claude_tasks
                   SET status='running', started_at=CURRENT_TIMESTAMP
                   WHERE id=?""",
                (task_id,),
            )

    def _mark_completed(self, task_id: str, result: ClaudeTaskResult):
        status = "done" if result.success else (
            "killed" if result.exit_code == -9 else "failed"
        )
        stdout_path = self.artifacts_dir / f"{task_id}.stdout"
        stderr_path = self.artifacts_dir / f"{task_id}.stderr"
        with self.registry._conn() as conn:
            conn.execute(
                """UPDATE claude_tasks
                   SET status=?, completed_at=CURRENT_TIMESTAMP,
                       duration_s=?, exit_code=?, worktree_path=?,
                       branch=?, stdout_path=?, stderr_path=?,
                       parsed_json=?, error=?
                   WHERE id=?""",
                (
                    status, result.duration_s, result.exit_code,
                    result.worktree_path or "", result.branch or "",
                    str(stdout_path), str(stderr_path),
                    json.dumps(result.parsed) if result.parsed else "",
                    result.error, task_id,
                ),
            )

    # ── Observability ─────────────────────────────────────────────────

    def list_tasks(self, status: Optional[str] = None,
                    limit: int = 100) -> list[dict]:
        with self.registry._conn() as conn:
            if status:
                rows = conn.execute(
                    """SELECT * FROM claude_tasks WHERE status=?
                       ORDER BY created_at DESC LIMIT ?""",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT * FROM claude_tasks
                       ORDER BY created_at DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_task(self, task_id: str) -> Optional[dict]:
        with self.registry._conn() as conn:
            row = conn.execute(
                "SELECT * FROM claude_tasks WHERE id=?", (task_id,),
            ).fetchone()
            return dict(row) if row else None
