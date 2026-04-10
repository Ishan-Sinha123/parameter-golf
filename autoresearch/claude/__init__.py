"""Claude Code subprocess integration.

This subpackage owns how the autoresearch system delegates open-ended
work to Claude Code (the CLI). All the "agent does git" tasks —
write_report, assess_pr, compose_recipe, implement_technique,
reproduce_record, monitor_run — go through the `ClaudeRunner` here,
which spawns `claude -p` subprocesses with strict tool allowlists and
(where applicable) per-task git worktrees.

Keep the runner boring: no business logic, just "spawn, enforce the
sandbox, collect the output." The tasks module is where prompts +
schemas live, so adding a new task is a one-file change.
"""
from .runner import ClaudeRunner, ClaudeTaskResult, ClaudeTaskSpec
from .tasks import TASK_REGISTRY, build_task, ClaudeTask
from .monitor import ClaudeMonitor

__all__ = [
    "ClaudeRunner",
    "ClaudeTaskResult",
    "ClaudeTaskSpec",
    "TASK_REGISTRY",
    "build_task",
    "ClaudeTask",
    "ClaudeMonitor",
]
