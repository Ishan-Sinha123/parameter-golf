"""System configuration for the autoresearch system."""
from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GPUNodeConfig:
    """Configuration for a single GPU node."""
    host: str
    user: str = "azureuser"
    ssh_key: Optional[str] = None
    ssh_port: int = 22
    work_dir: str = "/workspace/parameter-golf"
    gpu_count: Optional[int] = None  # auto-detected if None
    label: str = ""  # e.g. "4xH100", "8xA100"
    enabled: bool = True
    env_setup: str = ""  # shell prefix, e.g. "source /workspace/parameter-golf/.venv/bin/activate"

    def __post_init__(self):
        if not self.label:
            self.label = self.host


@dataclass
class AutoResearchConfig:
    """Top-level autoresearch configuration."""
    # Paths (relative to autoresearch root)
    workspace_dir: str = "/home/azureuser/parameter-golf/autoresearch"
    db_path: str = "db/autoresearch.db"
    logs_dir: str = "logs"
    ideas_dir: str = "logs/ideas"
    queue_dir: str = "queue"
    recipes_dir: str = "recipes"

    # GPU cluster nodes
    nodes: list[GPUNodeConfig] = field(default_factory=list)

    # Scheduler
    tick_interval_s: float = 10.0
    # Scheduler hard-kill timeout per stage. Must exceed train_wallclock_s by
    # enough to cover process startup, torch.compile, final eval and ssh poll
    # lag (~180s on vast-h100).
    screen_wallclock_s: int = 780
    gate_wallclock_s: int = 780
    promote_wallclock_s: int = 780
    # Training-side MAX_WALLCLOCK_SECONDS passed into train_gpt.py. This is
    # the competition training budget (10 min = 600s). Scheduler kill is
    # larger to give the final eval room to finish.
    train_wallclock_s: int = 600
    screen_gpus_per_job: int = 8
    gate_gpus_per_job: int = 8
    promote_gpus_per_job: int = 8
    max_retries: int = 2
    stall_timeout_s: float = 120.0
    health_check_interval_s: float = 10.0   # how often to health-check active jobs
    log_stream_interval_s: float = 5.0      # how often to pull logs from GPU nodes
    log_sync_to_experiment_logs: bool = True # sync logs to experiment_logs/ dir

    # Git-based deployment
    deploy_branch: str = ""  # branch to push to and pull from on GPU nodes
    deploy_auto_commit: bool = True  # auto-commit before deploying
    deploy_repo_dir: str = ""  # local repo root (auto-detected if empty)

    # Research agent
    github_repo: str = "openai/parameter-golf"
    poll_interval_m: int = 30  # minutes between PR/paper polls
    arxiv_categories: list[str] = field(default_factory=lambda: ["cs.LG", "cs.CL"])

    # Parallel Web Systems API (deep web research)
    parallel_api_key: str = ""
    parallel_default_processor: str = "pro"  # pro, ultra, pro-fast, ultra-fast
    parallel_enabled: bool = False  # auto-enabled when api_key is set

    # Dashboard
    http_host: str = "0.0.0.0"
    http_port: int = 8765
    ws_port: int = 8766

    # Agent
    agent_enabled: bool = False
    agent_model: str = "claude-sonnet-4-6"

    # Claude Code subprocess runner (implement_technique, write_report, ...)
    claude_enabled: bool = True
    claude_bin: str = "claude"
    claude_model: str = "opus"
    claude_effort: str = "max"
    claude_max_concurrent: int = 3
    claude_auto_report: bool = True        # spawn write_report on DONE experiments
    claude_auto_assess_pr: bool = True     # spawn assess_pr on newly-seen PRs
    claude_auto_implement: bool = False    # spawn implement_technique automatically
    claude_auto_reproduce: bool = False    # spawn reproduce_record on scanned records
    claude_monitor_enabled: bool = True    # spawn monitor_run periodically

    # Mission statement — seeds the research agent's framing for all web
    # queries, PR evaluations, and idea proposals. Path is resolved relative
    # to workspace_dir if not absolute.
    mission_file: str = "prompts/mission.md"

    def load_mission(self) -> str:
        """Load the mission statement from disk. Returns empty string if missing."""
        p = Path(self.mission_file)
        if not p.is_absolute():
            p = Path(self.workspace_dir) / p
        if not p.exists():
            return ""
        try:
            return p.read_text()
        except Exception:
            return ""

    def __post_init__(self):
        # Auto-enable Parallel if key is provided
        if self.parallel_api_key and not self.parallel_enabled:
            self.parallel_enabled = True

    @property
    def abs_db_path(self) -> Path:
        return Path(self.workspace_dir) / self.db_path

    @property
    def abs_logs_dir(self) -> Path:
        return Path(self.workspace_dir) / self.logs_dir

    @property
    def abs_ideas_dir(self) -> Path:
        return Path(self.workspace_dir) / self.ideas_dir

    @property
    def abs_queue_dir(self) -> Path:
        return Path(self.workspace_dir) / self.queue_dir

    @property
    def abs_recipes_dir(self) -> Path:
        return Path(self.workspace_dir) / self.recipes_dir

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AutoResearchConfig":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        nodes = []
        for n in (raw.pop("nodes", None) or []):
            nodes.append(GPUNodeConfig(**n))

        cfg = cls(**{k: v for k, v in raw.items() if k != "nodes"})
        cfg.nodes = nodes
        return cfg

    def to_yaml(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {
            k: v for k, v in self.__dict__.items()
            if k != "nodes"
        }
        d["nodes"] = [n.__dict__ for n in self.nodes]
        with open(path, "w") as f:
            yaml.dump(d, f, default_flow_style=False, sort_keys=False)


# Alias for compat with runner/monitor
SystemConfig = AutoResearchConfig
