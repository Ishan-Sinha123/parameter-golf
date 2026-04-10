"""Data models for the autoresearch system."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Aliases (compat with autoresearch core/types) ──────────────────

ExperimentId = str
EnvOverrides = dict[str, str]


@dataclass
class MetricEvent:
    """Single training-step snapshot."""
    step: int = 0
    train_loss: float = 0.0
    lr: float = 0.0
    ms_step: float = 0.0
    gpu_mem_gb: float = 0.0
    tokens_seen: int = 0


@dataclass
class GPUSlot:
    """Represents an allocatable block of GPUs."""
    slot_id: int = 0
    gpu_indices: list[int] = field(default_factory=list)
    assigned_experiment: Optional[str] = None


# Stage enum for runner
class Stage(str, enum.Enum):
    SCREEN = "screen"
    GATE = "gate"
    PROMOTE = "promote"


@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen experiment config (compat with autoresearch runner)."""
    id: str = ""
    name: str = ""
    hypothesis: str = ""
    category: str = "other"
    priority: int = 2
    env_overrides: dict = field(default_factory=dict)
    stages: tuple = ("screen", "gate")
    parent_id: Optional[str] = None
    script_path: str = "train_gpt.py"
    expected_direction: str = "negative"
    reject_if_worse_by: float = 0.05
    inspect: bool = False
    notes: str = ""
    warnings: list = field(default_factory=list)


# ── Ideas ──────────────────────────────────────────────────────────────

class IdeaStatus(str, enum.Enum):
    PROPOSED = "proposed"       # Agent/human proposed it
    APPROVED = "approved"       # Human approved for experimentation
    ACTIVE = "active"           # Experiments running
    COMPLETED = "completed"     # All experiments done, conclusion reached
    PARKED = "parked"           # Paused for later
    REJECTED = "rejected"       # Not worth pursuing

class IdeaSource(str, enum.Enum):
    HUMAN = "human"
    AGENT = "agent"
    PAPER = "paper"
    GITHUB_PR = "github_pr"
    RECORD_MINING = "record_mining"
    WEB_RESEARCH = "web_research"


@dataclass
class Idea:
    id: str                     # e.g. "idea_001_swiglu"
    title: str                  # Short title
    hypothesis: str             # Testable hypothesis
    source: IdeaSource          # Where this idea came from
    source_ref: str = ""        # URL, paper ID, PR number
    status: IdeaStatus = IdeaStatus.PROPOSED
    priority: int = 2           # 1=low, 2=normal, 3=high, 4=critical
    parent_idea: Optional[str] = None
    tags: list[str] = field(default_factory=list)  # e.g. ["architecture", "quantization"]
    notes: str = ""
    evaluation: str = ""        # Critical evaluation notes
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Populated from DB
    experiment_ids: list[str] = field(default_factory=list)


# ── Experiments ────────────────────────────────────────────────────────

class ExperimentStatus(str, enum.Enum):
    DEFINED = "defined"
    QUEUED = "queued"
    DEPLOYING = "deploying"     # Being sent to GPU node
    SCREENING = "screening"
    GATING = "gating"
    INSPECTING = "inspecting"
    PROMOTING = "promoting"
    DONE = "done"
    REJECTED = "rejected"
    FAILED = "failed"
    STOPPED = "stopped"         # User manually stopped

class ExperimentCategory(str, enum.Enum):
    ARCHITECTURE = "architecture"
    HYPERPARAMETER = "hyperparameter"
    EVALUATION = "evaluation"
    TTT = "ttt"
    QUANTIZATION = "quantization"
    OTHER = "other"


@dataclass
class Experiment:
    id: str
    name: str
    idea_id: str                # Links to parent idea
    hypothesis: str = ""
    category: ExperimentCategory = ExperimentCategory.OTHER
    priority: int = 2
    status: ExperimentStatus = ExperimentStatus.DEFINED
    rejection_reason: str = ""

    # Config
    env_overrides: dict = field(default_factory=dict)
    script_path: str = "train_gpt.py"
    parent_id: Optional[str] = None  # Parent experiment for lineage
    stages: list[str] = field(default_factory=lambda: ["screen", "gate"])
    commit_sha: str = ""  # Git commit SHA that ran this experiment (for repro)
    recipe_id: Optional[str] = None  # Links to Recipe (env_overrides + features)
    is_reproduction: bool = False    # True if rerunning a records/ submission
    source_ref: str = ""             # PR #, branch name, or records path

    # Assignment
    node_host: Optional[str] = None
    gpu_indices: list[int] = field(default_factory=list)

    # Screen results
    screen_steps: Optional[int] = None
    screen_ms_per_step: Optional[float] = None
    screen_train_bpb: Optional[float] = None
    screen_ema_bpb: Optional[float] = None
    screen_gpu_count: Optional[int] = None
    screen_wallclock_s: Optional[float] = None

    # Gate results
    gate_int6_bpb: Optional[float] = None
    gate_quant_gap: Optional[float] = None
    gate_artifact_mb: Optional[float] = None
    gate_passed: Optional[bool] = None

    # Promote results
    promote_train_bpb: Optional[float] = None
    promote_ema_bpb: Optional[float] = None
    promote_int6_bpb: Optional[float] = None
    promote_sw_bpb: Optional[float] = None
    promote_artifact_mb: Optional[float] = None
    promote_steps: Optional[int] = None

    # Timestamps
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    notes: str = ""


# ── Recipes ────────────────────────────────────────────────────────────
#
# A Recipe is the canonical, immutable unit of experiment configuration.
# It captures: (a) a set of named features ("swiglu", "ttt_lora_r16"),
# (b) the env_overrides needed to enable them, and (c) the base git commit
# the training code was forked from. Recipes form a DAG via parent_recipe,
# which is how we model feature stacking ("baseline + swiglu + ttt").
#
# The `current_best_baseline` pointer (stored in recipe_pointers) tracks
# which recipe is the reigning champion; it updates automatically when an
# experiment beats the incumbent. New ideas are stacked onto the pointer,
# not onto main, so we chase the SOTA chain instead of starting over.


@dataclass
class Recipe:
    id: str                              # e.g. "rec_20260410_swiglu_ttt_r16"
    name: str                            # Human-readable
    description: str = ""
    parent_recipe: Optional[str] = None  # Recipe this one stacks on
    features: list[str] = field(default_factory=list)  # canonical sorted set
    feature_set_hash: str = ""           # sha256 of features + env (dedup key)
    env_overrides: dict = field(default_factory=dict)
    script_path: str = "train_gpt.py"
    base_commit: str = ""                # git SHA the code lives on
    base_branch: str = ""                # e.g. "main" or "auto/technique/swiglu"
    source_experiment: Optional[str] = None  # experiment that produced this
    # Best metrics observed for any experiment bound to this recipe
    best_val_bpb: Optional[float] = None
    best_int6_bpb: Optional[float] = None
    best_artifact_mb: Optional[float] = None
    best_experiment_id: Optional[str] = None
    created_at: Optional[datetime] = None
    yaml_path: str = ""                  # on-disk YAML mirror


# ── GPU Nodes ──────────────────────────────────────────────────────────

class NodeStatus(str, enum.Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    DRAINING = "draining"       # No new jobs, waiting for current to finish
    ERROR = "error"


@dataclass
class GPUInfo:
    index: int
    name: str = ""
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    utilization_pct: int = 0
    temperature_c: int = 0
    assigned_experiment: Optional[str] = None

    @property
    def memory_free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb

    @property
    def is_free(self) -> bool:
        return self.assigned_experiment is None


@dataclass
class NodeState:
    host: str
    label: str
    status: NodeStatus = NodeStatus.OFFLINE
    gpus: list[GPUInfo] = field(default_factory=list)
    last_heartbeat: Optional[datetime] = None
    error_message: str = ""

    @property
    def total_gpus(self) -> int:
        return len(self.gpus)

    @property
    def free_gpus(self) -> list[GPUInfo]:
        return [g for g in self.gpus if g.is_free]

    @property
    def free_gpu_count(self) -> int:
        return len(self.free_gpus)


# ── Events ─────────────────────────────────────────────────────────────

class EventType(str, enum.Enum):
    # Idea events
    IDEA_CREATED = "idea_created"
    IDEA_STATUS_CHANGE = "idea_status_change"
    IDEA_EVALUATION = "idea_evaluation"

    # Experiment events
    EXP_CREATED = "exp_created"
    EXP_QUEUED = "exp_queued"
    EXP_DEPLOYED = "exp_deployed"
    EXP_STARTED = "exp_started"
    EXP_STEP = "exp_step"
    EXP_COMPLETED = "exp_completed"
    EXP_FAILED = "exp_failed"
    EXP_REJECTED = "exp_rejected"
    EXP_STOPPED = "exp_stopped"

    # Node events
    NODE_ONLINE = "node_online"
    NODE_OFFLINE = "node_offline"
    NODE_ERROR = "node_error"

    # Research events
    PR_FOUND = "pr_found"
    PR_EVALUATED = "pr_evaluated"
    PAPER_FOUND = "paper_found"
    SOTA_UPDATE = "sota_update"
    WEB_SEARCH = "web_search"
    DEEP_RESEARCH = "deep_research"
    KNOWLEDGE_STORED = "knowledge_stored"

    # User events
    USER_STEER = "user_steer"
    USER_STOP = "user_stop"
    USER_PRIORITIZE = "user_prioritize"
