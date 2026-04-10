"""Catalog of Claude Code task types (prompts + schemas + allowlists).

Each entry in TASK_REGISTRY is a `ClaudeTask` — a pure-function builder
that takes the registry + config + whatever domain args the task needs,
and returns a `ClaudeTaskSpec` the runner can execute.

Adding a new task type is a single-file change: write a builder here,
give it a stable `name`, and register it. The scheduler / autoresearch
loop / dashboard reach tasks through `build_task(name, **kwargs)`.

Design notes
------------
- **Tool allowlists are strict.** A task that only writes a markdown
  report never gets Bash; a task that opens a PR gets Bash(gh:*) but
  not Edit on main.
- **Every prompt starts from the mission.** `append_system_prompt` is
  always set to `config.load_mission()` so the competition rules frame
  every Claude invocation.
- **Structured output is preferred.** Tasks with a clear return shape
  set `json_schema` so we get typed results instead of fuzzy markdown.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import AutoResearchConfig
from ..db.models import Experiment
from ..db.registry import Registry
from .runner import ClaudeTaskSpec

log = logging.getLogger(__name__)


@dataclass
class ClaudeTask:
    """Metadata + builder for one task type."""
    name: str
    description: str
    builder: Callable[..., ClaudeTaskSpec]
    default_tools: list[str]
    needs_worktree: bool = False


TASK_REGISTRY: dict[str, ClaudeTask] = {}


def register(name: str, description: str, tools: list[str],
              needs_worktree: bool = False):
    def _decorator(fn):
        TASK_REGISTRY[name] = ClaudeTask(
            name=name, description=description, builder=fn,
            default_tools=tools, needs_worktree=needs_worktree,
        )
        return fn
    return _decorator


def build_task(name: str, *, config: AutoResearchConfig,
                registry: Registry, **kwargs) -> ClaudeTaskSpec:
    """Build a task spec by name. Raises KeyError for unknown types."""
    task = TASK_REGISTRY[name]
    return task.builder(config=config, registry=registry, **kwargs)


def _common_system_prompt(config: AutoResearchConfig) -> str:
    """The shared mission prelude prepended to every task."""
    mission = config.load_mission() or ""
    return (
        "You are a task-scoped subagent in the Parameter Golf autoresearch "
        "system. Work only on the specific task described in the user "
        "message. Do not start additional experiments, do not open new "
        "branches unless instructed, and always respect the tool allowlist.\n\n"
        + mission
    )


# ── Task: write_report ───────────────────────────────────────────────


_WRITE_REPORT_SCHEMA = {
    "type": "object",
    "required": ["verdict", "summary", "report_path"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["win", "promising", "neutral", "regression", "broken"],
        },
        "summary": {"type": "string"},
        "delta_vs_baseline_bpb": {"type": ["number", "null"]},
        "key_findings": {"type": "array", "items": {"type": "string"}},
        "suggested_followups": {
            "type": "array", "items": {"type": "string"},
        },
        "report_path": {"type": "string"},
    },
}


@register(
    name="write_report",
    description="Write an experiment_logs/ markdown report from a finished run.",
    tools=["Read", "Grep", "Glob", "Write", "Bash"],
)
def _build_write_report(*, config: AutoResearchConfig, registry: Registry,
                         experiment: Experiment,
                         log_path: str = "",
                         recipe_id: str = "",
                         baseline_bpb: Optional[float] = None) -> ClaudeTaskSpec:
    exp_dict = {
        "id": experiment.id,
        "name": experiment.name,
        "hypothesis": experiment.hypothesis,
        "env_overrides": experiment.env_overrides,
        "recipe_id": recipe_id or getattr(experiment, "recipe_id", None),
        "screen_ema_bpb": experiment.screen_ema_bpb,
        "gate_int6_bpb": experiment.gate_int6_bpb,
        "gate_quant_gap": experiment.gate_quant_gap,
        "gate_artifact_mb": experiment.gate_artifact_mb,
        "gate_passed": experiment.gate_passed,
        "promote_ema_bpb": experiment.promote_ema_bpb,
        "promote_int6_bpb": experiment.promote_int6_bpb,
        "source_ref": getattr(experiment, "source_ref", ""),
        "is_reproduction": getattr(experiment, "is_reproduction", False),
    }
    import json as _json
    exp_json = _json.dumps(exp_dict, indent=2, default=str)

    report_dir = Path(config.workspace_dir).parent / "experiment_logs" / "claude_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    target_report = report_dir / f"{experiment.id}.md"

    prompt = f"""Write a markdown experiment report for the finished run below.

Experiment metadata:
```json
{exp_json}
```

Training log (if available): {log_path or '(not captured)'}
Baseline val_bpb for delta calc: {baseline_bpb if baseline_bpb is not None else 'unknown'}

Tasks:
1. Read the training log if the path exists and is non-empty — `Read` is allowed.
   Quote the key lines (final train_bpb, ema_bpb, gate metrics, any warnings).
2. Write the report to EXACTLY this path:
     {target_report}
   Sections required:
     - `# {experiment.name}` (h1 title)
     - `## Hypothesis` — restate in one paragraph
     - `## Configuration` — env_overrides table + recipe link
     - `## Results` — metrics table with delta vs baseline if baseline is known
     - `## Verdict` — one of: win / promising / neutral / regression / broken
     - `## Suggested follow-ups` — bulleted next experiments
3. Return a JSON object matching the required schema. `report_path` must be
   the absolute path you wrote.

Do NOT run new experiments, do NOT edit train_gpt.py, do NOT push code.
You may use Bash only for `ls`, `head`, `tail`, `wc` style inspection.
"""
    return ClaudeTaskSpec(
        task_type="write_report",
        target_id=experiment.id,
        prompt=prompt,
        cwd=str(Path(config.workspace_dir).parent),
        tools=TASK_REGISTRY["write_report"].default_tools,
        append_system_prompt=_common_system_prompt(config),
        json_schema=_WRITE_REPORT_SCHEMA,
        timeout_s=900,
        notes=f"report for {experiment.id}",
    )


# ── Task: assess_pr ──────────────────────────────────────────────────


_ASSESS_PR_SCHEMA = {
    "type": "object",
    "required": ["technique", "composable", "novelty", "recommendation"],
    "properties": {
        "technique": {"type": "string"},
        "files_touched": {"type": "array", "items": {"type": "string"}},
        "is_env_var_change": {"type": "boolean"},
        "estimated_bpb_delta": {"type": ["number", "null"]},
        "composable": {"type": "boolean"},
        "novelty": {
            "type": "string",
            "enum": ["known", "incremental", "novel", "maintainer_request"],
        },
        "recommendation": {
            "type": "string",
            "enum": ["reproduce", "stack_on_best", "ignore", "implement_clone"],
        },
        "new_feature_name": {"type": ["string", "null"]},
        "env_overrides": {"type": "object"},
        "risks": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
}


@register(
    name="assess_pr",
    description="Evaluate an open PR against the Parameter Golf mission.",
    tools=["Read", "Grep", "Glob", "Bash", "WebFetch"],
)
def _build_assess_pr(*, config: AutoResearchConfig, registry: Registry,
                      pr_number: int, pr_url: str = "",
                      repo: str = "") -> ClaudeTaskSpec:
    repo = repo or config.github_repo
    pr_url = pr_url or f"https://github.com/{repo}/pull/{pr_number}"
    prompt = f"""Evaluate PR #{pr_number} ({pr_url}) from {repo} for the Parameter Golf competition.

Use `gh` via Bash to fetch the PR diff, title, body, and review comments:
  gh pr view {pr_number} --repo {repo} --json title,body,files,additions,deletions,author,state
  gh pr diff {pr_number} --repo {repo}

Then answer (structured JSON, schema enforced):
- technique: one-line description of what the PR changes
- files_touched: list of files in the diff
- is_env_var_change: true if the PR is purely an env-var tune on top of the existing train_gpt.py
- estimated_bpb_delta: author-claimed improvement in BPB (negative = better) or null
- composable: does it stack on top of the current SOTA chain, or is it a parallel rewrite?
- novelty: known / incremental / novel / maintainer_request
- recommendation: reproduce (just rerun their config) / stack_on_best (merge their env change onto current best recipe) / ignore / implement_clone (rewrite cleanly on our branch)
- new_feature_name: canonical slug if this introduces a new feature ('swiglu', 'xsa4', etc.)
- env_overrides: dict of {{ENV_VAR: value}} that reproduces the PR behaviour
- risks: short bullet list of reproducibility / rule-compliance concerns
- notes: anything else a human should know in ≤3 sentences

Do NOT check out or run the PR code. This task is read-only.
"""
    return ClaudeTaskSpec(
        task_type="assess_pr",
        target_id=f"pr_{pr_number}",
        prompt=prompt,
        cwd=str(Path(config.workspace_dir).parent),
        tools=TASK_REGISTRY["assess_pr"].default_tools,
        append_system_prompt=_common_system_prompt(config),
        json_schema=_ASSESS_PR_SCHEMA,
        timeout_s=1200,
        notes=f"assess PR #{pr_number}",
    )


# ── Task: compose_recipe ─────────────────────────────────────────────


_COMPOSE_RECIPE_SCHEMA = {
    "type": "object",
    "required": ["name", "features", "env_overrides", "rationale"],
    "properties": {
        "name": {"type": "string"},
        "features": {"type": "array", "items": {"type": "string"}},
        "env_overrides": {"type": "object"},
        "rationale": {"type": "string"},
        "expected_bpb": {"type": ["number", "null"]},
        "base_recipe_id": {"type": "string"},
    },
}


@register(
    name="compose_recipe",
    description="Stack a new feature onto the current best baseline recipe.",
    tools=["Read", "Grep", "Glob"],
)
def _build_compose_recipe(*, config: AutoResearchConfig, registry: Registry,
                           base_recipe_id: str,
                           base_recipe_features: list[str],
                           base_env_overrides: dict,
                           new_feature: str,
                           new_env_overrides: dict,
                           rationale_hint: str = "") -> ClaudeTaskSpec:
    import json as _json
    base_feat_json = _json.dumps(base_recipe_features)
    base_env_json = _json.dumps(base_env_overrides, indent=2)
    new_env_json = _json.dumps(new_env_overrides, indent=2)
    prompt = f"""Propose a stacked recipe that adds the feature `{new_feature}` on top of
the current best baseline recipe `{base_recipe_id}`.

Base recipe features: {base_feat_json}
Base env_overrides:
```json
{base_env_json}
```

New feature env_overrides (raw proposal — may conflict with base):
```json
{new_env_json}
```

Rationale hint: {rationale_hint or '(none)'}

Respond with JSON matching the schema:
- name: human-readable recipe name, short, snake_case
- features: the merged feature list (base + new_feature, canonical)
- env_overrides: the merged env_overrides (new values overwrite base on conflict)
- rationale: 2-3 sentences on why this stack is worth testing
- expected_bpb: estimated val_bpb if you have a grounded guess, else null
- base_recipe_id: "{base_recipe_id}"

Do NOT edit files. This task is a pure planning step.
"""
    return ClaudeTaskSpec(
        task_type="compose_recipe",
        target_id=base_recipe_id,
        prompt=prompt,
        cwd=config.workspace_dir,
        tools=TASK_REGISTRY["compose_recipe"].default_tools,
        append_system_prompt=_common_system_prompt(config),
        json_schema=_COMPOSE_RECIPE_SCHEMA,
        timeout_s=600,
        notes=f"compose {new_feature} onto {base_recipe_id}",
    )


# ── Task: implement_technique ────────────────────────────────────────


_IMPLEMENT_TECHNIQUE_SCHEMA = {
    "type": "object",
    "required": ["branch", "feature_slug", "env_flag", "files_touched",
                 "summary"],
    "properties": {
        "branch": {"type": "string"},
        "feature_slug": {"type": "string"},
        "env_flag": {"type": "string"},
        "files_touched": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
        "test_command": {"type": "string"},
        "expected_default_behavior_preserved": {"type": "boolean"},
        "pr_url": {"type": ["string", "null"]},
    },
}


@register(
    name="implement_technique",
    description="Implement a new technique behind an env flag on a feature branch.",
    tools=["Read", "Grep", "Glob", "Edit", "Write", "Bash"],
    needs_worktree=True,
)
def _build_implement_technique(*, config: AutoResearchConfig, registry: Registry,
                                 technique_name: str,
                                 hypothesis: str,
                                 research_summary: str,
                                 env_flag: str,
                                 base_commit: str = "") -> ClaudeTaskSpec:
    import hashlib, re
    raw_slug = technique_name.lower().replace(" ", "_")
    raw_slug = re.sub(r"[^a-z0-9_\-]", "", raw_slug)
    # Claude Code worktree names are capped at 64 chars; "technique-" prefix
    # is 10, so slug must be <= 54. Truncate + hash-suffix to keep uniqueness.
    MAX_SLUG = 54
    if len(raw_slug) > MAX_SLUG:
        h = hashlib.sha1(raw_slug.encode()).hexdigest()[:6]
        slug = f"{raw_slug[:MAX_SLUG - 7]}-{h}"
    else:
        slug = raw_slug
    branch = f"auto/technique/{slug}"
    prompt = f"""Implement the technique "{technique_name}" in `train_gpt.py` as a new
feature flag, on a dedicated feature branch. NEVER push to main.

Hypothesis: {hypothesis}

Research summary from Parallel deep research / papers:
{research_summary}

Requirements:
1. You are running inside a git worktree that was created for this task.
   Create (or check out) the branch `{branch}` before editing.
   Base commit, if given: `{base_commit or '(current HEAD)'}`
2. Gate the new code path behind the env variable `{env_flag}`. When the
   variable is unset or "0", train_gpt.py MUST behave exactly as before.
   This is a HARD requirement — the default baseline must be preserved
   so scheduler reruns of existing recipes still work.
3. Keep the change minimal. Do not refactor unrelated code. Do not add
   docstrings to untouched functions. Do not bump dependencies.
4. Preserve the 16,000,000-byte (decimal) artifact constraint. Prefer
   parameter-efficient variants (LoRA, bias-only, etc.) when the naive
   implementation would blow the budget.
5. Run a quick compile smoke test: `python -c "import py_compile;
   py_compile.compile('train_gpt.py')"`. Fix any syntax errors before
   reporting success.
6. Commit with a clear message mentioning the env flag. Do NOT push yet
   — the autoresearch worker handles the push after verifying the diff.
7. Return JSON matching the schema, including:
     - branch: "{branch}"
     - feature_slug: "{slug}"
     - env_flag: "{env_flag}"
     - files_touched: list of files you modified
     - summary: 2-3 sentences
     - test_command: the torchrun / env command that exercises the new flag
     - expected_default_behavior_preserved: true if you verified
     - pr_url: null (opening the PR is a separate task)

ABSOLUTE rules:
- Do not edit anything outside train_gpt.py unless strictly necessary.
- Do not modify files under autoresearch/ or records/.
- Do not force-push, do not rebase main, do not delete branches.
"""
    return ClaudeTaskSpec(
        task_type="implement_technique",
        target_id=slug,
        prompt=prompt,
        cwd=str(Path(config.workspace_dir).parent),
        tools=TASK_REGISTRY["implement_technique"].default_tools,
        worktree=True,
        worktree_name=f"technique-{slug}",
        append_system_prompt=_common_system_prompt(config),
        json_schema=_IMPLEMENT_TECHNIQUE_SCHEMA,
        timeout_s=3600,
        notes=f"implement {technique_name}",
    )


# ── Task: reproduce_record ───────────────────────────────────────────


_REPRODUCE_RECORD_SCHEMA = {
    "type": "object",
    "required": ["recipe_proposal", "confidence"],
    "properties": {
        "recipe_proposal": {
            "type": "object",
            "required": ["name", "features", "env_overrides"],
            "properties": {
                "name": {"type": "string"},
                "features": {"type": "array", "items": {"type": "string"}},
                "env_overrides": {"type": "object"},
                "description": {"type": "string"},
                "base_commit": {"type": ["string", "null"]},
                "source_ref": {"type": "string"},
            },
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low", "unreproducible"],
        },
        "claimed_bpb": {"type": ["number", "null"]},
        "notes": {"type": "string"},
    },
}


@register(
    name="reproduce_record",
    description="Read a records/ submission and propose a recipe to rerun it.",
    tools=["Read", "Grep", "Glob", "Bash"],
)
def _build_reproduce_record(*, config: AutoResearchConfig, registry: Registry,
                              record_path: str) -> ClaudeTaskSpec:
    prompt = f"""Read the records/ submission at `{record_path}` and propose a recipe
that reproduces it on our infrastructure.

Tasks:
1. Read README.md, submission.json, train_gpt.py, and any logs in that
   directory. Use `Read` and `Bash(ls)` only — this is strictly read-only.
2. Extract the exact env variables / hyperparameters that define the run.
3. Return a JSON recipe_proposal with:
     - name: short slug derived from the submission
     - features: canonical feature tags (e.g. ["swiglu","xsa4","int6"])
     - env_overrides: env_var -> value dict to pass to train_gpt.py
     - description: 1-2 sentences
     - base_commit: git SHA if the submission pins one, else null
     - source_ref: "{record_path}"
4. Also return:
     - confidence: high / medium / low / unreproducible
     - claimed_bpb: the val_bpb the submission claims, or null
     - notes: reproducibility caveats (missing files, non-determinism, etc.)

Do NOT edit or delete any files. Do NOT start training.
"""
    return ClaudeTaskSpec(
        task_type="reproduce_record",
        target_id=record_path,
        prompt=prompt,
        cwd=str(Path(config.workspace_dir).parent),
        tools=TASK_REGISTRY["reproduce_record"].default_tools,
        append_system_prompt=_common_system_prompt(config),
        json_schema=_REPRODUCE_RECORD_SCHEMA,
        timeout_s=900,
        notes=f"reproduce {record_path}",
    )


# ── Task: monitor_run ────────────────────────────────────────────────


_MONITOR_RUN_SCHEMA = {
    "type": "object",
    "required": ["verdict", "reason"],
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["healthy", "suspicious", "kill", "unknown"],
        },
        "reason": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
}


@register(
    name="monitor_run",
    description="Claude-based anomaly detector for a running experiment.",
    tools=["Read", "Grep", "Bash"],
)
def _build_monitor_run(*, config: AutoResearchConfig, registry: Registry,
                        experiment_id: str, log_path: str,
                        elapsed_s: float, last_metric: dict) -> ClaudeTaskSpec:
    import json as _json
    metric_json = _json.dumps(last_metric, indent=2, default=str)
    prompt = f"""Inspect the running training job `{experiment_id}` and decide whether
it is healthy, suspicious, or should be killed.

Log path: {log_path}
Elapsed wallclock: {elapsed_s:.0f} s
Latest metric snapshot:
```json
{metric_json}
```

Allowed: Read, Grep, Bash (only `tail`, `head`, `wc`, `grep`, `ls` —
do NOT run torchrun or edit files). Look for:
- Loss diverging / NaN / sudden spikes
- Repeated identical steps (indicating hang)
- OOM warnings, CUDA errors, GPU fell off bus
- Broken data loader messages
- Any error Python rule-based health checks would miss

Respond with JSON:
- verdict: healthy / suspicious / kill / unknown
- reason: 1-2 sentences
- evidence: up to 5 quoted log lines supporting the verdict

Prefer "suspicious" over "kill" unless you are confident the run is
already dead. The scheduler will escalate repeated suspicious verdicts
to a real kill.
"""
    return ClaudeTaskSpec(
        task_type="monitor_run",
        target_id=experiment_id,
        prompt=prompt,
        cwd=str(Path(config.workspace_dir).parent),
        tools=TASK_REGISTRY["monitor_run"].default_tools,
        append_system_prompt=_common_system_prompt(config),
        json_schema=_MONITOR_RUN_SCHEMA,
        timeout_s=300,
        notes=f"monitor {experiment_id}",
    )
