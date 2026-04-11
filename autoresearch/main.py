#!/usr/bin/env python3
"""Main entry point for the Parameter Golf autoresearch system.

Recommended deployment:
  # Terminal 1 — the worker (scheduler + research agent + autoresearch loop).
  # Long-running. Only restart when you change core logic.
  python -m autoresearch.main worker --config config.yaml

  # Terminal 2 — the dashboard. Safe to restart freely; worker keeps running.
  python -m autoresearch.main dashboard --config config.yaml

All other usage:
  # Legacy all-in-one (worker + dashboard in one process).
  python -m autoresearch.main daemon --config config.yaml

  # Seed demo data for testing without GPUs
  python -m autoresearch.main seed-demo

  # Show cluster status
  python -m autoresearch.main cluster-status

  # Create an idea from CLI
  python -m autoresearch.main create-idea --title "..." --hypothesis "..."

  # Import historical experiments from autoresearch
  python -m autoresearch.main import-v2 ../autoresearch/db/registry.db
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path

from .config import AutoResearchConfig, GPUNodeConfig
from .db.registry import Registry
from .db.knowledge import KnowledgeBase
from .cluster.manager import ClusterManager
from .ideas.tracker import IdeaTracker
from .scheduler.scheduler import Scheduler
from .research.agent import ResearchAgent
from .research.autoresearch_loop import AutoResearchLoop
from .claude import ClaudeRunner, ClaudeMonitor
from .api import server as api_server

log = logging.getLogger("autoresearch")


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_core(config: AutoResearchConfig):
    """Build the core stateful components shared by worker and dashboard.

    All components are backed by the on-disk registry (SQLite WAL) and the
    LanceDB knowledge store, both of which support concurrent readers across
    processes. That's what lets the worker and dashboard run independently.
    """
    registry = Registry(config.abs_db_path)
    # Initialize the span tracer against the same DB so spans land in the
    # traces table. Safe to call multiple times.
    from .tracing import init as _tracing_init
    _tracing_init(config.abs_db_path)
    kb_path = Path(config.workspace_dir) / "db" / "knowledge_lance"
    knowledge = KnowledgeBase(kb_path)
    cluster = ClusterManager(config, registry)
    ideas = IdeaTracker(config, registry)
    claude = None
    if config.claude_enabled:
        try:
            claude = ClaudeRunner(
                config, registry,
                claude_bin=config.claude_bin,
                model=config.claude_model,
                effort=config.claude_effort,
                max_concurrent=config.claude_max_concurrent,
            )
        except Exception as e:
            log.warning("ClaudeRunner init failed, disabling: %s", e)
            claude = None
    return registry, knowledge, cluster, ideas, claude


def cmd_worker(args, config: AutoResearchConfig):
    """Run scheduler + research agent + autoresearch loop, no dashboard.

    This is the long-running process that actually manages GPU jobs and
    generates ideas. You should only restart it when changing scheduler
    or research-agent logic — dashboard tweaks do NOT require a restart.
    """
    registry, knowledge, cluster, ideas, claude = _build_core(config)
    scheduler = Scheduler(config, registry, cluster, ideas)
    research = ResearchAgent(config, registry, ideas, knowledge, claude=claude)
    autoresearch = AutoResearchLoop(
        config, registry, ideas, research, knowledge, claude=claude,
    )

    # Start research agent in background (polls PRs/papers)
    if config.agent_enabled:
        t = threading.Thread(target=research.run, daemon=True, name="research-agent")
        t.start()
        log.info("Research agent started (poll every %dm)", config.poll_interval_m)
    else:
        log.info("Research agent disabled (agent_enabled=false)")

    # Start autoresearch loop in background
    t = threading.Thread(target=autoresearch.run, daemon=True, name="autoresearch-loop")
    t.start()
    log.info("AutoResearch loop started (assesses results every 30s)")

    # Start Claude monitor (anomaly detection on running experiments)
    claude_monitor = None
    if claude is not None and config.claude_monitor_enabled:
        claude_monitor = ClaudeMonitor(config, registry, claude)
        threading.Thread(
            target=claude_monitor.run, daemon=True, name="claude-monitor",
        ).start()
        log.info("Claude monitor started")

    def _shutdown(signum, frame):
        log.info("Worker shutting down...")
        scheduler.stop()
        research.stop()
        autoresearch.stop()
        if claude_monitor is not None:
            claude_monitor.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("Worker running — dashboard is a separate process")
    scheduler.run()  # blocking


def cmd_daemon(args, config: AutoResearchConfig):
    """Legacy all-in-one: worker + dashboard in one process.

    Prefer `worker` + `dashboard` as separate processes so you can iterate
    on the dashboard without restarting GPU job scheduling.
    """
    registry, knowledge, cluster, ideas, claude = _build_core(config)
    scheduler = Scheduler(config, registry, cluster, ideas)
    research = ResearchAgent(config, registry, ideas, knowledge, claude=claude)
    autoresearch = AutoResearchLoop(
        config, registry, ideas, research, knowledge, claude=claude,
    )

    api_server.configure(registry, scheduler, cluster, ideas, research,
                          claude=claude, recipes_dir=config.abs_recipes_dir)
    api_server.run_server_threaded(config.http_host, config.http_port)
    log.info("Dashboard: http://localhost:%d", config.http_port)

    if config.agent_enabled:
        threading.Thread(target=research.run, daemon=True, name="research-agent").start()
        log.info("Research agent started (poll every %dm)", config.poll_interval_m)

    threading.Thread(target=autoresearch.run, daemon=True, name="autoresearch-loop").start()
    log.info("AutoResearch loop started (assesses results every 30s)")

    def _shutdown(signum, frame):
        log.info("Shutting down...")
        scheduler.stop()
        research.stop()
        autoresearch.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    scheduler.run()


def cmd_dashboard(args, config: AutoResearchConfig):
    """Run only the dashboard — safe to restart anytime.

    The dashboard reads the same registry + knowledge base the worker is
    writing to (SQLite WAL + LanceDB both support concurrent cross-process
    access). Its Scheduler/ResearchAgent instances are local to this
    process and NOT the same ones the worker runs — they exist only to
    satisfy the API server's typed handles. Control endpoints that try to
    mutate scheduler state (start/stop/kill) from the dashboard process
    will not affect the worker; use the CLI or a command-queue endpoint
    for that. Read-only status/metrics/knowledge endpoints work fine.
    """
    registry, knowledge, cluster, ideas, claude = _build_core(config)
    scheduler = Scheduler(config, registry, cluster, ideas)  # read-only handle
    research = ResearchAgent(config, registry, ideas, knowledge, claude=claude)

    api_server.configure(registry, scheduler, cluster, ideas, research,
                          claude=claude, recipes_dir=config.abs_recipes_dir)

    # Poll discover_all on a timer so the dashboard's in-memory cluster
    # state stays fresh. The worker has its own scheduler driving probes,
    # but this process has its own ClusterManager instance that would
    # otherwise remain frozen at OFFLINE forever.
    def _probe_loop():
        while True:
            try:
                cluster.discover_all()
            except Exception as e:
                log.warning("dashboard cluster probe failed: %s", e)
            time.sleep(config.health_check_interval_s)
    threading.Thread(target=_probe_loop, daemon=True,
                     name="dashboard-cluster-probe").start()

    log.info("Dashboard only: http://localhost:%d (worker runs separately)",
             config.http_port)
    api_server.run_server(config.http_host, config.http_port)


def cmd_seed_demo(args, config: AutoResearchConfig):
    """Seed the database with demo data for testing without GPUs."""
    registry = Registry(config.abs_db_path)
    ideas_tracker = IdeaTracker(config, registry)

    from .db.models import IdeaSource, ExperimentCategory, ExperimentStatus

    # Create demo ideas
    idea1 = ideas_tracker.create_idea(
        title="SwiGLU activation for MLP layers",
        hypothesis="Replacing GELU with SwiGLU will reduce BPB by ~0.005 based on recent transformer efficiency papers",
        source=IdeaSource.PAPER,
        source_ref="https://arxiv.org/abs/2002.05202",
        priority=3,
        tags=["architecture", "mlp", "activation"],
        notes="SwiGLU showed strong results in PaLM and LLaMA architectures",
    )
    ideas_tracker.evaluate_idea(idea1.id,
        "Does not violate rules. Scales well — activation change is O(1) added params. "
        "Novel approach for this competition. Related: GLU variants paper (Shazeer 2020)."
    )
    ideas_tracker.approve_idea(idea1.id, "Approved — low risk, high potential")

    # Experiments under idea 1
    exp1 = ideas_tracker.create_experiment(
        idea_id=idea1.id,
        name="SwiGLU baseline comparison",
        env_overrides={"ACTIVATION": "swiglu", "MLP_MULT": "2.67"},
        category=ExperimentCategory.ARCHITECTURE,
        hypothesis="SwiGLU with adjusted MLP ratio should match or beat GELU baseline",
        stages=["screen", "gate"],
        priority=3,
    )
    registry.update_experiment_status(exp1.id, ExperimentStatus.DONE)
    registry.update_screen_results(exp1.id, train_bpb=1.325, ema_bpb=1.318,
                                    wallclock_s=175.2, gpu_count=2)
    registry.update_gate_results(exp1.id, int6_bpb=1.695, quant_gap=0.098,
                                  artifact_mb=7.2, gate_passed=True)

    exp2 = ideas_tracker.create_experiment(
        idea_id=idea1.id,
        name="SwiGLU with higher MLP ratio",
        env_overrides={"ACTIVATION": "swiglu", "MLP_MULT": "3.0"},
        category=ExperimentCategory.ARCHITECTURE,
        hypothesis="Larger MLP with SwiGLU may further improve",
        parent_exp=exp1.id,
        stages=["screen", "gate"],
    )
    registry.update_experiment_status(exp2.id, ExperimentStatus.SCREENING)
    registry.update_screen_results(exp2.id, train_bpb=1.331, ema_bpb=1.322,
                                    wallclock_s=140.0, gpu_count=2)

    # Idea 2: Learning rate exploration
    idea2 = ideas_tracker.create_idea(
        title="Muon LR sweep around 0.03",
        hypothesis="LR 0.03 was optimal in Track 4; fine-grained sweep around it may find a better value",
        source=IdeaSource.RECORD_MINING,
        source_ref="records/track_10min_16mb/baseline",
        priority=2,
        tags=["hyperparameter", "optimizer", "learning-rate"],
    )
    ideas_tracker.approve_idea(idea2.id)

    for i, lr in enumerate(["0.025", "0.028", "0.030", "0.032", "0.035"]):
        exp = ideas_tracker.create_experiment(
            idea_id=idea2.id,
            name=f"Muon LR={lr}",
            env_overrides={"MUON_LR": lr},
            category=ExperimentCategory.HYPERPARAMETER,
            stages=["screen", "gate"],
        )
        if i < 3:
            registry.update_experiment_status(exp.id, ExperimentStatus.DONE)
            bpb = 1.338 - (0.003 * (2 - abs(i - 2)))
            registry.update_screen_results(exp.id, train_bpb=bpb, ema_bpb=bpb - 0.005,
                                            wallclock_s=178.0, gpu_count=2)
            registry.update_gate_results(exp.id, int6_bpb=bpb + 0.37,
                                          quant_gap=0.10 + i * 0.01,
                                          artifact_mb=7.1, gate_passed=True)
        elif i == 3:
            registry.update_experiment_status(exp.id, ExperimentStatus.GATING)
        else:
            registry.update_experiment_status(exp.id, ExperimentStatus.QUEUED)

    # Idea 3: From a PR
    idea3 = ideas_tracker.create_idea(
        title="PR#1105: Mixed int5/int6 quantization",
        hypothesis="Using int5 for attention and int6 for MLP may reduce artifact size while maintaining BPB",
        source=IdeaSource.GITHUB_PR,
        source_ref="https://github.com/openai/parameter-golf/pull/1105",
        priority=2,
        tags=["quantization", "compression"],
        notes="PR reports val_bpb=1.08 with 12.5MB artifact",
    )

    # Idea 4: Parked idea
    idea4 = ideas_tracker.create_idea(
        title="Test-time training on validation",
        hypothesis="TTT on validation tokens may improve BPB by 0.01-0.02 based on TTT papers",
        source=IdeaSource.PAPER,
        priority=1,
        tags=["ttt", "evaluation"],
    )
    ideas_tracker.park_idea(idea4.id, "Parking until core architecture is finalized")

    # Add some SOTA entries
    registry.add_sota_entry("records", "track_10min_16mb/baseline", 1.138, 7.3, "baseline")
    registry.add_sota_entry("github_pr", "PR#1089", 1.085, 11.2, "int6+zstd")
    registry.add_sota_entry("github_pr", "PR#1105", 1.080, 12.5, "mixed int5/int6")

    # Add local test node
    cluster = ClusterManager(config, registry)
    cluster.add_local_node(gpu_count=8)

    log.info("Demo data seeded: 4 ideas, %d experiments, 1 local node",
             len(registry.list_experiments()))
    print(f"\nDashboard: python -m autoresearch.main dashboard")
    print(f"  then visit http://localhost:{config.http_port}")


def cmd_cluster_status(args, config: AutoResearchConfig):
    """Show cluster status."""
    registry = Registry(config.abs_db_path)
    cluster = ClusterManager(config, registry)
    cluster.discover_all()
    summary = cluster.get_cluster_summary()

    print(f"\n{'='*60}")
    print(f"  Cluster Status: {summary['online_nodes']}/{summary['total_nodes']} nodes online")
    print(f"  GPUs: {summary['free_gpus']}/{summary['total_gpus']} free")
    print(f"  Running jobs: {summary['running_jobs']}")
    print(f"{'='*60}")

    for host, node in summary["nodes"].items():
        status_icon = "●" if node["status"] == "online" else "○"
        print(f"\n  {status_icon} {node['label']} ({host})")
        print(f"    GPUs: {node['total_gpus']} total, {node['free_gpus']} free")
        for gpu in node.get("gpus", []):
            pct = round(gpu["mem_used_mb"] / gpu["mem_total_mb"] * 100) if gpu["mem_total_mb"] else 0
            exp = f" → {gpu['experiment']}" if gpu.get("experiment") else ""
            print(f"    GPU {gpu['index']}: {gpu['name']}  {pct}% mem  {gpu['util_pct']}% util  {gpu['temp_c']}°C{exp}")


def cmd_create_idea(args, config: AutoResearchConfig):
    """Create a research idea from CLI."""
    registry = Registry(config.abs_db_path)
    ideas = IdeaTracker(config, registry)
    from .db.models import IdeaSource
    idea = ideas.create_idea(
        title=args.title,
        hypothesis=args.hypothesis,
        source=IdeaSource(args.source),
        priority=args.priority,
        tags=args.tags.split(",") if args.tags else [],
        notes=args.notes or "",
    )
    print(f"Created idea: {idea.id}")
    print(f"  Title: {idea.title}")
    print(f"  Status: {idea.status.value}")


def cmd_migrate_recipes(args, config: AutoResearchConfig):
    """Seed the naive baseline recipe + migrate existing experiments to recipes.

    Idempotent: rerunning only fills in missing bindings. Safe to run against
    a registry the worker is actively writing to (SQLite WAL handles it).
    """
    import subprocess
    from .db.recipes import RecipeStore, seed_naive_baseline

    registry = Registry(config.abs_db_path)
    store = RecipeStore(registry, config.abs_recipes_dir)

    # Pin the baseline to the current git commit if available
    try:
        base_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=config.workspace_dir, text=True,
        ).strip()
    except Exception:
        base_commit = ""

    baseline = seed_naive_baseline(store, base_commit=base_commit)
    print(f"Naive baseline recipe: {baseline.id}")
    print(f"  features={baseline.features}")
    print(f"  yaml={baseline.yaml_path}")

    # Ensure the current_best pointer exists so downstream code has a target
    if store.current_best() is None:
        store.set_pointer(
            RecipeStore.CURRENT_BEST_POINTER, baseline.id,
            notes="seeded (no experiments yet)",
        )
        print(f"  pointer: current_best_baseline -> {baseline.id}")

    # Back-fill recipe_id on existing experiments. Each unique env_overrides
    # set becomes one recipe (dedup handles collisions), stacked on baseline.
    experiments = registry.list_experiments()
    bound = 0
    for exp in experiments:
        if getattr(exp, "recipe_id", None):
            continue
        if not exp.env_overrides:
            continue
        feature_tag = [f"{k.lower()}={v}" for k, v in sorted(exp.env_overrides.items())]
        rec = store.create(
            name=exp.name or exp.id,
            features=feature_tag,
            env_overrides=exp.env_overrides,
            description=f"Auto-migrated from experiment {exp.id}",
            parent_recipe=baseline.id,
            source_experiment=exp.id,
            base_commit=getattr(exp, "commit_sha", "") or base_commit,
            dedup=True,
        )
        registry.set_experiment_recipe(exp.id, rec.id)
        val = exp.promote_ema_bpb or exp.screen_ema_bpb or exp.promote_train_bpb
        int6 = exp.promote_int6_bpb or exp.gate_int6_bpb
        art = exp.promote_artifact_mb or exp.gate_artifact_mb
        if val is not None or int6 is not None:
            store.update_best_metrics(
                rec.id, exp.id, val_bpb=val, int6_bpb=int6, artifact_mb=art,
            )
        bound += 1

    print(f"Bound {bound} experiments to recipes")
    print(f"Total recipes: {len(store.list(limit=10000))}")


def cmd_sync_baselines(args, config: AutoResearchConfig):
    """Decode records/ SOTAs into baseline recipes and bump current_best."""
    from .db.recipes import RecipeStore
    from .research import sota_fork

    registry = Registry(config.abs_db_path)
    store = RecipeStore(registry, config.abs_recipes_dir)
    repo_root = Path(config.workspace_dir).parent
    records_dir = repo_root / "records" / "track_10min_16mb"
    if not records_dir.exists():
        print(f"No records directory: {records_dir}")
        return
    installed = sota_fork.sync_from_records(
        records_dir, recipes_store=store, repo_root=repo_root,
    )
    print(f"Installed {len(installed)} baseline recipe(s)")
    for r in installed:
        print(f"  {r.id}  best_val_bpb={r.best_val_bpb}  script={r.script_path}")
    current = store.current_best()
    if current:
        print(f"current_best_baseline -> {current.id} (val_bpb={current.best_val_bpb})")
    else:
        print("No current_best_baseline set")


def cmd_backfill_ideas(args, config: AutoResearchConfig):
    """Re-dispatch Claude assess_pr for orphan PROPOSED PR ideas.

    Finds ideas with source=github_pr in PROPOSED state with zero
    experiments and fires a new Claude assess_pr task for each. The
    callback will compose a recipe + queue an experiment on the existing
    idea row.
    """
    from .research.agent import ResearchAgent

    registry, knowledge, _cluster, ideas, claude = _build_core(config)
    if claude is None:
        print("Claude runner disabled — cannot dispatch assess_pr tasks.")
        return
    agent = ResearchAgent(config, registry, ideas=ideas, knowledge=knowledge,
                           claude=claude)

    count = agent.backfill_orphan_pr_ideas(limit=args.limit)
    print(f"Dispatched {count} assess_pr task(s) for orphan PR ideas")
    threads = getattr(agent, "_backfill_threads", [])
    if threads:
        print(f"Waiting for {len(threads)} Claude task thread(s)...")
        for i, t in enumerate(threads, 1):
            t.join()
            if i % 5 == 0:
                print(f"  {i}/{len(threads)} done")
        print("All assess_pr tasks complete.")


def cmd_import_v2(args, config: AutoResearchConfig):
    """Import experiments from autoresearch registry."""
    import sqlite3
    v2_path = Path(args.db_path)
    if not v2_path.exists():
        print(f"Error: {v2_path} not found")
        sys.exit(1)

    registry = Registry(config.abs_db_path)
    ideas = IdeaTracker(config, registry)
    from .db.models import IdeaSource, ExperimentCategory, ExperimentStatus

    # Create a catch-all idea for imported experiments
    idea = ideas.create_idea(
        title="Imported from autoresearch v2",
        hypothesis="Historical experiments from prior autoresearch runs",
        source=IdeaSource.HUMAN,
        priority=1,
        tags=["imported", "historical"],
    )
    ideas.approve_idea(idea.id)

    # Read v2 experiments
    conn = sqlite3.connect(str(v2_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM experiments ORDER BY created_at").fetchall()

    count = 0
    for row in rows:
        try:
            exp = ideas.create_experiment(
                idea_id=idea.id,
                name=row["name"],
                env_overrides=json.loads(row["env_overrides"]) if row["env_overrides"] else {},
                category=ExperimentCategory(row["category"]) if row["category"] else ExperimentCategory.OTHER,
                hypothesis=row.get("hypothesis", ""),
                stages=json.loads(row["stages"]) if row.get("stages") else ["screen", "gate"],
                notes=row.get("notes", ""),
            )
            # Copy metrics
            status = row["status"]
            if row["screen_train_bpb"]:
                registry.update_screen_results(
                    exp.id,
                    train_bpb=row["screen_train_bpb"],
                    ema_bpb=row["screen_ema_bpb"],
                    wallclock_s=row["screen_wallclock_s"],
                    gpu_count=row["screen_gpu_count"],
                )
            if row["gate_int6_bpb"]:
                registry.update_gate_results(
                    exp.id,
                    int6_bpb=row["gate_int6_bpb"],
                    quant_gap=row["gate_quant_gap"] or 0,
                    artifact_mb=row["gate_artifact_mb"] or 0,
                    gate_passed=bool(row["gate_passed"]),
                )
            # Set status
            try:
                registry.update_experiment_status(exp.id, ExperimentStatus(status))
            except (ValueError, KeyError):
                registry.update_experiment_status(exp.id, ExperimentStatus.DONE)
            count += 1
        except Exception as e:
            log.warning("Failed to import %s: %s", row["id"], e)

    conn.close()
    print(f"Imported {count} experiments under idea '{idea.id}'")


def main():
    parser = argparse.ArgumentParser(description="Parameter Golf AutoResearch")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("worker", help="Run scheduler + research agent (no dashboard)")
    sub.add_parser("dashboard", help="Run dashboard only (separate from worker)")
    sub.add_parser("daemon", help="Legacy: worker + dashboard in one process")
    sub.add_parser("seed-demo", help="Seed demo data")
    sub.add_parser("cluster-status", help="Show cluster status")
    sub.add_parser("migrate-recipes",
                   help="Seed naive baseline recipe and bind existing experiments")
    sub.add_parser("sync-baselines",
                   help="Fork records/track_10min_16mb SOTAs into baseline recipes")
    p_backfill = sub.add_parser(
        "backfill-ideas",
        help="Re-dispatch Claude assess_pr for orphan PROPOSED PR ideas",
    )
    p_backfill.add_argument("--limit", type=int, default=100)

    p_idea = sub.add_parser("create-idea", help="Create a research idea")
    p_idea.add_argument("--title", required=True)
    p_idea.add_argument("--hypothesis", required=True)
    p_idea.add_argument("--source", default="human")
    p_idea.add_argument("--priority", type=int, default=2)
    p_idea.add_argument("--tags", default="")
    p_idea.add_argument("--notes", default="")

    p_import = sub.add_parser("import-v2", help="Import from autoresearch DB")
    p_import.add_argument("db_path", help="Path to v2 registry.db")

    args = parser.parse_args()
    setup_logging(args.log_level)

    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    config = AutoResearchConfig.from_yaml(config_path)

    commands = {
        "worker": cmd_worker,
        "daemon": cmd_daemon,
        "dashboard": cmd_dashboard,
        "seed-demo": cmd_seed_demo,
        "cluster-status": cmd_cluster_status,
        "migrate-recipes": cmd_migrate_recipes,
        "sync-baselines": cmd_sync_baselines,
        "backfill-ideas": cmd_backfill_ideas,
        "create-idea": cmd_create_idea,
        "import-v2": cmd_import_v2,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
