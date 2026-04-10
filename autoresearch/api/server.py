"""HTTP API server for the autoresearch dashboard and controls.

Serves both the dashboard HTML and the REST/WebSocket API.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, parse_qs

log = logging.getLogger(__name__)

# Will be set by main.py at startup
_registry = None
_scheduler = None
_cluster = None
_ideas = None
_research = None
_claude = None
_recipes = None
_dashboard_html = None


def configure(registry, scheduler, cluster, ideas, research,
               *, claude=None, recipes_dir=None):
    global _registry, _scheduler, _cluster, _ideas, _research
    global _claude, _recipes, _dashboard_html
    _registry = registry
    _scheduler = scheduler
    _cluster = cluster
    _ideas = ideas
    _research = research
    _claude = claude
    if recipes_dir is not None:
        from ..db.recipes import RecipeStore
        _recipes = RecipeStore(registry, recipes_dir)
    # Load dashboard HTML
    html_path = Path(__file__).parent.parent / "dashboard" / "dashboard.html"
    if html_path.exists():
        _dashboard_html = html_path.read_text()


class APIHandler(BaseHTTPRequestHandler):
    """HTTP request handler with REST API + dashboard serving."""

    # Disable reverse DNS lookup — it adds seconds per request
    def address_string(self):
        return self.client_address[0]

    def log_message(self, fmt, *args):
        log.debug(fmt, *args)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        query = parse_qs(parsed.query)

        routes = [
            (r"^$", self._serve_dashboard),
            (r"^/api/status$", self._api_status),
            (r"^/api/cluster$", self._api_cluster),
            (r"^/api/ideas$", self._api_ideas),
            (r"^/api/ideas/([^/]+)$", self._api_idea_detail),
            (r"^/api/experiments$", self._api_experiments),
            (r"^/api/experiments/([^/]+)$", self._api_experiment_detail),
            (r"^/api/experiments/([^/]+)/logs$", self._api_experiment_logs),
            (r"^/api/events$", self._api_events),
            (r"^/api/research/sota$", self._api_sota),
            (r"^/api/knowledge$", self._api_knowledge),
            (r"^/api/knowledge/search$", self._api_knowledge_search),
            (r"^/api/knowledge/recent$", self._api_knowledge_recent),
            (r"^/api/commands$", self._api_commands),
            (r"^/api/commands/(\d+)$", self._api_command_status),
            (r"^/api/leaderboard$", self._api_leaderboard),
            (r"^/api/recipes$", self._api_recipes),
            (r"^/api/recipes/([^/]+)$", self._api_recipe_detail),
            (r"^/api/claude/tasks$", self._api_claude_tasks),
            (r"^/api/claude/tasks/([^/]+)$", self._api_claude_task_detail),
            (r"^/api/claude/tasks/([^/]+)/stdout$", self._api_claude_task_stdout),
        ]

        for pattern, handler in routes:
            m = re.match(pattern, path)
            if m:
                try:
                    handler(query, *m.groups())
                except Exception as e:
                    log.exception("API error: %s", e)
                    self._json_response({"error": str(e)}, 500)
                return

        self._json_response({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        # Read body
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode() if content_len else "{}"
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            data = {}

        routes = [
            (r"^/api/ideas$", self._api_create_idea),
            (r"^/api/ideas/([^/]+)/approve$", self._api_approve_idea),
            (r"^/api/ideas/([^/]+)/reject$", self._api_reject_idea),
            (r"^/api/ideas/([^/]+)/priority$", self._api_set_idea_priority),
            (r"^/api/ideas/([^/]+)/evaluate$", self._api_evaluate_idea),
            (r"^/api/ideas/([^/]+)/experiments$", self._api_create_experiment),
            (r"^/api/experiments/([^/]+)/stop$", self._api_stop_experiment),
            (r"^/api/experiments/([^/]+)/priority$", self._api_set_exp_priority),
            (r"^/api/experiments/([^/]+)/promote$", self._api_promote_experiment),
            (r"^/api/stop-all$", self._api_stop_all),
            (r"^/api/research/poll$", self._api_research_poll),
            (r"^/api/research/web-search$", self._api_web_search),
            (r"^/api/research/deep-research$", self._api_deep_research),
        ]

        for pattern, handler in routes:
            m = re.match(pattern, path)
            if m:
                try:
                    handler(data, *m.groups())
                except Exception as e:
                    log.exception("API error: %s", e)
                    self._json_response({"error": str(e)}, 500)
                return

        self._json_response({"error": "not found"}, 404)

    # ── GET Handlers ───────────────────────────────────────────────────

    def _serve_dashboard(self, query):
        if _dashboard_html:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_dashboard_html.encode())
        else:
            self._json_response({"error": "dashboard.html not found"}, 404)

    def _api_status(self, query):
        status = _scheduler.get_status() if _scheduler else {}
        self._json_response(status)

    def _api_cluster(self, query):
        summary = _cluster.get_cluster_summary() if _cluster else {}
        self._json_response(summary)

    def _api_ideas(self, query):
        summaries = _ideas.get_idea_summary_table() if _ideas else []
        self._json_response({"ideas": summaries})

    def _api_idea_detail(self, query, idea_id: str):
        detail = _ideas.get_idea_with_experiments(idea_id) if _ideas else None
        if not detail:
            self._json_response({"error": "not found"}, 404)
            return
        # Serialize
        idea = detail["idea"]
        exps = detail["experiments"]
        self._json_response({
            "idea": {
                "id": idea.id, "title": idea.title,
                "hypothesis": idea.hypothesis,
                "source": idea.source.value,
                "source_ref": idea.source_ref,
                "status": idea.status.value,
                "priority": idea.priority,
                "tags": idea.tags,
                "notes": idea.notes,
                "evaluation": idea.evaluation,
                "created_at": str(idea.created_at),
            },
            "experiments": [
                {
                    "id": e.id, "name": e.name, "status": e.status.value,
                    "priority": e.priority,
                    "screen_ema_bpb": e.screen_ema_bpb,
                    "gate_int6_bpb": e.gate_int6_bpb,
                    "gate_quant_gap": e.gate_quant_gap,
                    "gate_passed": e.gate_passed,
                    "promote_ema_bpb": e.promote_ema_bpb,
                    "promote_int6_bpb": e.promote_int6_bpb,
                    "node_host": e.node_host,
                    "created_at": str(e.created_at) if e.created_at else None,
                    "started_at": str(e.started_at) if e.started_at else None,
                    "completed_at": str(e.completed_at) if e.completed_at else None,
                    "recipe_id": getattr(e, "recipe_id", None),
                }
                for e in exps
            ],
            "log": detail["log"],
        })

    def _api_experiments(self, query):
        status_filter = query.get("status", [None])[0]
        idea_filter = query.get("idea_id", [None])[0]
        from ..db.models import ExperimentStatus
        st = ExperimentStatus(status_filter) if status_filter else None
        exps = _registry.list_experiments(status=st, idea_id=idea_filter) if _registry else []
        self._json_response({
            "experiments": [
                {
                    "id": e.id, "name": e.name, "idea_id": e.idea_id,
                    "status": e.status.value, "priority": e.priority,
                    "category": e.category.value,
                    "screen_ema_bpb": e.screen_ema_bpb,
                    "screen_train_bpb": e.screen_train_bpb,
                    "gate_int6_bpb": e.gate_int6_bpb,
                    "gate_quant_gap": e.gate_quant_gap,
                    "gate_passed": e.gate_passed,
                    "promote_ema_bpb": e.promote_ema_bpb,
                    "node_host": e.node_host,
                    "gpu_indices": e.gpu_indices,
                    "created_at": str(e.created_at),
                    "started_at": str(e.started_at) if e.started_at else None,
                    "completed_at": str(e.completed_at) if e.completed_at else None,
                    "rejection_reason": e.rejection_reason,
                }
                for e in exps
            ]
        })

    def _api_experiment_detail(self, query, exp_id: str):
        exp = _registry.get_experiment(exp_id) if _registry else None
        if not exp:
            self._json_response({"error": "not found"}, 404)
            return
        self._json_response({
            "id": exp.id, "name": exp.name, "idea_id": exp.idea_id,
            "hypothesis": exp.hypothesis,
            "category": exp.category.value,
            "status": exp.status.value, "priority": exp.priority,
            "env_overrides": exp.env_overrides,
            "stages": exp.stages,
            "parent_id": exp.parent_id,
            "node_host": exp.node_host,
            "gpu_indices": exp.gpu_indices,
            "screen_steps": exp.screen_steps,
            "screen_ms_per_step": exp.screen_ms_per_step,
            "screen_train_bpb": exp.screen_train_bpb,
            "screen_ema_bpb": exp.screen_ema_bpb,
            "screen_wallclock_s": exp.screen_wallclock_s,
            "gate_int6_bpb": exp.gate_int6_bpb,
            "gate_quant_gap": exp.gate_quant_gap,
            "gate_artifact_mb": exp.gate_artifact_mb,
            "gate_passed": exp.gate_passed,
            "promote_train_bpb": exp.promote_train_bpb,
            "promote_ema_bpb": exp.promote_ema_bpb,
            "promote_int6_bpb": exp.promote_int6_bpb,
            "promote_sw_bpb": exp.promote_sw_bpb,
            "promote_artifact_mb": exp.promote_artifact_mb,
            "rejection_reason": exp.rejection_reason,
            "notes": exp.notes,
            "created_at": str(exp.created_at),
            "started_at": str(exp.started_at) if exp.started_at else None,
            "completed_at": str(exp.completed_at) if exp.completed_at else None,
        })

    def _api_experiment_logs(self, query, exp_id: str):
        lines = int(query.get("lines", [100])[0])
        # 1. Try the live cluster (works for jobs still in _running_jobs in
        #    *this* process — mainly the worker).
        log_text = _cluster.get_log_tail(exp_id, lines) if _cluster else ""
        # 2. Fallback to the locally-synced train.log under experiment_logs/.
        #    The scheduler rsyncs remote logs into
        #    <workspace_parent>/experiment_logs/<idea_id>/<exp_id>/train.log
        #    while the job runs and one final time on completion, so this
        #    path is the right source for finished/failed experiments and
        #    for read-only dashboard processes that don't own _running_jobs.
        if not log_text and _registry:
            try:
                from pathlib import Path
                exp = _registry.get_experiment(exp_id)
                if exp:
                    base = Path(_registry.db_path).parent.parent.parent \
                        / "experiment_logs" / exp.idea_id / exp.id / "train.log"
                    if base.exists():
                        with open(base, "rb") as f:
                            # Read last ~lines*512 bytes, then tail
                            try:
                                f.seek(0, 2)
                                size = f.tell()
                                f.seek(max(0, size - lines * 512))
                                tail = f.read().decode("utf-8", errors="replace")
                                log_text = "\n".join(tail.splitlines()[-lines:])
                            except Exception:
                                log_text = base.read_text(errors="replace")
            except Exception as e:
                log.warning("log fallback failed for %s: %s", exp_id, e)
        self._json_response({"exp_id": exp_id, "log": log_text})

    def _api_events(self, query):
        entity_type = query.get("entity_type", [None])[0]
        entity_id = query.get("entity_id", [None])[0]
        limit = int(query.get("limit", [50])[0])
        events = _registry.list_events(entity_type, entity_id, limit) if _registry else []
        for ev in events:
            if isinstance(ev.get("payload"), str):
                try:
                    ev["payload"] = json.loads(ev["payload"])
                except (json.JSONDecodeError, TypeError):
                    pass
        self._json_response({"events": events})

    def _api_sota(self, query):
        best = _registry.get_best_sota() if _registry else None
        self._json_response({"best_sota": best})

    # ── Leaderboard / Recipes ──────────────────────────────────────────

    def _api_leaderboard(self, query):
        """Persistent leaderboard sorted by best val_bpb.

        Joins recipes with their best_experiment_id so the dashboard gets
        a single row per recipe with links to logs, the source PR/branch,
        and any claude-generated report. Rows without metrics are omitted.
        """
        if not _recipes:
            self._json_response({"leaderboard": []})
            return
        limit = int(query.get("limit", [50])[0])
        recipes = _recipes.list(order_by="best_val_bpb", limit=limit)
        rows = []
        current_best_id = None
        cb = _recipes.current_best()
        if cb:
            current_best_id = cb.id
        for r in recipes:
            if r.best_val_bpb is None and r.best_int6_bpb is None:
                continue
            exp = None
            if r.best_experiment_id and _registry:
                exp = _registry.get_experiment(r.best_experiment_id)
            # Link to the experiment_logs/ training log + claude report
            log_link = None
            report_link = None
            pr_or_source = None
            is_reproduction = False
            if exp:
                # Training log file streamed by the scheduler
                log_candidate = Path(_registry.db_path).parent.parent / "experiment_logs" / f"{exp.id}.log"
                if log_candidate.exists():
                    log_link = f"/api/experiments/{exp.id}/logs?lines=400"
                report_candidate = (Path(_registry.db_path).parent.parent.parent
                                    / "experiment_logs" / "claude_reports"
                                    / f"{exp.id}.md")
                if report_candidate.exists():
                    report_link = str(report_candidate)
                pr_or_source = getattr(exp, "source_ref", "") or None
                is_reproduction = bool(getattr(exp, "is_reproduction", False))
            rows.append({
                "recipe_id": r.id,
                "name": r.name,
                "parent_recipe": r.parent_recipe,
                "features": r.features,
                "env_overrides": r.env_overrides,
                "best_val_bpb": r.best_val_bpb,
                "best_int6_bpb": r.best_int6_bpb,
                "best_artifact_mb": r.best_artifact_mb,
                "best_experiment_id": r.best_experiment_id,
                "log_link": log_link,
                "report_link": report_link,
                "source_ref": pr_or_source,
                "is_reproduction": is_reproduction,
                "is_current_best": r.id == current_best_id,
                "base_branch": r.base_branch,
                "base_commit": r.base_commit,
                "created_at": str(r.created_at) if r.created_at else None,
            })
        self._json_response({
            "leaderboard": rows,
            "current_best_recipe_id": current_best_id,
        })

    def _api_recipes(self, query):
        if not _recipes:
            self._json_response({"recipes": []})
            return
        order = query.get("order_by", ["best_val_bpb"])[0]
        limit = int(query.get("limit", [200])[0])
        recipes = _recipes.list(order_by=order, limit=limit)
        self._json_response({
            "recipes": [
                {
                    "id": r.id, "name": r.name,
                    "description": r.description,
                    "parent_recipe": r.parent_recipe,
                    "features": r.features,
                    "env_overrides": r.env_overrides,
                    "feature_set_hash": r.feature_set_hash,
                    "base_commit": r.base_commit,
                    "base_branch": r.base_branch,
                    "best_val_bpb": r.best_val_bpb,
                    "best_int6_bpb": r.best_int6_bpb,
                    "best_artifact_mb": r.best_artifact_mb,
                    "best_experiment_id": r.best_experiment_id,
                    "yaml_path": r.yaml_path,
                    "created_at": str(r.created_at) if r.created_at else None,
                }
                for r in recipes
            ],
        })

    def _api_recipe_detail(self, query, recipe_id: str):
        if not _recipes:
            self._json_response({"error": "recipes disabled"}, 404)
            return
        r = _recipes.get(recipe_id)
        if not r:
            self._json_response({"error": "not found"}, 404)
            return
        children = _recipes.list_children(recipe_id)
        ancestors = _recipes.ancestors(recipe_id)
        self._json_response({
            "recipe": {
                "id": r.id, "name": r.name,
                "description": r.description,
                "parent_recipe": r.parent_recipe,
                "features": r.features,
                "env_overrides": r.env_overrides,
                "feature_set_hash": r.feature_set_hash,
                "base_commit": r.base_commit,
                "base_branch": r.base_branch,
                "best_val_bpb": r.best_val_bpb,
                "best_int6_bpb": r.best_int6_bpb,
                "best_artifact_mb": r.best_artifact_mb,
                "best_experiment_id": r.best_experiment_id,
                "yaml_path": r.yaml_path,
            },
            "ancestors": [{"id": a.id, "name": a.name} for a in ancestors],
            "children": [{"id": c.id, "name": c.name,
                          "best_val_bpb": c.best_val_bpb} for c in children],
        })

    # ── Claude tasks ───────────────────────────────────────────────────

    def _api_claude_tasks(self, query):
        if not _claude:
            self._json_response({"tasks": [], "enabled": False})
            return
        status = query.get("status", [None])[0]
        limit = int(query.get("limit", [100])[0])
        tasks = _claude.list_tasks(status=status, limit=limit)
        self._json_response({"tasks": tasks, "enabled": True})

    def _api_claude_task_detail(self, query, task_id: str):
        if not _claude:
            self._json_response({"error": "claude disabled"}, 404)
            return
        task = _claude.get_task(task_id)
        if not task:
            self._json_response({"error": "not found"}, 404)
            return
        self._json_response({"task": task})

    def _api_claude_task_stdout(self, query, task_id: str):
        if not _claude:
            self._json_response({"error": "claude disabled"}, 404)
            return
        task = _claude.get_task(task_id)
        if not task:
            self._json_response({"error": "not found"}, 404)
            return
        stdout_path = task.get("stdout_path") or ""
        text = ""
        if stdout_path and Path(stdout_path).exists():
            try:
                text = Path(stdout_path).read_text()[-50000:]  # tail 50KB
            except Exception as e:
                text = f"(read error: {e})"
        self._json_response({"task_id": task_id, "stdout": text})

    # ── POST Handlers ──────────────────────────────────────────────────

    def _api_create_idea(self, data):
        from ..db.models import IdeaSource
        idea = _ideas.create_idea(
            title=data["title"],
            hypothesis=data["hypothesis"],
            source=IdeaSource(data.get("source", "human")),
            source_ref=data.get("source_ref", ""),
            priority=data.get("priority", 2),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )
        self._json_response({"id": idea.id, "status": "created"}, 201)

    def _api_approve_idea(self, data, idea_id: str):
        _ideas.approve_idea(idea_id, data.get("notes", ""))
        self._json_response({"status": "approved"})

    def _api_reject_idea(self, data, idea_id: str):
        _ideas.reject_idea(idea_id, data.get("reason", "rejected by user"))
        self._json_response({"status": "rejected"})

    def _api_set_idea_priority(self, data, idea_id: str):
        _ideas.set_priority(idea_id, data["priority"])
        self._json_response({"status": "updated"})

    def _api_evaluate_idea(self, data, idea_id: str):
        _ideas.evaluate_idea(idea_id, data["evaluation"])
        self._json_response({"status": "evaluated"})

    def _api_create_experiment(self, data, idea_id: str):
        from ..db.models import ExperimentCategory
        exp = _ideas.create_experiment(
            idea_id=idea_id,
            name=data["name"],
            env_overrides=data.get("env_overrides", {}),
            category=ExperimentCategory(data.get("category", "other")),
            hypothesis=data.get("hypothesis", ""),
            parent_exp=data.get("parent_id"),
            stages=data.get("stages", ["screen", "gate"]),
            priority=data.get("priority"),
            notes=data.get("notes", ""),
        )
        # Queue it
        from ..db.models import ExperimentStatus
        _registry.update_experiment_status(exp.id, ExperimentStatus.QUEUED)
        self._json_response({"id": exp.id, "status": "queued"}, 201)

    def _api_stop_experiment(self, data, exp_id: str):
        """Stop a running experiment.

        Routes through the command queue so the dashboard process can
        control the worker process without shared memory. Payload:
          {"reason": str, "mode": "graceful"|"force", "grace_period_s": int}
        Default mode is 'graceful' (SIGTERM → wait → SIGKILL).
        """
        reason = data.get("reason", "stopped by user")
        mode = data.get("mode", "graceful")
        if mode not in ("graceful", "force"):
            self._json_response(
                {"error": "mode must be 'graceful' or 'force'"}, 400)
            return
        grace = int(data.get("grace_period_s", 10))
        cmd_id = _registry.enqueue_command(
            "kill_experiment",
            target_id=exp_id,
            payload={"reason": reason, "mode": mode,
                      "grace_period_s": grace},
        )
        self._json_response({
            "status": "queued",
            "command_id": cmd_id,
            "mode": mode,
            "note": "Worker will execute on next scheduler tick",
        })

    def _api_set_exp_priority(self, data, exp_id: str):
        cmd_id = _registry.enqueue_command(
            "prioritize_experiment",
            target_id=exp_id,
            payload={"priority": int(data["priority"])},
        )
        self._json_response({"status": "queued", "command_id": cmd_id})

    def _api_promote_experiment(self, data, exp_id: str):
        cmd_id = _registry.enqueue_command(
            "promote_experiment", target_id=exp_id)
        self._json_response({"status": "queued", "command_id": cmd_id})

    def _api_stop_all(self, data):
        cmd_id = _registry.enqueue_command("stop_all")
        self._json_response({"status": "queued", "command_id": cmd_id})

    def _api_command_status(self, query, cmd_id: str):
        try:
            cmd = _registry.get_command(int(cmd_id))
        except ValueError:
            self._json_response({"error": "invalid command id"}, 400)
            return
        if not cmd:
            self._json_response({"error": "not found"}, 404)
            return
        self._json_response(cmd)

    def _api_commands(self, query):
        status = query.get("status", [None])[0]
        limit = int(query.get("limit", ["50"])[0])
        self._json_response({
            "commands": _registry.list_commands(status=status, limit=limit),
        })

    def _api_research_poll(self, data):
        result = _research.poll_now() if _research else {"error": "agent not running"}
        self._json_response(result)

    def _api_web_search(self, data):
        """POST /api/research/web-search — quick Parallel web search."""
        query = data.get("query", "")
        terms = data.get("search_terms", [query])
        if not query:
            self._json_response({"error": "query required"}, 400)
            return
        results = _research.web_search(query, terms) if _research else []
        self._json_response({
            "query": query,
            "results": [
                {"url": r.url, "title": r.title, "excerpts": r.excerpts,
                 "publish_date": r.publish_date}
                for r in results
            ],
        })

    def _api_deep_research(self, data):
        """POST /api/research/deep-research — Parallel deep research."""
        query = data.get("query", "")
        processor = data.get("processor")
        if not query:
            self._json_response({"error": "query required"}, 400)
            return
        result = _research.deep_research(query, processor) if _research else None
        if not result:
            self._json_response({"error": "research agent not available"}, 503)
            return
        self._json_response({
            "run_id": result.run_id,
            "status": result.status,
            "processor": result.processor,
            "output": result.output,
            "error": result.error,
            "citations": len(result.basis),
        })

    # ── Knowledge Base GET Handlers ───────────────────────────────────

    def _api_knowledge(self, query):
        """GET /api/knowledge — knowledge base summary stats."""
        if not _research:
            self._json_response({"error": "not available"}, 503)
            return
        self._json_response(_research.get_knowledge_summary())

    def _api_knowledge_search(self, query):
        """GET /api/knowledge/search?q=...&type=... — search knowledge base."""
        q = query.get("q", [""])[0]
        source_type = query.get("type", [None])[0]
        if not q:
            self._json_response({"error": "q parameter required"}, 400)
            return
        results = _research.query_knowledge(q, source_type) if _research else []
        self._json_response({"query": q, "results": results})

    def _api_knowledge_recent(self, query):
        """GET /api/knowledge/recent — recent knowledge entries."""
        limit = int(query.get("limit", [20])[0])
        source_type = query.get("type", [None])[0]
        if not _research:
            self._json_response({"error": "not available"}, 503)
            return
        entries = _research.kb.recent(limit=limit, source_type=source_type)
        self._json_response({
            "entries": [
                {"id": e.id, "type": e.source_type, "title": e.title,
                 "content": e.content[:500], "tags": e.tags,
                 "created_at": e.created_at}
                for e in entries
            ],
        })

    # ── Helpers ─────────────────────────────────────────────────────────

    def _json_response(self, data: dict, status: int = 200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run_server(host: str = "0.0.0.0", port: int = 8765):
    """Start the HTTP server (blocking)."""
    server = _ThreadingHTTPServer((host, port), APIHandler)
    log.info("API server listening on %s:%d", host, port)
    server.serve_forever()


def run_server_threaded(host: str = "0.0.0.0", port: int = 8765) -> threading.Thread:
    """Start the HTTP server in a background thread."""
    t = threading.Thread(target=run_server, args=(host, port), daemon=True)
    t.start()
    return t
