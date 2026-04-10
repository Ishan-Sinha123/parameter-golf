"""SQLite registry for ideas, experiments, nodes, and events."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import (
    Idea, IdeaStatus, IdeaSource,
    Experiment, ExperimentStatus, ExperimentCategory,
    EventType,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ideas (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    hypothesis      TEXT NOT NULL,
    source          TEXT NOT NULL DEFAULT 'human',
    source_ref      TEXT DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'proposed',
    priority        INTEGER DEFAULT 2,
    parent_idea     TEXT REFERENCES ideas(id),
    tags            TEXT DEFAULT '[]',
    notes           TEXT DEFAULT '',
    evaluation      TEXT DEFAULT '',
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiments (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    idea_id             TEXT NOT NULL REFERENCES ideas(id),
    hypothesis          TEXT DEFAULT '',
    category            TEXT DEFAULT 'other',
    priority            INTEGER DEFAULT 2,
    status              TEXT NOT NULL DEFAULT 'defined',
    rejection_reason    TEXT DEFAULT '',

    env_overrides       TEXT DEFAULT '{}',
    script_path         TEXT DEFAULT 'train_gpt.py',
    parent_id           TEXT REFERENCES experiments(id),
    stages              TEXT DEFAULT '["screen","gate"]',

    node_host           TEXT,
    gpu_indices         TEXT DEFAULT '[]',

    screen_steps        INTEGER,
    screen_ms_per_step  REAL,
    screen_train_bpb    REAL,
    screen_ema_bpb      REAL,
    screen_gpu_count    INTEGER,
    screen_wallclock_s  REAL,

    gate_int6_bpb       REAL,
    gate_quant_gap      REAL,
    gate_artifact_mb    REAL,
    gate_passed         INTEGER,

    promote_train_bpb   REAL,
    promote_ema_bpb     REAL,
    promote_int6_bpb    REAL,
    promote_sw_bpb      REAL,
    promote_artifact_mb REAL,
    promote_steps       INTEGER,

    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    started_at          DATETIME,
    completed_at        DATETIME,
    notes               TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,
    event_type      TEXT NOT NULL,
    entity_type     TEXT NOT NULL DEFAULT 'experiment',
    entity_id       TEXT NOT NULL,
    payload         TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS nodes (
    host            TEXT PRIMARY KEY,
    label           TEXT DEFAULT '',
    status          TEXT DEFAULT 'offline',
    gpu_count       INTEGER DEFAULT 0,
    gpu_info        TEXT DEFAULT '[]',
    last_heartbeat  DATETIME,
    error_message   TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS sota_tracker (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,
    source          TEXT NOT NULL,
    source_ref      TEXT NOT NULL,
    val_bpb         REAL,
    artifact_mb     REAL,
    technique       TEXT,
    notes           TEXT DEFAULT ''
);

-- Recipes: canonical, immutable experiment configuration.
-- See db/models.py docstring for the full rationale.
CREATE TABLE IF NOT EXISTS recipes (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    description         TEXT DEFAULT '',
    parent_recipe       TEXT REFERENCES recipes(id),
    features            TEXT DEFAULT '[]',        -- JSON array, canonical sorted
    feature_set_hash    TEXT NOT NULL,            -- dedup key
    env_overrides       TEXT DEFAULT '{}',        -- JSON object
    script_path         TEXT DEFAULT 'train_gpt.py',
    base_commit         TEXT DEFAULT '',
    base_branch         TEXT DEFAULT '',
    source_experiment   TEXT REFERENCES experiments(id),
    best_val_bpb        REAL,
    best_int6_bpb       REAL,
    best_artifact_mb    REAL,
    best_experiment_id  TEXT REFERENCES experiments(id),
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP,
    yaml_path           TEXT DEFAULT ''
);

-- Named pointers into the recipe DAG (e.g. 'current_best_baseline').
CREATE TABLE IF NOT EXISTS recipe_pointers (
    name        TEXT PRIMARY KEY,
    recipe_id   TEXT NOT NULL REFERENCES recipes(id),
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    notes       TEXT DEFAULT ''
);

-- Command queue: cross-process RPC from dashboard → worker.
-- The dashboard process enqueues rows; the worker's scheduler polls
-- pending rows in its main loop and executes them.
CREATE TABLE IF NOT EXISTS commands (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    picked_at       DATETIME,
    completed_at    DATETIME,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|failed
    command         TEXT NOT NULL,                     -- e.g. 'kill_experiment'
    target_id       TEXT DEFAULT '',                   -- e.g. experiment id
    payload         TEXT DEFAULT '{}',                 -- JSON args
    result          TEXT DEFAULT '',                   -- JSON or error msg
    issued_by       TEXT DEFAULT 'dashboard'
);

-- Traces: span tree for the research/experiment loop. Each row is a span;
-- parent_span_id is null for root spans. A trace is the set of spans
-- sharing one trace_id. The dashboard renders this as a cytoscape graph.
CREATE TABLE IF NOT EXISTS traces (
    span_id         TEXT PRIMARY KEY,
    trace_id        TEXT NOT NULL,
    parent_span_id  TEXT,
    kind            TEXT NOT NULL,
    name            TEXT NOT NULL,
    entity_type     TEXT DEFAULT '',
    entity_id       TEXT DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'running',
    started_at      REAL NOT NULL,
    ended_at        REAL,
    duration_ms     INTEGER,
    attrs           TEXT DEFAULT '{}',
    error           TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_experiments_idea ON experiments(idea_id);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_events_entity ON events(entity_type, entity_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_ideas_status ON ideas(status);
CREATE INDEX IF NOT EXISTS idx_commands_status ON commands(status, created_at);
CREATE INDEX IF NOT EXISTS idx_recipes_hash ON recipes(feature_set_hash);
CREATE INDEX IF NOT EXISTS idx_recipes_parent ON recipes(parent_recipe);
CREATE INDEX IF NOT EXISTS idx_recipes_bpb ON recipes(best_val_bpb);
CREATE INDEX IF NOT EXISTS idx_traces_trace ON traces(trace_id);
CREATE INDEX IF NOT EXISTS idx_traces_parent ON traces(parent_span_id);
CREATE INDEX IF NOT EXISTS idx_traces_started ON traces(started_at);
CREATE INDEX IF NOT EXISTS idx_traces_kind ON traces(kind);
CREATE INDEX IF NOT EXISTS idx_traces_entity ON traces(entity_type, entity_id);
"""

# Additive columns on existing tables. SQLite doesn't support
# "ADD COLUMN IF NOT EXISTS" directly, so we introspect table_info and
# ALTER only what's missing. Each entry: (table, column, definition).
_ADDITIVE_COLUMNS = [
    ("experiments", "recipe_id",       "TEXT"),
    ("experiments", "is_reproduction", "INTEGER DEFAULT 0"),
    ("experiments", "source_ref",      "TEXT DEFAULT ''"),
    ("experiments", "commit_sha",      "TEXT DEFAULT ''"),
]


class Registry:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            self._apply_additive_migrations(conn)

    def _apply_additive_migrations(self, conn):
        """Add columns to existing tables if they're missing.

        Idempotent: safe to run on fresh or already-migrated DBs.
        """
        for table, column, definition in _ADDITIVE_COLUMNS:
            cols = {row[1] for row in conn.execute(
                f"PRAGMA table_info({table})").fetchall()}
            if column not in cols:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Ideas ──────────────────────────────────────────────────────────

    def insert_idea(self, idea: Idea):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO ideas (id, title, hypothesis, source, source_ref,
                   status, priority, parent_idea, tags, notes, evaluation)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (idea.id, idea.title, idea.hypothesis, idea.source.value,
                 idea.source_ref, idea.status.value, idea.priority,
                 idea.parent_idea, json.dumps(idea.tags), idea.notes,
                 idea.evaluation),
            )
        self.emit_event(EventType.IDEA_CREATED, "idea", idea.id, {"title": idea.title})

    def update_idea_status(self, idea_id: str, status: IdeaStatus, notes: str = ""):
        with self._conn() as conn:
            conn.execute(
                "UPDATE ideas SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (status.value, idea_id),
            )
            if notes:
                conn.execute(
                    "UPDATE ideas SET notes = notes || '\n' || ? WHERE id=?",
                    (notes, idea_id),
                )
        self.emit_event(EventType.IDEA_STATUS_CHANGE, "idea", idea_id,
                        {"status": status.value, "notes": notes})

    def update_idea_evaluation(self, idea_id: str, evaluation: str):
        with self._conn() as conn:
            conn.execute(
                "UPDATE ideas SET evaluation=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (evaluation, idea_id),
            )
        self.emit_event(EventType.IDEA_EVALUATION, "idea", idea_id,
                        {"evaluation": evaluation[:200]})

    def get_idea(self, idea_id: str) -> Optional[Idea]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM ideas WHERE id=?", (idea_id,)).fetchone()
            if not row:
                return None
            idea = self._row_to_idea(row)
            # Attach experiment IDs
            exps = conn.execute(
                "SELECT id FROM experiments WHERE idea_id=? ORDER BY created_at",
                (idea_id,),
            ).fetchall()
            idea.experiment_ids = [e["id"] for e in exps]
            return idea

    def list_ideas(self, status: Optional[IdeaStatus] = None) -> list[Idea]:
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM ideas WHERE status=? ORDER BY priority DESC, created_at",
                    (status.value,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM ideas ORDER BY priority DESC, created_at",
                ).fetchall()
            ideas = [self._row_to_idea(r) for r in rows]
            # Attach experiment IDs
            for idea in ideas:
                exps = conn.execute(
                    "SELECT id FROM experiments WHERE idea_id=? ORDER BY created_at",
                    (idea.id,),
                ).fetchall()
                idea.experiment_ids = [e["id"] for e in exps]
            return ideas

    def _row_to_idea(self, row) -> Idea:
        return Idea(
            id=row["id"], title=row["title"], hypothesis=row["hypothesis"],
            source=IdeaSource(row["source"]), source_ref=row["source_ref"],
            status=IdeaStatus(row["status"]), priority=row["priority"],
            parent_idea=row["parent_idea"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            notes=row["notes"] or "", evaluation=row["evaluation"] or "",
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    # ── Experiments ────────────────────────────────────────────────────

    def insert_experiment(self, exp: Experiment):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO experiments (id, name, idea_id, hypothesis, category,
                   priority, status, env_overrides, script_path, parent_id, stages, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (exp.id, exp.name, exp.idea_id, exp.hypothesis, exp.category.value,
                 exp.priority, exp.status.value, json.dumps(exp.env_overrides),
                 exp.script_path, exp.parent_id, json.dumps(exp.stages), exp.notes),
            )
        self.emit_event(EventType.EXP_CREATED, "experiment", exp.id,
                        {"name": exp.name, "idea_id": exp.idea_id})

    def update_experiment_status(self, exp_id: str, status: ExperimentStatus,
                                  reason: str = ""):
        with self._conn() as conn:
            updates = ["status=?"]
            params: list = [status.value]
            if status == ExperimentStatus.SCREENING:
                updates.append("started_at=CURRENT_TIMESTAMP")
            if status in (ExperimentStatus.DONE, ExperimentStatus.REJECTED,
                          ExperimentStatus.FAILED, ExperimentStatus.STOPPED):
                updates.append("completed_at=CURRENT_TIMESTAMP")
            if reason:
                updates.append("rejection_reason=?")
                params.append(reason)
            params.append(exp_id)
            conn.execute(
                f"UPDATE experiments SET {', '.join(updates)} WHERE id=?",
                params,
            )

    def assign_experiment_node(self, exp_id: str, host: str, gpu_indices: list[int]):
        with self._conn() as conn:
            conn.execute(
                "UPDATE experiments SET node_host=?, gpu_indices=? WHERE id=?",
                (host, json.dumps(gpu_indices), exp_id),
            )

    def update_screen_results(self, exp_id: str, **kwargs):
        sets = ", ".join(f"screen_{k}=?" for k in kwargs)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE experiments SET {sets} WHERE id=?",
                (*kwargs.values(), exp_id),
            )

    def update_gate_results(self, exp_id: str, int6_bpb: float, quant_gap: float,
                             artifact_mb: float, gate_passed: bool):
        with self._conn() as conn:
            conn.execute(
                """UPDATE experiments SET gate_int6_bpb=?, gate_quant_gap=?,
                   gate_artifact_mb=?, gate_passed=? WHERE id=?""",
                (int6_bpb, quant_gap, artifact_mb, int(gate_passed), exp_id),
            )

    def update_promote_results(self, exp_id: str, **kwargs):
        sets = ", ".join(f"promote_{k}=?" for k in kwargs)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE experiments SET {sets} WHERE id=?",
                (*kwargs.values(), exp_id),
            )

    def get_experiment(self, exp_id: str) -> Optional[Experiment]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
            return self._row_to_experiment(row) if row else None

    def list_experiments(self, status: Optional[ExperimentStatus] = None,
                          idea_id: Optional[str] = None) -> list[Experiment]:
        with self._conn() as conn:
            clauses, params = [], []
            if status:
                clauses.append("status=?")
                params.append(status.value)
            if idea_id:
                clauses.append("idea_id=?")
                params.append(idea_id)
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = conn.execute(
                f"SELECT * FROM experiments {where} ORDER BY priority DESC, created_at",
                params,
            ).fetchall()
            return [self._row_to_experiment(r) for r in rows]

    def _row_to_experiment(self, row) -> Experiment:
        return Experiment(
            id=row["id"], name=row["name"], idea_id=row["idea_id"],
            hypothesis=row["hypothesis"] or "",
            category=ExperimentCategory(row["category"]) if row["category"] else ExperimentCategory.OTHER,
            priority=row["priority"], status=ExperimentStatus(row["status"]),
            rejection_reason=row["rejection_reason"] or "",
            env_overrides=json.loads(row["env_overrides"]) if row["env_overrides"] else {},
            script_path=row["script_path"] or "train_gpt.py",
            parent_id=row["parent_id"],
            stages=json.loads(row["stages"]) if row["stages"] else ["screen", "gate"],
            node_host=row["node_host"],
            gpu_indices=json.loads(row["gpu_indices"]) if row["gpu_indices"] else [],
            screen_steps=row["screen_steps"],
            screen_ms_per_step=row["screen_ms_per_step"],
            screen_train_bpb=row["screen_train_bpb"],
            screen_ema_bpb=row["screen_ema_bpb"],
            screen_gpu_count=row["screen_gpu_count"],
            screen_wallclock_s=row["screen_wallclock_s"],
            gate_int6_bpb=row["gate_int6_bpb"],
            gate_quant_gap=row["gate_quant_gap"],
            gate_artifact_mb=row["gate_artifact_mb"],
            gate_passed=bool(row["gate_passed"]) if row["gate_passed"] is not None else None,
            promote_train_bpb=row["promote_train_bpb"],
            promote_ema_bpb=row["promote_ema_bpb"],
            promote_int6_bpb=row["promote_int6_bpb"],
            promote_sw_bpb=row["promote_sw_bpb"],
            promote_artifact_mb=row["promote_artifact_mb"],
            promote_steps=row["promote_steps"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            notes=row["notes"] or "",
            # Additive columns — use .keys() check because older DBs that
            # somehow missed the migration shouldn't crash the reader.
            recipe_id=(row["recipe_id"] if "recipe_id" in row.keys() else None),
            is_reproduction=bool(row["is_reproduction"])
                if "is_reproduction" in row.keys() and row["is_reproduction"] is not None
                else False,
            source_ref=(row["source_ref"] if "source_ref" in row.keys() else "") or "",
            commit_sha=(row["commit_sha"] if "commit_sha" in row.keys() else "") or "",
        )

    def set_experiment_recipe(self, exp_id: str, recipe_id: str):
        """Link an experiment to its canonical recipe."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE experiments SET recipe_id=? WHERE id=?",
                (recipe_id, exp_id),
            )

    def set_experiment_source(self, exp_id: str, source_ref: str,
                                is_reproduction: bool = False):
        """Record where an experiment came from (PR, branch, records/)."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE experiments
                   SET source_ref=?, is_reproduction=? WHERE id=?""",
                (source_ref, int(is_reproduction), exp_id),
            )

    # ── Nodes ──────────────────────────────────────────────────────────

    def upsert_node(self, host: str, label: str, status: str, gpu_count: int,
                     gpu_info: str = "[]", error_message: str = ""):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO nodes (host, label, status, gpu_count, gpu_info,
                   last_heartbeat, error_message)
                   VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                   ON CONFLICT(host) DO UPDATE SET
                   label=?, status=?, gpu_count=?, gpu_info=?,
                   last_heartbeat=CURRENT_TIMESTAMP, error_message=?""",
                (host, label, status, gpu_count, gpu_info, error_message,
                 label, status, gpu_count, gpu_info, error_message),
            )

    def list_nodes(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM nodes ORDER BY host").fetchall()
            return [dict(r) for r in rows]

    # ── Events ─────────────────────────────────────────────────────────

    def emit_event(self, event_type: EventType, entity_type: str,
                    entity_id: str, payload: dict | None = None):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO events (event_type, entity_type, entity_id, payload)
                   VALUES (?, ?, ?, ?)""",
                (event_type.value, entity_type, entity_id,
                 json.dumps(payload or {})),
            )

    def list_events(self, entity_type: Optional[str] = None,
                     entity_id: Optional[str] = None,
                     limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            clauses, params = [], []
            if entity_type:
                clauses.append("entity_type=?")
                params.append(entity_type)
            if entity_id:
                clauses.append("entity_id=?")
                params.append(entity_id)
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            params.append(limit)
            rows = conn.execute(
                f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    # ── SOTA Tracker ───────────────────────────────────────────────────

    def add_sota_entry(self, source: str, source_ref: str, val_bpb: float,
                        artifact_mb: float = 0, technique: str = "", notes: str = ""):
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO sota_tracker (source, source_ref, val_bpb,
                   artifact_mb, technique, notes) VALUES (?, ?, ?, ?, ?, ?)""",
                (source, source_ref, val_bpb, artifact_mb, technique, notes),
            )

    def get_best_sota(self) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sota_tracker ORDER BY val_bpb ASC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    # ── Command Queue (dashboard → worker RPC) ────────────────────────

    def enqueue_command(self, command: str, target_id: str = "",
                         payload: Optional[dict] = None,
                         issued_by: str = "dashboard") -> int:
        """Enqueue a command for the worker to execute. Returns command id."""
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO commands (command, target_id, payload, issued_by)
                   VALUES (?, ?, ?, ?)""",
                (command, target_id, json.dumps(payload or {}), issued_by),
            )
            return cur.lastrowid

    def claim_pending_commands(self, limit: int = 20) -> list[dict]:
        """Atomically mark pending commands as running and return them.

        Called by the scheduler's poll loop. Using a single UPDATE with
        RETURNING would be cleaner but older SQLite lacks it — instead we
        SELECT ids, then UPDATE, in the same transaction.
        """
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM commands WHERE status='pending'
                   ORDER BY created_at ASC LIMIT ?""",
                (limit,),
            ).fetchall()
            if not rows:
                return []
            ids = [r["id"] for r in rows]
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"""UPDATE commands SET status='running',
                    picked_at=CURRENT_TIMESTAMP WHERE id IN ({placeholders})""",
                ids,
            )
            return [dict(r) for r in rows]

    def complete_command(self, command_id: int, result: str = "",
                          failed: bool = False):
        with self._conn() as conn:
            conn.execute(
                """UPDATE commands SET status=?, completed_at=CURRENT_TIMESTAMP,
                   result=? WHERE id=?""",
                ("failed" if failed else "done", result, command_id),
            )

    def get_command(self, command_id: int) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM commands WHERE id=?", (command_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_commands(self, status: Optional[str] = None,
                       limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    """SELECT * FROM commands WHERE status=?
                       ORDER BY created_at DESC LIMIT ?""",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM commands ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    # ── Stats ──────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._conn() as conn:
            ideas = conn.execute("SELECT COUNT(*) as c FROM ideas").fetchone()["c"]
            exps = conn.execute("SELECT COUNT(*) as c FROM experiments").fetchone()["c"]
            running = conn.execute(
                "SELECT COUNT(*) as c FROM experiments WHERE status IN ('screening','gating','promoting','deploying')"
            ).fetchone()["c"]
            done = conn.execute(
                "SELECT COUNT(*) as c FROM experiments WHERE status='done'"
            ).fetchone()["c"]
            nodes = conn.execute(
                "SELECT COUNT(*) as c FROM nodes WHERE status='online'"
            ).fetchone()["c"]
            return {
                "total_ideas": ideas, "total_experiments": exps,
                "running_experiments": running, "completed_experiments": done,
                "online_nodes": nodes,
            }
