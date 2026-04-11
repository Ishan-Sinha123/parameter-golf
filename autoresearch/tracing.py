"""Lightweight span-tree tracer backed by the SQLite registry DB.

Spans are OpenTelemetry-shaped: each span has an id, a trace_id (root
correlation id), an optional parent_span_id, a kind (what the span does:
poll_cycle, pr_eval, deep_research, experiment, stage, ssh_probe, …), a
human-readable name, optional entity pointer (entity_type/entity_id) back
into ideas/experiments/recipes/prs, a status (running/ok/error/timeout),
started/ended timestamps, and free-form attrs.

Usage:

    from autoresearch.tracing import tracer

    with tracer.span("poll_cycle", name="research_poll") as s:
        s.set("new_prs", 3)
        with tracer.span("pr_eval", name=f"PR#{pr.number}",
                         entity=("pr", str(pr.number))) as s2:
            s2.set("novelty", 3)

Parent/child nesting is implicit via a contextvars.ContextVar, so it
works across nested function calls without threading a tracer argument.

The tracer is a singleton configured once at startup by calling
`tracing.init(db_path)`. Before init, spans are no-ops — this keeps
imports cheap and short-circuits tests that don't need tracing.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

log = logging.getLogger(__name__)

_current_span: contextvars.ContextVar[Optional["Span"]] = contextvars.ContextVar(
    "autoresearch_current_span", default=None)


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


class Span:
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    kind: str
    name: str
    entity_type: str
    entity_id: str
    started_at: float
    ended_at: Optional[float]
    status: str
    attrs: dict[str, Any]
    error: str

    __slots__ = ("span_id", "trace_id", "parent_span_id", "kind", "name",
                 "entity_type", "entity_id", "started_at", "ended_at",
                 "status", "attrs", "error", "_tracer")

    def __init__(self, tracer: "Tracer", *, kind: str, name: str,
                 entity: Optional[tuple[str, str]],
                 parent: Optional["Span"]):
        self._tracer = tracer
        self.span_id = _new_id()
        self.trace_id = parent.trace_id if parent is not None else _new_id()
        self.parent_span_id = parent.span_id if parent is not None else None
        self.kind = kind
        self.name = name
        self.entity_type = entity[0] if entity else ""
        self.entity_id = entity[1] if entity else ""
        self.started_at = time.time()
        self.ended_at = None
        self.status = "running"
        self.attrs = {}
        self.error = ""

    def set(self, key: str, value: Any) -> None:
        try:
            json.dumps(value)
            self.attrs[key] = value
        except (TypeError, ValueError):
            self.attrs[key] = repr(value)[:500]

    def set_status(self, status: str) -> None:
        self.status = status

    def fail(self, error: str) -> None:
        self.status = "error"
        self.error = error[:1000]


class Tracer:
    """Singleton tracer. No-op until `init(db_path)` is called."""

    def __init__(self) -> None:
        self._db_path: Optional[Path] = None
        self._lock = threading.Lock()
        self._enabled = False

    def init(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._enabled = True
        log.info("Tracer initialized: %s", self._db_path)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def span(self, kind: str, *, name: str = "",
             entity: Optional[tuple[str, str]] = None,
             **attrs: Any) -> Iterator[Span]:
        """Open a span. If tracing is disabled, yields a throwaway Span.

        Nesting is implicit via contextvars — no need to pass spans around.
        """
        if not self._enabled:
            # Yield a bare Span object that doesn't touch the DB.
            dummy = Span.__new__(Span)
            dummy.span_id = ""
            dummy.trace_id = ""
            dummy.parent_span_id = None
            dummy.kind = kind
            dummy.name = name or kind
            dummy.entity_type = entity[0] if entity else ""
            dummy.entity_id = entity[1] if entity else ""
            dummy.started_at = time.time()
            dummy.ended_at = None
            dummy.status = "running"
            dummy.attrs = {}
            dummy.error = ""
            dummy._tracer = self
            try:
                yield dummy
            except Exception:
                raise
            return

        parent = _current_span.get()
        sp = Span(self, kind=kind, name=name or kind, entity=entity,
                  parent=parent)
        for k, v in attrs.items():
            sp.set(k, v)

        self._write_start(sp)
        token = _current_span.set(sp)
        try:
            yield sp
            if sp.status == "running":
                sp.status = "ok"
        except Exception as e:
            sp.fail(f"{type(e).__name__}: {e}")
            raise
        finally:
            _current_span.reset(token)
            sp.ended_at = time.time()
            self._write_end(sp)

    # ── Back-reference: attach an entity to the current span post-hoc ──

    def set_current_entity(self, entity_type: str, entity_id: str) -> None:
        sp = _current_span.get()
        if sp is None or not self._enabled:
            return
        sp.entity_type = entity_type
        sp.entity_id = entity_id
        with self._connect() as conn:
            conn.execute(
                "UPDATE traces SET entity_type=?, entity_id=? WHERE span_id=?",
                (entity_type, entity_id, sp.span_id),
            )

    def set_current_attr(self, key: str, value: Any) -> None:
        sp = _current_span.get()
        if sp is None:
            return
        sp.set(key, value)

    def record(self, *, kind: str, name: str,
               started_at: float, ended_at: float,
               entity: Optional[tuple[str, str]] = None,
               parent_trace_id: Optional[str] = None,
               status: str = "ok",
               attrs: Optional[dict[str, Any]] = None,
               error: str = "") -> Optional[str]:
        """Write a retroactive, already-closed span.

        Useful for async completions (e.g. experiment stage lifetimes
        that span the deploy→poll→completion cycle across threads).
        Returns the span_id, or None if tracing is disabled.
        """
        if not self._enabled:
            return None
        span_id = _new_id()
        trace_id = parent_trace_id or _new_id()
        duration_ms = int((ended_at - started_at) * 1000)
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """INSERT INTO traces (span_id, trace_id, parent_span_id,
                        kind, name, entity_type, entity_id, status,
                        started_at, ended_at, duration_ms, attrs, error)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (span_id, trace_id, None, kind, name,
                     entity[0] if entity else "",
                     entity[1] if entity else "",
                     status, started_at, ended_at, duration_ms,
                     json.dumps(attrs or {}), error),
                )
                conn.commit()
            return span_id
        except Exception as e:
            log.debug("trace record failed: %s", e)
            return None

    # ── DB writes ─────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        assert self._db_path is not None
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _write_start(self, sp: Span) -> None:
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """INSERT INTO traces (span_id, trace_id, parent_span_id,
                        kind, name, entity_type, entity_id, status,
                        started_at, attrs)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (sp.span_id, sp.trace_id, sp.parent_span_id,
                     sp.kind, sp.name, sp.entity_type, sp.entity_id,
                     sp.status, sp.started_at, json.dumps(sp.attrs)),
                )
                conn.commit()
        except Exception as e:
            log.debug("trace write_start failed: %s", e)

    def _write_end(self, sp: Span) -> None:
        try:
            duration_ms = None
            if sp.ended_at is not None:
                duration_ms = int((sp.ended_at - sp.started_at) * 1000)
            with self._lock, self._connect() as conn:
                conn.execute(
                    """UPDATE traces SET
                        ended_at=?, duration_ms=?, status=?, attrs=?, error=?
                       WHERE span_id=?""",
                    (sp.ended_at, duration_ms, sp.status,
                     json.dumps(sp.attrs), sp.error, sp.span_id),
                )
                conn.commit()
        except Exception as e:
            log.debug("trace write_end failed: %s", e)


# Module-level singleton
tracer = Tracer()


def init(db_path: str | Path) -> None:
    tracer.init(db_path)
