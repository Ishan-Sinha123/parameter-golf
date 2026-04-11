"""Semantic knowledge base backed by LanceDB + sentence-transformers.

Stores and queries knowledge from three sources:
1. Experiment results — what we've tried, configs, and outcomes
2. Web research findings — what Parallel deep research returned
3. PR evaluations — what other competitors are doing

Uses semantic vector search so the autoresearcher can ask:
- "Have we tried gated linear unit activations?" → matches SwiGLU, GeGLU, etc.
- "What do web sources say about quantization for small models?" → matches int6, GPTQ, AWQ
- "Has anyone tried adaptive learning rate methods?" → matches Muon, Lion, cosine schedule

LanceDB is an embedded vector database (no server needed, like SQLite for vectors).
sentence-transformers/all-MiniLM-L6-v2 provides 384-dim embeddings on CPU.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import lancedb
import pyarrow as pa

log = logging.getLogger(__name__)

# Lazy-loaded embedding model (shared across instances)
_model = None
_model_lock = threading.Lock()
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384


def _get_model():
    """Lazy-load the sentence-transformer model (once, thread-safe)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                log.info("Loading embedding model: %s", EMBED_MODEL)
                _model = SentenceTransformer(EMBED_MODEL)
                log.info("Embedding model loaded (dim=%d)", EMBED_DIM)
    return _model


def embed_text(text: str) -> list[float]:
    """Embed a single text string into a vector."""
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts into vectors (batched for efficiency)."""
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return vecs.tolist()


@dataclass
class KnowledgeEntry:
    """A single piece of knowledge."""
    id: int = 0
    source_type: str = ""       # "experiment", "web_research", "pr_evaluation", "paper"
    source_ref: str = ""        # experiment_id, run_id, PR URL, paper URL
    title: str = ""
    content: str = ""           # Full text — searchable
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)  # Structured data (BPB, config, etc.)
    created_at: Optional[str] = None
    relevance_score: float = 0.0  # Distance score from vector search (lower = more similar)


# LanceDB schema
_SCHEMA = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("source_type", pa.utf8()),
    pa.field("source_ref", pa.utf8()),
    pa.field("title", pa.utf8()),
    pa.field("content", pa.utf8()),
    pa.field("tags", pa.utf8()),        # JSON-encoded list
    pa.field("metadata", pa.utf8()),    # JSON-encoded dict
    pa.field("created_at", pa.utf8()),
    pa.field("vector", pa.list_(pa.float32(), EMBED_DIM)),
])


class KnowledgeBase:
    """LanceDB-backed semantic knowledge base for the autoresearcher."""

    TABLE_NAME = "knowledge"

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.db_path))
        self._next_id = 1
        self._lock = threading.Lock()
        self._ensure_table()

    def _ensure_table(self):
        """Create the table if it doesn't exist, or open it."""
        if self.TABLE_NAME in self._db.table_names():
            self._table = self._db.open_table(self.TABLE_NAME)
            # Find the max ID for auto-increment
            try:
                rows = self._table.to_pandas()
                if len(rows) > 0:
                    self._next_id = int(rows["id"].max()) + 1
            except Exception:
                self._next_id = 1
        else:
            self._table = self._db.create_table(
                self.TABLE_NAME, schema=_SCHEMA,
            )

    def _allocate_id(self) -> int:
        with self._lock:
            eid = self._next_id
            self._next_id += 1
            return eid

    # ── Store ─────────────────────────────────────────────────────────

    def store(
        self,
        source_type: str,
        title: str,
        content: str,
        source_ref: str = "",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """Store a knowledge entry with auto-generated embedding. Returns entry ID."""
        entry_id = self._allocate_id()
        now = datetime.utcnow().isoformat()
        tags_list = tags or []

        # Build embedding from title + content (truncated to keep fast)
        embed_input = f"{title}\n{' '.join(tags_list)}\n{content[:2000]}"
        vector = embed_text(embed_input)

        row = {
            "id": entry_id,
            "source_type": source_type,
            "source_ref": source_ref,
            "title": title,
            "content": content,
            "tags": json.dumps(tags_list),
            "metadata": json.dumps(metadata or {}),
            "created_at": now,
            "vector": vector,
        }
        self._table.add([row])
        log.info("Stored knowledge #%d [%s]: %s", entry_id, source_type, title[:60])
        return entry_id

    def store_experiment_result(
        self,
        experiment_id: str,
        name: str,
        hypothesis: str,
        env_overrides: dict,
        results: dict,
        verdict: str,
        tags: Optional[list[str]] = None,
    ) -> int:
        """Store an experiment's configuration and results as knowledge."""
        config_str = ", ".join(f"{k}={v}" for k, v in env_overrides.items())
        content = (
            f"Experiment: {name}\n"
            f"Hypothesis: {hypothesis}\n"
            f"Configuration: {config_str}\n"
            f"Verdict: {verdict}\n"
            f"Results: {json.dumps(results, indent=2)}"
        )
        return self.store(
            source_type="experiment",
            source_ref=experiment_id,
            title=f"{name} — {verdict}",
            content=content,
            tags=tags,
            metadata={"env_overrides": env_overrides, "results": results,
                       "verdict": verdict},
        )

    def store_web_research(
        self,
        run_id: str,
        query: str,
        findings: str,
        citations: list[dict],
        tags: Optional[list[str]] = None,
    ) -> int:
        """Store web research findings from Parallel."""
        citations = citations or []
        citation_text = "\n".join(
            f"- {c.get('url', 'unknown')}: {c.get('excerpt', '')[:200]}"
            for c in citations[:10]
            if isinstance(c, dict)
        )
        content = (
            f"Research query: {query}\n\n"
            f"Findings:\n{findings}\n\n"
            f"Citations:\n{citation_text}"
        )
        return self.store(
            source_type="web_research",
            source_ref=run_id,
            title=f"Web research: {query[:80]}",
            content=content,
            tags=tags,
            metadata={"query": query, "citation_count": len(citations)},
        )

    def store_pr_evaluation(
        self,
        pr_number: int,
        pr_title: str,
        author: str,
        evaluation: str,
        techniques: list[str],
        reported_bpb: Optional[float] = None,
        url: str = "",
    ) -> int:
        """Store a PR evaluation as knowledge."""
        content = (
            f"PR #{pr_number}: {pr_title}\n"
            f"Author: {author}\n"
            f"Techniques: {', '.join(techniques)}\n"
            f"Reported BPB: {reported_bpb or 'not reported'}\n\n"
            f"Evaluation:\n{evaluation}"
        )
        return self.store(
            source_type="pr_evaluation",
            source_ref=url or f"PR#{pr_number}",
            title=f"PR#{pr_number}: {pr_title}",
            content=content,
            tags=techniques,
            metadata={"pr_number": pr_number, "author": author,
                       "reported_bpb": reported_bpb, "techniques": techniques},
        )

    # ── Query (Semantic Vector Search) ────────────────────────────────

    def search(
        self,
        query: str,
        source_type: Optional[str] = None,
        limit: int = 10,
    ) -> list[KnowledgeEntry]:
        """Semantic search across all knowledge.

        Uses cosine similarity between query embedding and stored embeddings.
        This means "gated linear units" will match entries about SwiGLU/GeGLU,
        "model compression" will match int6/pruning/distillation, etc.

        Args:
            query: Natural language search query.
            source_type: Filter by source — "experiment", "web_research",
                         "pr_evaluation", "paper".
            limit: Max results to return.

        Returns:
            List of KnowledgeEntry, ranked by semantic similarity.
        """
        try:
            query_vec = embed_text(query)
            search_builder = self._table.search(query_vec).limit(limit)

            if source_type:
                search_builder = search_builder.where(
                    f"source_type = '{source_type}'", prefilter=True)

            results_df = search_builder.to_pandas()

            entries = []
            for _, row in results_df.iterrows():
                entries.append(KnowledgeEntry(
                    id=int(row["id"]),
                    source_type=row["source_type"],
                    source_ref=row["source_ref"],
                    title=row["title"],
                    content=row["content"],
                    tags=json.loads(row["tags"]) if row["tags"] else [],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=row["created_at"],
                    relevance_score=float(row.get("_distance", 0.0)),
                ))
            return entries

        except Exception as e:
            log.warning("Knowledge search error for '%s': %s", query, e)
            return []

    def get_by_source(self, source_type: str, source_ref: str) -> Optional[KnowledgeEntry]:
        """Look up a specific entry by source."""
        try:
            ref_escaped = source_ref.replace("'", "''")
            df = self._table.search().where(
                f"source_type = '{source_type}' AND source_ref = '{ref_escaped}'"
            ).limit(1).to_pandas()
            if len(df) == 0:
                return None
            row = df.iloc[0]
            return KnowledgeEntry(
                id=int(row["id"]),
                source_type=row["source_type"],
                source_ref=row["source_ref"],
                title=row["title"],
                content=row["content"],
                tags=json.loads(row["tags"]) if row["tags"] else [],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
            )
        except Exception as e:
            log.warning("get_by_source error: %s", e)
            return None

    def get_experiment_context(self, topic: str, limit: int = 5) -> str:
        """Get a text summary of what we know about a topic from experiments.

        Uses semantic search so "gated activations" matches SwiGLU experiments.
        """
        entries = self.search(topic, source_type="experiment", limit=limit)
        if not entries:
            return f"No previous experiments found related to: {topic}"

        parts = []
        for e in entries:
            meta = e.metadata
            verdict = meta.get("verdict", "unknown")
            results = meta.get("results", {})
            bpb = results.get("screen_ema_bpb") or results.get("gate_int6_bpb")
            bpb_str = f", BPB={bpb:.4f}" if bpb else ""
            dist = f" (sim={1 - e.relevance_score:.2f})" if e.relevance_score else ""
            parts.append(f"- {e.title}{bpb_str}{dist}")
        return f"Previous experiments on '{topic}':\n" + "\n".join(parts)

    def get_web_context(self, topic: str, limit: int = 5) -> str:
        """Get a text summary of web research findings on a topic."""
        entries = self.search(topic, source_type="web_research", limit=limit)
        if not entries:
            return f"No web research found related to: {topic}"

        parts = []
        for e in entries:
            snippet = e.content[:300].replace("\n", " ")
            parts.append(f"- {e.title}: {snippet}...")
        return f"Web research on '{topic}':\n" + "\n".join(parts)

    def get_full_context(self, topic: str, limit: int = 5) -> str:
        """Get combined experiment + web research + PR context for a topic."""
        exp_ctx = self.get_experiment_context(topic, limit)
        web_ctx = self.get_web_context(topic, limit)
        pr_entries = self.search(topic, source_type="pr_evaluation", limit=limit)

        pr_ctx = f"No PR evaluations found related to: {topic}"
        if pr_entries:
            parts = [f"- {e.title}" for e in pr_entries]
            pr_ctx = f"PR evaluations on '{topic}':\n" + "\n".join(parts)

        return f"{exp_ctx}\n\n{web_ctx}\n\n{pr_ctx}"

    def count(self, source_type: Optional[str] = None) -> int:
        """Count entries, optionally filtered by source type."""
        try:
            df = self._table.to_pandas()
            if source_type:
                return int((df["source_type"] == source_type).sum())
            return len(df)
        except Exception:
            return 0

    def recent(self, limit: int = 20, source_type: Optional[str] = None) -> list[KnowledgeEntry]:
        """Get most recent entries."""
        try:
            df = self._table.to_pandas()
            if source_type:
                df = df[df["source_type"] == source_type]
            df = df.sort_values("id", ascending=False).head(limit)
            return [
                KnowledgeEntry(
                    id=int(row["id"]),
                    source_type=row["source_type"],
                    source_ref=row["source_ref"],
                    title=row["title"],
                    content=row["content"],
                    tags=json.loads(row["tags"]) if row["tags"] else [],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    created_at=row["created_at"],
                )
                for _, row in df.iterrows()
            ]
        except Exception as e:
            log.warning("recent() error: %s", e)
            return []
