"""Recipe store: canonical experiment configurations with feature stacking.

A Recipe is the unit of "what does this experiment actually run" —
features + env_overrides + base_commit + a link to its parent recipe in
the stacking DAG. See db/models.py for the full rationale.

This module owns:
- CRUD for the `recipes` and `recipe_pointers` SQLite tables
- Canonicalization + hashing of feature sets (so dedup works)
- YAML mirror files under <workspace>/recipes/<recipe_id>.yaml
- The `current_best_baseline` pointer, updated when experiments finish

The Registry stays generic DB plumbing; this file owns recipe semantics.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from .models import Recipe
from .registry import Registry

log = logging.getLogger(__name__)


# ── Feature canonicalization ──────────────────────────────────────────

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def canonicalize_features(features: list[str]) -> list[str]:
    """Normalize a feature list: lowercase, strip, dedupe, sort.

    We want {'SwiGLU', 'swiglu', ' swiglu '} to collapse to ['swiglu'],
    because otherwise the hash will produce duplicate recipes for the
    same logical configuration.
    """
    cleaned = set()
    for f in features:
        if not isinstance(f, str):
            continue
        s = f.strip().lower()
        if s:
            cleaned.add(s)
    return sorted(cleaned)


def recipe_hash(features: list[str], env_overrides: dict) -> str:
    """Stable hash identifying a recipe by its logical content.

    Two recipes with identical features + env_overrides produce the same
    hash — this is how we dedup during migration and when composing new
    recipes from PRs.
    """
    canonical = canonicalize_features(features)
    env_sorted = sorted(
        (str(k), str(v)) for k, v in (env_overrides or {}).items())
    payload = json.dumps({"f": canonical, "e": env_sorted}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _slug(text: str) -> str:
    """Turn arbitrary text into a safe filename fragment."""
    s = _SLUG_RE.sub("_", text.lower()).strip("_")
    return s[:40] or "unnamed"


def make_recipe_id(name: str, feature_hash: str) -> str:
    """Deterministic human-readable recipe id."""
    date = datetime.utcnow().strftime("%Y%m%d")
    return f"rec_{date}_{_slug(name)}_{feature_hash[:8]}"


# ── Store ─────────────────────────────────────────────────────────────


class RecipeStore:
    """Persistence + semantics for recipes.

    Wraps the Registry for DB access and manages the YAML mirror dir.
    Instances are cheap — the class holds no cached state; everything
    lives in SQLite + YAML files so the dashboard process and worker
    process see the same data.
    """

    CURRENT_BEST_POINTER = "current_best_baseline"

    def __init__(self, registry: Registry, recipes_dir: str | Path):
        self.registry = registry
        self.recipes_dir = Path(recipes_dir)
        self.recipes_dir.mkdir(parents=True, exist_ok=True)

    # ── Creation ──────────────────────────────────────────────────────

    def create(
        self,
        name: str,
        features: list[str],
        env_overrides: dict,
        description: str = "",
        parent_recipe: Optional[str] = None,
        base_commit: str = "",
        base_branch: str = "",
        script_path: str = "train_gpt.py",
        source_experiment: Optional[str] = None,
        dedup: bool = True,
    ) -> Recipe:
        """Create (or return existing) recipe with this logical content.

        If `dedup=True` and a recipe with the same feature_set_hash already
        exists, returns the existing one instead of creating a duplicate.
        This is what makes migration idempotent.
        """
        canon_features = canonicalize_features(features)
        fhash = recipe_hash(canon_features, env_overrides)

        if dedup:
            existing = self.get_by_hash(fhash)
            if existing:
                log.debug("Recipe exists for hash=%s, returning %s",
                          fhash, existing.id)
                return existing

        recipe_id = make_recipe_id(name, fhash)
        recipe = Recipe(
            id=recipe_id,
            name=name,
            description=description,
            parent_recipe=parent_recipe,
            features=canon_features,
            feature_set_hash=fhash,
            env_overrides=dict(env_overrides or {}),
            script_path=script_path,
            base_commit=base_commit,
            base_branch=base_branch,
            source_experiment=source_experiment,
        )

        # Write YAML first so if the DB insert fails, the YAML is still
        # recoverable for manual inspection
        yaml_path = self._write_yaml(recipe)
        recipe.yaml_path = str(yaml_path)

        with self.registry._conn() as conn:
            conn.execute(
                """INSERT INTO recipes (id, name, description, parent_recipe,
                   features, feature_set_hash, env_overrides, script_path,
                   base_commit, base_branch, source_experiment, yaml_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    recipe.id, recipe.name, recipe.description,
                    recipe.parent_recipe, json.dumps(recipe.features),
                    recipe.feature_set_hash,
                    json.dumps(recipe.env_overrides),
                    recipe.script_path, recipe.base_commit,
                    recipe.base_branch, recipe.source_experiment,
                    recipe.yaml_path,
                ),
            )

        log.info("Created recipe %s (%d features, parent=%s)",
                 recipe.id, len(canon_features), parent_recipe or "—")
        return recipe

    def compose(
        self,
        base: Recipe | str,
        new_features: list[str],
        added_env_overrides: dict,
        name: str,
        description: str = "",
        base_commit: str = "",
        base_branch: str = "",
    ) -> Recipe:
        """Stack new features onto an existing recipe.

        The child inherits all parent features + env_overrides and adds
        the new ones on top. Existing env keys are overwritten by the
        new values (this is intentional — lets us tune a hyperparameter
        on top of a stacked recipe without forking the whole chain).
        """
        base_recipe = base if isinstance(base, Recipe) else self.get(base)
        if not base_recipe:
            raise ValueError(f"base recipe not found: {base}")

        stacked_features = list(base_recipe.features) + list(new_features or [])
        stacked_env = dict(base_recipe.env_overrides)
        stacked_env.update(added_env_overrides or {})

        return self.create(
            name=name,
            features=stacked_features,
            env_overrides=stacked_env,
            description=description or f"Stacks {new_features} onto {base_recipe.id}",
            parent_recipe=base_recipe.id,
            base_commit=base_commit or base_recipe.base_commit,
            base_branch=base_branch or base_recipe.base_branch,
            script_path=base_recipe.script_path,
        )

    # ── Reads ─────────────────────────────────────────────────────────

    def get(self, recipe_id: str) -> Optional[Recipe]:
        with self.registry._conn() as conn:
            row = conn.execute(
                "SELECT * FROM recipes WHERE id=?", (recipe_id,),
            ).fetchone()
            return self._row_to_recipe(row) if row else None

    def get_by_hash(self, feature_set_hash: str) -> Optional[Recipe]:
        with self.registry._conn() as conn:
            row = conn.execute(
                "SELECT * FROM recipes WHERE feature_set_hash=? LIMIT 1",
                (feature_set_hash,),
            ).fetchone()
            return self._row_to_recipe(row) if row else None

    def list(
        self,
        order_by: str = "best_val_bpb",
        limit: int = 200,
    ) -> list[Recipe]:
        # Whitelist to prevent SQL injection via caller-supplied order_by
        allowed = {
            "best_val_bpb": "best_val_bpb ASC NULLS LAST",
            "best_int6_bpb": "best_int6_bpb ASC NULLS LAST",
            "created_at": "created_at DESC",
            "name": "name ASC",
        }
        order_clause = allowed.get(order_by, allowed["best_val_bpb"])
        with self.registry._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM recipes ORDER BY {order_clause} LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_recipe(r) for r in rows]

    def list_children(self, parent_recipe_id: str) -> list[Recipe]:
        with self.registry._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM recipes WHERE parent_recipe=? ORDER BY created_at",
                (parent_recipe_id,),
            ).fetchall()
            return [self._row_to_recipe(r) for r in rows]

    def ancestors(self, recipe_id: str) -> list[Recipe]:
        """Return the parent chain from root → recipe (exclusive of recipe)."""
        chain = []
        current = self.get(recipe_id)
        seen = set()
        while current and current.parent_recipe:
            if current.parent_recipe in seen:
                break  # cycle guard, should never happen
            seen.add(current.parent_recipe)
            parent = self.get(current.parent_recipe)
            if not parent:
                break
            chain.append(parent)
            current = parent
        return list(reversed(chain))

    # ── Metric updates ────────────────────────────────────────────────

    def update_best_metrics(
        self,
        recipe_id: str,
        experiment_id: str,
        val_bpb: Optional[float] = None,
        int6_bpb: Optional[float] = None,
        artifact_mb: Optional[float] = None,
    ):
        """Update a recipe's best-observed metrics, if the new run is better.

        "Better" here is strictly lower val_bpb (or lower int6_bpb if that's
        all we have). Ties don't update, keeping the best_experiment_id
        stable across reruns.
        """
        recipe = self.get(recipe_id)
        if not recipe:
            log.warning("update_best_metrics: recipe %s not found", recipe_id)
            return

        improved = False
        new_val = recipe.best_val_bpb
        new_int6 = recipe.best_int6_bpb
        new_art = recipe.best_artifact_mb
        new_best_exp = recipe.best_experiment_id

        # Prefer val_bpb comparison; fall back to int6_bpb
        if val_bpb is not None and (new_val is None or val_bpb < new_val):
            new_val = val_bpb
            improved = True
        if int6_bpb is not None and (new_int6 is None or int6_bpb < new_int6):
            new_int6 = int6_bpb
            improved = True
        if improved:
            if artifact_mb is not None:
                new_art = artifact_mb
            new_best_exp = experiment_id

        if improved:
            with self.registry._conn() as conn:
                conn.execute(
                    """UPDATE recipes SET best_val_bpb=?, best_int6_bpb=?,
                       best_artifact_mb=?, best_experiment_id=? WHERE id=?""",
                    (new_val, new_int6, new_art, new_best_exp, recipe_id),
                )
            log.info("Recipe %s new best: val_bpb=%s int6=%s (exp=%s)",
                     recipe_id, new_val, new_int6, experiment_id)
            # Also update YAML mirror so it stays in sync
            refreshed = self.get(recipe_id)
            if refreshed:
                self._write_yaml(refreshed)

            # Check if this recipe should become the new current_best
            self._maybe_update_current_best(refreshed)

    def _maybe_update_current_best(self, recipe: Recipe):
        """Promote this recipe to current_best_baseline if it beats the pointer."""
        current_best = self.get_pointer(self.CURRENT_BEST_POINTER)
        if current_best is None:
            self.set_pointer(
                self.CURRENT_BEST_POINTER, recipe.id,
                notes="initial best (no prior pointer)")
            return

        def _score(r: Recipe) -> float:
            # Lower is better. Prefer int6 since that's what the competition
            # scores on, fall back to val_bpb.
            if r.best_int6_bpb is not None:
                return r.best_int6_bpb
            if r.best_val_bpb is not None:
                return r.best_val_bpb
            return float("inf")

        if _score(recipe) < _score(current_best):
            self.set_pointer(
                self.CURRENT_BEST_POINTER, recipe.id,
                notes=f"beat {current_best.id} "
                       f"({_score(current_best):.4f} → {_score(recipe):.4f})")

    # ── Pointers ──────────────────────────────────────────────────────

    def set_pointer(self, name: str, recipe_id: str, notes: str = ""):
        with self.registry._conn() as conn:
            conn.execute(
                """INSERT INTO recipe_pointers (name, recipe_id, notes)
                   VALUES (?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET
                     recipe_id=excluded.recipe_id,
                     updated_at=CURRENT_TIMESTAMP,
                     notes=excluded.notes""",
                (name, recipe_id, notes),
            )
        log.info("Pointer %s -> %s (%s)", name, recipe_id, notes)

    def get_pointer(self, name: str) -> Optional[Recipe]:
        with self.registry._conn() as conn:
            row = conn.execute(
                "SELECT recipe_id FROM recipe_pointers WHERE name=?", (name,),
            ).fetchone()
        if not row:
            return None
        return self.get(row["recipe_id"])

    def current_best(self) -> Optional[Recipe]:
        return self.get_pointer(self.CURRENT_BEST_POINTER)

    # ── YAML mirror ───────────────────────────────────────────────────

    def _write_yaml(self, recipe: Recipe) -> Path:
        path = self.recipes_dir / f"{recipe.id}.yaml"
        payload = {
            "id": recipe.id,
            "name": recipe.name,
            "description": recipe.description,
            "parent_recipe": recipe.parent_recipe,
            "features": recipe.features,
            "feature_set_hash": recipe.feature_set_hash,
            "env_overrides": recipe.env_overrides,
            "script_path": recipe.script_path,
            "base_commit": recipe.base_commit,
            "base_branch": recipe.base_branch,
            "source_experiment": recipe.source_experiment,
            "best_val_bpb": recipe.best_val_bpb,
            "best_int6_bpb": recipe.best_int6_bpb,
            "best_artifact_mb": recipe.best_artifact_mb,
            "best_experiment_id": recipe.best_experiment_id,
        }
        with open(path, "w") as f:
            yaml.dump(payload, f, default_flow_style=False, sort_keys=False)
        return path

    # ── Row mapping ───────────────────────────────────────────────────

    def _row_to_recipe(self, row) -> Recipe:
        return Recipe(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            parent_recipe=row["parent_recipe"],
            features=json.loads(row["features"]) if row["features"] else [],
            feature_set_hash=row["feature_set_hash"],
            env_overrides=json.loads(row["env_overrides"])
                if row["env_overrides"] else {},
            script_path=row["script_path"] or "train_gpt.py",
            base_commit=row["base_commit"] or "",
            base_branch=row["base_branch"] or "",
            source_experiment=row["source_experiment"],
            best_val_bpb=row["best_val_bpb"],
            best_int6_bpb=row["best_int6_bpb"],
            best_artifact_mb=row["best_artifact_mb"],
            best_experiment_id=row["best_experiment_id"],
            created_at=row["created_at"],
            yaml_path=row["yaml_path"] or "",
        )


# ── Naive baseline seed (from the Parameter Golf README) ──────────────

NAIVE_BASELINE_ENV = {
    "NUM_LAYERS": "9",
    "MODEL_DIM": "512",
    "NUM_HEADS": "8",
    "NUM_KV_HEADS": "4",
    "MLP_MULT": "2",
    "VOCAB_SIZE": "1024",
    "TIE_EMBEDDINGS": "1",
}

NAIVE_BASELINE_FEATURES = [
    "9_layers", "512_dim", "1024_vocab", "tied_embeddings", "4_kv_heads",
]


def seed_naive_baseline(store: RecipeStore, base_commit: str = "") -> Recipe:
    """Seed the naive baseline from the Parameter Golf README.

    This is the 1.2244 BPB baseline every new submission stacks on top of.
    Idempotent — returns the existing recipe if it's already been seeded.
    """
    return store.create(
        name="naive_baseline",
        features=NAIVE_BASELINE_FEATURES,
        env_overrides=NAIVE_BASELINE_ENV,
        description=(
            "Parameter Golf naive baseline from the README: "
            "9 layers, 512 dim, 1024 vocab, tied embeddings, 4 KV heads. "
            "Reference val_bpb = 1.2244 on FineWeb."
        ),
        base_commit=base_commit,
        base_branch="main",
    )
