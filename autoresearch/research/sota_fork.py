"""Fork a records/ SOTA submission into an autoresearch baseline Recipe.

Records stored under `records/track_10min_16mb/<name>/train_gpt.py` are
distributed as a 2-line lzma-base85-exec blob. This module:

1. Decodes the blob into readable Python source.
2. Extracts the env-var defaults that parameterize the recipe.
3. Writes the decoded source to `autoresearch/baselines/<recipe_id>/train_gpt.py`.
4. Creates a Recipe row with script_path pointing at that file.
5. Updates `recipe_pointers.current_best_baseline` iff the new recipe's
   val_bpb strictly beats the current pointer.
6. Auto-commits the decoded source + recipe YAML to the deploy branch.

Subsequent experiments — whose launch path resolves `script_path` from
`exp.recipe_id` — will then run on top of the forked SOTA instead of the
vanilla root `train_gpt.py`.
"""
from __future__ import annotations

import base64
import json
import logging
import lzma
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..db.models import Recipe
from ..db.recipes import RecipeStore

log = logging.getLogger(__name__)


# ── Decode ────────────────────────────────────────────────────────────

_B85_CALL_OPEN = "b85decode("


def _extract_b85_payload(raw: str) -> str:
    """Extract the (possibly multi-chunk) string literal passed to b85decode.

    The challenge: the base85 alphabet includes `(` and `)`, so naive
    paren/regex matching will stop inside the payload itself. We locate
    `b85decode(` and then scan forward, collecting every `"..."` or
    `'...'` string literal separated only by whitespace (Python's implicit
    concatenation). The first non-string, non-whitespace character (e.g.
    the closing `)`) ends the argument.
    """
    idx = raw.find(_B85_CALL_OPEN)
    if idx < 0:
        raise ValueError("Could not locate b85decode call")
    i = idx + len(_B85_CALL_OPEN)
    n = len(raw)
    chunks: list[str] = []
    while i < n:
        # Skip whitespace
        while i < n and raw[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        ch = raw[i]
        if ch not in ("\"", "'"):
            break
        # Find matching closing quote. Record files don't use escaped
        # quotes inside the blob, so a simple scan is sufficient.
        j = raw.find(ch, i + 1)
        if j < 0:
            raise ValueError("Unterminated string in b85decode argument")
        chunks.append(raw[i + 1 : j])
        i = j + 1
    if not chunks:
        raise ValueError("No string literals found inside b85decode(...)")
    return "".join(chunks)


def decode_record_train_gpt(path: Path) -> str:
    """Return the readable Python source of a record's train_gpt.py.

    Handles both the `exec(lzma.decompress(b85decode(...)))` blob format
    used by competition submissions and plain Python files (which are
    returned as-is). The b85 payload may be a single string literal OR
    multiple adjacent string literals that Python implicitly concatenates.
    """
    raw = path.read_text()
    if "b85decode" not in raw:
        # Plain python file
        return raw

    try:
        payload = _extract_b85_payload(raw)
    except ValueError as e:
        raise ValueError(f"{e} in {path}") from None
    blob = base64.b85decode(payload)
    try:
        decoded = lzma.decompress(
            blob, format=lzma.FORMAT_RAW,
            filters=[{"id": lzma.FILTER_LZMA2}],
        )
    except lzma.LZMAError:
        # Fallback: some older records may use autodetect format
        decoded = lzma.decompress(blob)
    return decoded.decode("utf-8")


# ── Extract env defaults ──────────────────────────────────────────────

# Matches things like:
#   int(os.environ.get('NUM_LAYERS', 11))
#   float(os.environ.get('MLP_MULT', 4.))
#   os.environ.get('DATA_DIR', './data/')
#   bool(int(os.environ.get('TIE_EMBEDDINGS', '1')))
_ENV_DEFAULT_RE = re.compile(
    r"os\.environ\.get\(\s*[\"'](?P<key>[A-Z_][A-Z0-9_]*)[\"']\s*,\s*"
    r"(?P<val>[^)]+?)\s*\)"
)

# Env keys whose values are structural/path-like and should not be baked
# into the recipe (they're instance-specific, not recipe-specific).
_INSTANCE_ENV_KEYS = {
    "DATA_DIR", "RUN_ID", "SEED", "MAX_WALLCLOCK_SECONDS",
    "RESULTS_TSV_PATH", "LOG_DIR", "EXPERIMENT_DESC",
}


_NUMERIC_RE = re.compile(r"^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _clean_value(val: str) -> Optional[str]:
    """Normalize a default expression from os.environ.get(..., DEFAULT).

    Returns None if the default is a variable reference (e.g. `_D`) or
    any non-literal expression we can't safely bake into env_overrides.
    Baking a variable name as a string default causes the recipe to set
    the env var to the literal symbol at runtime, which then blows up
    when the script tries to convert it to int/float.
    """
    v = val.strip()
    # Strip surrounding quotes → string literal
    if (v.startswith("'") and v.endswith("'")) or (
            v.startswith('"') and v.endswith('"')):
        return v[1:-1]
    # Numeric literals (ints, floats with trailing dot, scientific notation)
    if _NUMERIC_RE.match(v):
        return v
    # Everything else (`_D`, `N_LAYERS`, `uuid.uuid4()`, `None`, etc.) is
    # not a safe literal — let the script fall back to its own default
    # rather than forcing a string into the env.
    return None


def extract_env_defaults(source: str) -> dict[str, str]:
    """Pull every `os.environ.get(KEY, DEFAULT)` default out of source."""
    env: dict[str, str] = {}
    for m in _ENV_DEFAULT_RE.finditer(source):
        key = m.group("key")
        if key in _INSTANCE_ENV_KEYS:
            continue
        if key in env:
            continue  # first occurrence wins
        cleaned = _clean_value(m.group("val"))
        if cleaned is None:
            continue
        env[key] = cleaned
    return env


# ── Draft + install ───────────────────────────────────────────────────

@dataclass
class RecipeDraft:
    name: str
    description: str
    features: list[str]
    env_overrides: dict[str, str]
    val_bpb: Optional[float]
    artifact_mb: Optional[float]
    source_ref: str  # e.g. "records/track_10min_16mb/<name>"
    base_branch: str = ""
    base_commit: str = ""


def _submission_to_features(submission: dict) -> list[str]:
    """Derive canonical feature tags from submission.json fields."""
    feats: list[str] = []
    summary = (submission.get("technique_summary") or "").lower()
    # Cheap keyword-based feature tagging — canonicalize_features will
    # lowercase+dedupe anyway.
    markers = [
        ("sp8192", "sp8192"), ("sp1024", "sp1024"),
        ("depth recurrence", "depth_recurrence"),
        ("parallel residuals", "parallel_residuals"),
        ("qk-gain", "qk_gain"), ("qk gain", "qk_gain"),
        ("ttt", "legal_ttt"), ("gptq", "gptq"),
        ("sdclip", "sdclip"), ("brotli", "brotli"),
        ("ema", "ema"), ("muon", "muon"),
    ]
    for needle, tag in markers:
        if needle in summary:
            feats.append(tag)
    feats.append("sota_fork")
    return feats


def extract_recipe_metadata(
    source: str,
    submission: dict,
    record_dir: Path,
) -> RecipeDraft:
    """Build a RecipeDraft from decoded source + its submission.json."""
    env_overrides = extract_env_defaults(source)
    features = _submission_to_features(submission)

    name = submission.get("name") or record_dir.name
    # Short name for the recipe id slug
    short_name = record_dir.name

    val_bpb = submission.get("val_bpb")
    try:
        val_bpb = float(val_bpb) if val_bpb is not None else None
    except (TypeError, ValueError):
        val_bpb = None

    # artifact size: pick the smallest seed artifact if given
    artifact_mb = None
    seed_results = submission.get("seed_results") or {}
    if seed_results:
        arts = [
            float(r.get("artifact_bytes", 0)) / (1024 * 1024)
            for r in seed_results.values()
            if r.get("artifact_bytes")
        ]
        if arts:
            artifact_mb = min(arts)

    description = (
        f"Forked from {record_dir.name}. "
        f"{submission.get('technique_summary', '')[:300]}"
    )
    return RecipeDraft(
        name=short_name,
        description=description,
        features=features,
        env_overrides=env_overrides,
        val_bpb=val_bpb,
        artifact_mb=artifact_mb,
        source_ref=str(record_dir),
    )


def install_as_baseline(
    draft: RecipeDraft,
    decoded_source: str,
    *,
    recipes_store: RecipeStore,
    repo_root: Path,
    auto_commit: bool = True,
    promote_to_current_best: bool = True,
) -> Recipe:
    """Create Recipe + write baseline source file + (maybe) bump pointer.

    This is idempotent: if a recipe with the same feature hash already
    exists, returns the existing row and only updates the pointer if the
    caller asked for it and the score beats the incumbent.
    """
    recipe = recipes_store.create(
        name=draft.name,
        features=draft.features,
        env_overrides=draft.env_overrides,
        description=draft.description,
        # script_path is filled in after we know the recipe id
        script_path="train_gpt.py",
        base_branch=draft.base_branch,
        base_commit=draft.base_commit,
    )

    # Write decoded source under autoresearch/baselines/<recipe_id>/
    baselines_dir = repo_root / "autoresearch" / "baselines" / recipe.id
    baselines_dir.mkdir(parents=True, exist_ok=True)
    source_path = baselines_dir / "train_gpt.py"
    source_path.write_text(decoded_source)

    # Metadata sidecar so humans can trace where this came from.
    meta = {
        "recipe_id": recipe.id,
        "source_ref": draft.source_ref,
        "val_bpb": draft.val_bpb,
        "artifact_mb": draft.artifact_mb,
        "features": draft.features,
    }
    (baselines_dir / "recipe.json").write_text(json.dumps(meta, indent=2))

    # Update the recipe row's script_path to point at the decoded file.
    rel_script = f"autoresearch/baselines/{recipe.id}/train_gpt.py"
    with recipes_store.registry._conn() as conn:
        conn.execute(
            "UPDATE recipes SET script_path=? WHERE id=?",
            (rel_script, recipe.id),
        )
    # Also seed best_val_bpb so the leaderboard ranks it immediately.
    if draft.val_bpb is not None:
        with recipes_store.registry._conn() as conn:
            conn.execute(
                "UPDATE recipes SET best_val_bpb=COALESCE(best_val_bpb, ?),"
                " best_artifact_mb=COALESCE(best_artifact_mb, ?)"
                " WHERE id=?",
                (draft.val_bpb, draft.artifact_mb, recipe.id),
            )
    refreshed = recipes_store.get(recipe.id) or recipe

    # Promote to current_best_baseline iff strictly better.
    if promote_to_current_best and draft.val_bpb is not None:
        current = recipes_store.current_best()
        incumbent_score = _score(current) if current else float("inf")
        if draft.val_bpb < incumbent_score:
            recipes_store.set_pointer(
                RecipeStore.CURRENT_BEST_POINTER, refreshed.id,
                notes=(f"sota_fork: {draft.source_ref} "
                       f"val_bpb={draft.val_bpb:.4f} "
                       f"(prev={incumbent_score:.4f})"),
            )
            log.info("Promoted %s to current_best_baseline (val_bpb=%.4f)",
                     refreshed.id, draft.val_bpb)

    if auto_commit:
        _git_commit_baseline(repo_root, refreshed.id, draft.val_bpb)

    return refreshed


def _score(recipe: Optional[Recipe]) -> float:
    if not recipe:
        return float("inf")
    if recipe.best_int6_bpb is not None:
        return recipe.best_int6_bpb
    if recipe.best_val_bpb is not None:
        return recipe.best_val_bpb
    return float("inf")


def _git_commit_baseline(
    repo_root: Path, recipe_id: str, val_bpb: Optional[float]
) -> None:
    """Best-effort commit the new baseline + recipe YAML to HEAD branch.

    Uses narrow `git add` so we don't sweep up unrelated dirty state. No
    push — the scheduler's `_ensure_pushed()` handles that before the
    next experiment launch.
    """
    rel_dir = f"autoresearch/baselines/{recipe_id}"
    rel_yaml = f"autoresearch/recipes/{recipe_id}.yaml"
    try:
        subprocess.run(
            ["git", "add", rel_dir, rel_yaml],
            cwd=repo_root, capture_output=True, timeout=10,
        )
        msg = f"autoresearch: fork SOTA baseline {recipe_id}"
        if val_bpb is not None:
            msg += f" (val_bpb={val_bpb:.4f})"
        r = subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=repo_root, capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0:
            log.info("Committed baseline %s", recipe_id)
        else:
            # Likely "nothing to commit" — benign
            log.debug("git commit for %s: %s", recipe_id, r.stderr.strip())
    except Exception as e:
        log.warning("Baseline commit failed for %s: %s", recipe_id, e)


# ── Sync from records tree ────────────────────────────────────────────


def sync_from_records(
    records_dir: Path,
    *,
    recipes_store: RecipeStore,
    repo_root: Path,
    min_improvement: float = 0.0,
) -> list[Recipe]:
    """Scan records/ for SOTA submissions; fork any that beat the incumbent.

    Idempotent: recipes dedupe by feature hash, decoded-source writes
    overwrite in place, and the pointer only moves when strictly better.
    """
    installed: list[Recipe] = []
    if not records_dir.exists():
        return installed

    # Current incumbent score
    current = recipes_store.current_best()
    incumbent = _score(current) - min_improvement

    for sub_path in sorted(records_dir.glob("*/submission.json")):
        try:
            submission = json.loads(sub_path.read_text())
        except Exception as e:
            log.warning("Bad submission.json %s: %s", sub_path, e)
            continue
        val_bpb = submission.get("val_bpb")
        try:
            val_bpb = float(val_bpb) if val_bpb is not None else None
        except (TypeError, ValueError):
            val_bpb = None
        if val_bpb is None:
            continue
        # Only fork if it would actually beat the current best (or if we
        # have no current best yet — bootstrap case).
        if current is not None and val_bpb >= incumbent:
            continue

        record_dir = sub_path.parent
        train_file = record_dir / "train_gpt.py"
        if not train_file.exists():
            continue
        try:
            decoded = decode_record_train_gpt(train_file)
        except Exception as e:
            log.warning("Decode failed for %s: %s", train_file, e)
            continue

        draft = extract_recipe_metadata(decoded, submission, record_dir)
        try:
            recipe = install_as_baseline(
                draft, decoded,
                recipes_store=recipes_store,
                repo_root=repo_root,
            )
            installed.append(recipe)
            # Update incumbent so we don't install an older record next
            incumbent = min(incumbent, val_bpb)
        except Exception as e:
            log.exception("install_as_baseline failed for %s: %s",
                          record_dir, e)
    if installed:
        log.info("sync_from_records: installed %d new baseline(s)",
                 len(installed))
    return installed
