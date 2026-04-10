"""Research retrieval CLI — local record search, arxiv, and GitHub PR lookup.

Subcommands::

    python -m autoresearch.scripts.retriever search "query"
    python -m autoresearch.scripts.retriever arxiv "query"
    python -m autoresearch.scripts.retriever pr NUMBER
    python -m autoresearch.scripts.retriever records --bottleneck "query"
    python -m autoresearch.scripts.retriever propose --parent EXP_ID --bottleneck "query"

Results are cached in autoresearch/db/retriever_cache.json.
"""

from __future__ import annotations

__all__ = ["Retriever", "RecordEntry", "ProposedRoute"]

import argparse
import json
import logging
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

logger = logging.getLogger(__name__)

RECORDS_DIR = Path(__file__).parent.parent.parent / "records"
DB_DIR = Path(__file__).parent.parent / "db"
CACHE_FILE = DB_DIR / "retriever_cache.json"
ARXIV_API = "https://export.arxiv.org/api/query"
GITHUB_API = "https://api.github.com"
GITHUB_REPO = "JianYan11/parameter-golf"
REQUEST_TIMEOUT = 10

# Mechanism categories for propose diversity constraint.
_MECHANISMS = ("architecture", "quantization", "optimizer")


@dataclass(frozen=True)
class RecordEntry:
    """A parsed submission from records/."""

    path: str
    author: str
    name: str
    val_bpb: float
    bytes_total: int
    summary: str
    techniques: List[str] = field(default_factory=list)
    ablation_tables: List[str] = field(default_factory=list)
    relevance_score: float = 0.0


@dataclass(frozen=True)
class ProposedRoute:
    """A proposed experiment route from records-mining."""

    name: str
    mechanism: str
    hypothesis: str
    cited_records: List[Dict[str, Any]]
    env_overrides: Dict[str, str]
    caveats: List[str]


class RetrieverError(Exception):
    """Raised when a retrieval operation fails."""


class Retriever:
    """Retrieve research artifacts from local records, arxiv, and GitHub."""

    def __init__(
        self,
        records_dir: Path = RECORDS_DIR,
        cache_file: Path = CACHE_FILE,
    ) -> None:
        self._records_dir = records_dir
        self._cache_file = cache_file
        self._cache: Dict[str, Any] = self._load_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Grep local records/ directory for files matching *query*.

        Returns:
            List of dicts with keys: path, line, text.
        """
        results: List[Dict[str, Any]] = []
        if not self._records_dir.exists():
            logger.warning("Records directory not found: %s", self._records_dir)
            return results

        query_lower = query.lower()
        for path in sorted(self._records_dir.rglob("*")):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                logger.debug("Cannot read %s: %s", path, exc)
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if query_lower in line.lower():
                    results.append(
                        {
                            "path": str(path.relative_to(self._records_dir)),
                            "line": lineno,
                            "text": line.strip(),
                        }
                    )
        logger.info("search(%r): %d matches in records/", query, len(results))
        return results

    def arxiv(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search arxiv via the export API.

        Results are cached; subsequent calls with the same query return cached data.

        Returns:
            List of dicts with keys: id, title, authors, summary, published, url.
        """
        cache_key = f"arxiv:{query}:{max_results}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for arxiv query %r", query)
            return cached  # type: ignore[return-value]

        params = urllib.parse.urlencode(
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
            }
        )
        url = f"{ARXIV_API}?{params}"
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            logger.error("arxiv API request failed: %s", exc)
            return []

        results = self._parse_arxiv_feed(body)
        self._cache[cache_key] = results
        self._save_cache()
        logger.info("arxiv(%r): %d results", query, len(results))
        return results

    def pr(self, number: int) -> Optional[Dict[str, Any]]:
        """Fetch a GitHub PR by number from GITHUB_REPO.

        Results are cached.

        Returns:
            Dict with PR metadata, or None if the request fails.
        """
        cache_key = f"pr:{GITHUB_REPO}:{number}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for PR #%d", number)
            return cached  # type: ignore[return-value]

        url = f"{GITHUB_API}/repos/{GITHUB_REPO}/pulls/{number}"
        try:
            req = urllib.request.Request(
                url,
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            logger.error("GitHub API HTTP error for PR #%d: %s", number, exc)
            return None
        except urllib.error.URLError as exc:
            logger.error("GitHub API request failed for PR #%d: %s", number, exc)
            return None

        result = {
            "number": data.get("number"),
            "title": data.get("title"),
            "state": data.get("state"),
            "author": (data.get("user") or {}).get("login"),
            "body": data.get("body"),
            "created_at": data.get("created_at"),
            "merged_at": data.get("merged_at"),
            "url": data.get("html_url"),
        }
        self._cache[cache_key] = result
        self._save_cache()
        logger.info("Fetched PR #%d: %s", number, result.get("title"))
        return result

    def records(self, bottleneck: str) -> List[RecordEntry]:
        """Scan records/ directory and return entries ranked by relevance.

        For each submission, parses ``submission.json`` (val_bpb, bytes_total,
        author, summary) and ``README.md`` (technique descriptions, ablation
        tables). Results are ranked by keyword overlap with *bottleneck*.

        Args:
            bottleneck: Query describing the current research bottleneck.

        Returns:
            List of RecordEntry objects sorted by relevance (highest first).
        """
        entries: List[RecordEntry] = []
        if not self._records_dir.exists():
            logger.warning("Records directory not found: %s", self._records_dir)
            return entries

        bottleneck_lower = bottleneck.lower()
        bottleneck_words = set(re.findall(r"\w+", bottleneck_lower))

        for track_dir in sorted(self._records_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            for sub_dir in sorted(track_dir.iterdir()):
                if not sub_dir.is_dir():
                    continue

                submission_path = sub_dir / "submission.json"
                if not submission_path.exists():
                    continue

                try:
                    data = json.loads(
                        submission_path.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, OSError) as exc:
                    logger.debug("Cannot read %s: %s", submission_path, exc)
                    continue

                val_bpb = float(data.get("val_bpb", 0.0))
                bytes_total = int(data.get("bytes_total", 0))
                author = str(data.get("author", "unknown"))
                name = str(data.get("name", sub_dir.name))
                summary = str(data.get("blurb", ""))

                # Parse README for techniques and ablation tables
                techniques: List[str] = []
                ablation_tables: List[str] = []
                readme_path = sub_dir / "README.md"
                readme_text = ""
                if readme_path.exists():
                    try:
                        readme_text = readme_path.read_text(
                            encoding="utf-8", errors="replace"
                        )
                    except OSError:
                        pass

                    # Extract technique bullet points (lines starting with
                    # a number followed by **bold text**)
                    for m in re.finditer(
                        r"^\d+\.\s+\*\*(.+?)\*\*",
                        readme_text,
                        re.MULTILINE,
                    ):
                        techniques.append(m.group(1).strip())

                    # Extract markdown tables (lines with |)
                    table_lines: List[str] = []
                    for line in readme_text.splitlines():
                        if "|" in line and line.strip().startswith("|"):
                            table_lines.append(line.strip())
                        elif table_lines:
                            ablation_tables.append("\n".join(table_lines))
                            table_lines = []
                    if table_lines:
                        ablation_tables.append("\n".join(table_lines))

                # Score relevance by keyword overlap with bottleneck
                searchable = (
                    f"{name} {summary} {' '.join(techniques)} "
                    f"{readme_text}"
                ).lower()
                searchable_words = set(re.findall(r"\w+", searchable))
                overlap = len(bottleneck_words & searchable_words)
                # Bonus for exact phrase match
                if bottleneck_lower in searchable:
                    overlap += len(bottleneck_words)
                # Normalize by query length
                relevance = overlap / max(len(bottleneck_words), 1)

                rel_path = str(sub_dir.relative_to(self._records_dir))
                entries.append(
                    RecordEntry(
                        path=rel_path,
                        author=author,
                        name=name,
                        val_bpb=val_bpb,
                        bytes_total=bytes_total,
                        summary=summary,
                        techniques=techniques,
                        ablation_tables=ablation_tables,
                        relevance_score=relevance,
                    )
                )

        # Sort by relevance descending, then by val_bpb ascending (lower is better)
        entries.sort(key=lambda e: (-e.relevance_score, e.val_bpb))
        logger.info(
            "records(%r): %d entries found, top relevance=%.2f",
            bottleneck,
            len(entries),
            entries[0].relevance_score if entries else 0.0,
        )
        return entries

    def propose(
        self,
        parent: str,
        bottleneck: str,
    ) -> List[ProposedRoute]:
        """Propose exactly 3 experiment routes from records-mining.

        Each route differs in mechanism (architecture vs quantization vs
        optimizer) per §4.1 of the spec. Routes cite specific records
        with val_bpb gaps and include env_overrides and caveats.

        Args:
            parent: Parent experiment ID for lineage.
            bottleneck: Description of current research bottleneck.

        Returns:
            List of exactly 3 ProposedRoute objects.
        """
        all_entries = self.records(bottleneck)
        if not all_entries:
            logger.warning("No records found — returning empty proposals")
            return []

        # Classify entries by mechanism based on technique keywords
        arch_keywords = {
            "layer", "layers", "mlp", "attention", "head", "heads", "dim",
            "transformer", "skip", "u-net", "unet", "depth", "width",
            "expansion", "gqa", "rope", "relu", "swiglu", "gelu",
            "architecture", "block", "blocks",
        }
        quant_keywords = {
            "quant", "quantization", "int6", "int8", "int5", "gptq", "qat",
            "zstd", "zlib", "lzma", "compression", "artifact", "fp16",
            "mixed", "precision", "bitwidth",
        }
        optim_keywords = {
            "lr", "learning", "rate", "optimizer", "muon", "adamw", "adam",
            "momentum", "weight", "decay", "wd", "schedule", "warmup",
            "cooldown", "batch", "grad", "gradient",
        }

        def _classify(entry: RecordEntry) -> str:
            """Assign a mechanism category based on technique keywords."""
            text = (
                f"{entry.name} {entry.summary} {' '.join(entry.techniques)}"
            ).lower()
            words = set(re.findall(r"\w+", text))
            scores = {
                "architecture": len(words & arch_keywords),
                "quantization": len(words & quant_keywords),
                "optimizer": len(words & optim_keywords),
            }
            return max(scores, key=lambda k: scores[k])

        # Bucket entries by mechanism
        buckets: Dict[str, List[RecordEntry]] = {
            m: [] for m in _MECHANISMS
        }
        for entry in all_entries:
            mechanism = _classify(entry)
            buckets[mechanism].append(entry)

        routes: List[ProposedRoute] = []
        for mechanism in _MECHANISMS:
            bucket = buckets[mechanism]
            if not bucket:
                # Fall back: use best overall entries
                bucket = all_entries[:3]

            # Pick the top entry by relevance for this mechanism
            top = bucket[0]

            # Build cited records with val_bpb gaps
            cited: List[Dict[str, Any]] = []
            for entry in bucket[:3]:
                cited.append({
                    "name": entry.name,
                    "path": entry.path,
                    "val_bpb": entry.val_bpb,
                    "author": entry.author,
                    "techniques": entry.techniques[:3],
                })

            # Generate hypothesis from top entry's techniques
            tech_str = ", ".join(top.techniques[:2]) if top.techniques else top.name
            hypothesis = (
                f"Applying {tech_str} (from {top.name}, val_bpb={top.val_bpb:.4f}) "
                f"to {parent} may improve {bottleneck}"
            )

            # Generate env_overrides based on mechanism
            env_overrides: Dict[str, str] = {}
            if mechanism == "architecture":
                for tech in top.techniques:
                    tech_l = tech.lower()
                    if "layer" in tech_l:
                        m = re.search(r"(\d+)\s*(?:transformer\s+)?layer", tech_l)
                        if m:
                            env_overrides["NUM_LAYERS"] = m.group(1)
                    if "mlp" in tech_l:
                        m = re.search(r"mlp.*?(\d+)x", tech_l)
                        if m:
                            env_overrides["MLP_MULT"] = m.group(1)
            elif mechanism == "quantization":
                for tech in top.techniques:
                    tech_l = tech.lower()
                    if "qat" in tech_l:
                        env_overrides["QAT"] = "1"
                    if "zstd" in tech_l:
                        env_overrides["COMPRESSION"] = "zstd"
            elif mechanism == "optimizer":
                for tech in top.techniques:
                    tech_l = tech.lower()
                    if "lr" in tech_l or "learning rate" in tech_l:
                        m = re.search(r"[\d.]+", tech_l)
                        if m:
                            env_overrides["MUON_LR"] = m.group(0)
                    if "weight decay" in tech_l or "wd" in tech_l:
                        m = re.search(r"[\d.]+", tech_l)
                        if m:
                            env_overrides["WD"] = m.group(0)

            # Caveats
            caveats: List[str] = []
            if top.bytes_total > 15_500_000:
                caveats.append(
                    f"Source artifact is {top.bytes_total / 1e6:.1f}MB — "
                    "close to 16MB cap"
                )
            if "zstd" in top.summary.lower():
                caveats.append(
                    "Source uses zstd compression (we use zlib) — "
                    "artifact size may differ"
                )
            if "different tokenizer" in top.summary.lower():
                caveats.append("Source may use a different tokenizer")
            if not caveats:
                caveats.append("No major compatibility caveats identified")

            routes.append(
                ProposedRoute(
                    name=f"{mechanism}_{top.name.replace(' ', '_')[:30]}",
                    mechanism=mechanism,
                    hypothesis=hypothesis,
                    cited_records=cited,
                    env_overrides=env_overrides,
                    caveats=caveats,
                )
            )

        logger.info(
            "propose(%r, %r): generated %d routes",
            parent,
            bottleneck,
            len(routes),
        )
        return routes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> Dict[str, Any]:
        if self._cache_file.exists():
            try:
                return json.loads(self._cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Cache read error (%s) — starting fresh", exc)
        return {}

    def _save_cache(self) -> None:
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._cache_file.write_text(
                json.dumps(self._cache, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Cache write error: %s", exc)

    def _parse_arxiv_feed(self, xml_text: str) -> List[Dict[str, Any]]:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            logger.error("Failed to parse arxiv XML: %s", exc)
            return []

        results: List[Dict[str, Any]] = []
        for entry in root.findall("atom:entry", ns):
            arxiv_id_text = (entry.findtext("atom:id", default="", namespaces=ns) or "")
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
            authors = [
                (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
                for a in entry.findall("atom:author", ns)
            ]
            results.append(
                {
                    "id": arxiv_id_text.split("/abs/")[-1] if "/abs/" in arxiv_id_text else arxiv_id_text,
                    "title": title,
                    "authors": authors,
                    "summary": summary,
                    "published": published,
                    "url": arxiv_id_text,
                }
            )
        return results


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research retrieval CLI for autoresearch."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    search_p = sub.add_parser("search", help="Search local records/ directory.")
    search_p.add_argument("query", help="Search query string.")

    arxiv_p = sub.add_parser("arxiv", help="Search arxiv API.")
    arxiv_p.add_argument("query", help="Search query string.")
    arxiv_p.add_argument(
        "--max-results", type=int, default=10, help="Maximum results (default: 10)."
    )

    pr_p = sub.add_parser("pr", help="Fetch a GitHub PR.")
    pr_p.add_argument("number", type=int, help="PR number.")

    records_p = sub.add_parser(
        "records", help="Mine records/ for ideas relevant to a bottleneck."
    )
    records_p.add_argument(
        "--bottleneck",
        type=str,
        required=True,
        help="Description of the current research bottleneck.",
    )

    propose_p = sub.add_parser(
        "propose",
        help="Propose 3 experiment routes from records-mining.",
    )
    propose_p.add_argument(
        "--parent",
        type=str,
        required=True,
        help="Parent experiment ID.",
    )
    propose_p.add_argument(
        "--bottleneck",
        type=str,
        required=True,
        help="Description of the current research bottleneck.",
    )

    return parser


def main(argv: List[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    retriever = Retriever()

    if args.command == "search":
        results = retriever.search(args.query)
        for r in results:
            print(f"{r['path']}:{r['line']}: {r['text']}")

    elif args.command == "arxiv":
        results = retriever.arxiv(args.query, max_results=args.max_results)
        for r in results:
            print(f"[{r['id']}] {r['title']}")
            print(f"  Authors: {', '.join(r['authors'][:3])}")
            print(f"  Published: {r['published']}")
            print(f"  URL: {r['url']}")
            print()

    elif args.command == "pr":
        result = retriever.pr(args.number)
        if result is None:
            print(f"Failed to fetch PR #{args.number}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(result, indent=2))

    elif args.command == "records":
        entries = retriever.records(args.bottleneck)
        output = [
            {
                "path": e.path,
                "author": e.author,
                "name": e.name,
                "val_bpb": e.val_bpb,
                "bytes_total": e.bytes_total,
                "summary": e.summary,
                "techniques": e.techniques,
                "relevance_score": round(e.relevance_score, 3),
            }
            for e in entries
        ]
        print(json.dumps(output, indent=2))

    elif args.command == "propose":
        routes = retriever.propose(
            parent=args.parent,
            bottleneck=args.bottleneck,
        )
        output = [
            {
                "name": r.name,
                "mechanism": r.mechanism,
                "hypothesis": r.hypothesis,
                "cited_records": r.cited_records,
                "env_overrides": r.env_overrides,
                "caveats": r.caveats,
            }
            for r in routes
        ]
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
