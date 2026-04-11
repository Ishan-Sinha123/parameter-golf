"""Parallel Web Systems API client for deep web research.

Uses Parallel's Search API for quick web lookups and Task API (pro/ultra
processors) for deep, multi-step research. The autoresearcher constructs
detailed, multi-faceted queries so Parallel's deep research can return
comprehensive, citation-backed findings.

API docs: https://docs.parallel.ai
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

SEARCH_URL = "https://api.parallel.ai/v1beta/search"
TASK_CREATE_URL = "https://api.parallel.ai/v1/tasks/runs"
TASK_RESULT_URL = "https://api.parallel.ai/v1/tasks/runs/{run_id}/result"
TASK_STATUS_URL = "https://api.parallel.ai/v1/tasks/runs/{run_id}"


@dataclass
class SearchResult:
    """A single result from Parallel Search."""
    url: str
    title: str
    excerpts: list[str] = field(default_factory=list)
    publish_date: Optional[str] = None


@dataclass
class DeepResearchResult:
    """Result from a Parallel Task API deep research run."""
    run_id: str
    status: str  # queued, running, completed, failed
    processor: str
    output: Optional[str] = None  # text/markdown output
    content: Optional[dict] = None  # structured JSON output
    basis: list[dict] = field(default_factory=list)  # citations
    error: Optional[str] = None


class ParallelClient:
    """Client for the Parallel Web Systems API."""

    def __init__(self, api_key: str, default_processor: str = "pro"):
        self.api_key = api_key
        self.default_processor = default_processor

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _request(self, url: str, data: dict, timeout: int = 30) -> dict:
        """Make a POST request to the Parallel API."""
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            url, data=body, headers=self._headers(), method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def _get(self, url: str, timeout: int = 60) -> dict:
        """Make a GET request to the Parallel API."""
        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    # ── Search API ────────────────────────────────────────────────────

    def search(
        self,
        objective: str,
        queries: list[str],
        mode: str = "fast",
        max_chars_per_result: int = 10000,
    ) -> list[SearchResult]:
        """Run a web search via Parallel Search API.

        Args:
            objective: Natural language description of what you're looking for.
            queries: List of keyword search terms to run.
            mode: "fast" for quick results.
            max_chars_per_result: Max excerpt length per result.

        Returns:
            List of SearchResult with URLs, titles, and excerpts.
        """
        data = {
            "objective": objective,
            "search_queries": queries,
            "mode": mode,
            "excerpts": {"max_chars_per_result": max_chars_per_result},
        }

        try:
            resp = self._request(SEARCH_URL, data, timeout=30)
            results = []
            for item in resp.get("results", []):
                results.append(SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    excerpts=item.get("excerpts", []),
                    publish_date=item.get("publish_date"),
                ))
            log.info("Parallel search returned %d results for: %s",
                     len(results), objective[:80])
            return results
        except urllib.error.HTTPError as e:
            log.error("Parallel search HTTP error %d: %s", e.code,
                      e.read().decode()[:500])
            return []
        except Exception as e:
            log.error("Parallel search error: %s", e)
            return []

    # ── Task API (Deep Research) ──────────────────────────────────────

    def submit_deep_research(
        self,
        query: str,
        processor: Optional[str] = None,
        output_type: str = "text",
        json_schema: Optional[dict] = None,
    ) -> DeepResearchResult:
        """Fire-and-forget variant: creates the task, returns run_id, does
        not wait for the result. Use `fetch_deep_research(run_id)` later.
        """
        proc = processor or self.default_processor
        if output_type == "json" and json_schema:
            output_schema = {"type": "json", "json_schema": json_schema}
        elif output_type == "text":
            output_schema = {"type": "text"}
        else:
            output_schema = {"type": "auto"}
        data = {
            "input": query,
            "processor": proc,
            "task_spec": {"output_schema": output_schema},
        }
        try:
            create_resp = self._request(TASK_CREATE_URL, data, timeout=30)
            run_id = create_resp.get("run_id") or create_resp.get("id", "")
            log.info("Parallel deep research submitted: run_id=%s processor=%s",
                     run_id, proc)
            if not run_id:
                return DeepResearchResult(
                    run_id="", status="failed", processor=proc,
                    error=f"No run_id in response: {create_resp}",
                )
            return DeepResearchResult(
                run_id=run_id, status="submitted", processor=proc,
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:500]
            log.error("Parallel submit HTTP error %d: %s", e.code, body)
            return DeepResearchResult(
                run_id="", status="failed", processor=proc,
                error=f"HTTP {e.code}: {body}",
            )
        except Exception as e:
            log.error("Parallel submit error: %s", e)
            return DeepResearchResult(
                run_id="", status="failed", processor=proc,
                error=str(e),
            )

    def fetch_deep_research(self, run_id: str, processor: str = "pro") -> DeepResearchResult:
        """Poll /status once and, if finished, fetch /result.

        Returns a result whose `status` is one of: submitted | running |
        queued | completed | failed | error. Caller is responsible for
        retrying while status is still pending.
        """
        status_url = TASK_STATUS_URL.format(run_id=run_id)
        result_url = TASK_RESULT_URL.format(run_id=run_id)
        try:
            status_resp = self._get(status_url, timeout=30)
        except Exception as e:
            return DeepResearchResult(
                run_id=run_id, status="running", processor=processor,
                error=f"status poll error: {e}",
            )
        status = (status_resp.get("status") or "").lower()
        if status in ("completed", "done", "succeeded", "success"):
            try:
                result_resp = self._get(result_url, timeout=60)
            except Exception as e:
                return DeepResearchResult(
                    run_id=run_id, status="failed", processor=processor,
                    error=f"result fetch failed: {e}",
                )
            return self._parse_task_result(run_id, processor, result_resp)
        if status in ("failed", "error", "cancelled", "canceled"):
            return DeepResearchResult(
                run_id=run_id, status="failed", processor=processor,
                error=status_resp.get("error") or f"status={status}",
            )
        # queued / running — still pending
        return DeepResearchResult(
            run_id=run_id, status=status or "running", processor=processor,
        )

    def deep_research(
        self,
        query: str,
        processor: Optional[str] = None,
        output_type: str = "text",
        json_schema: Optional[dict] = None,
        poll_interval: float = 5.0,
        max_wait: float = 600.0,
    ) -> DeepResearchResult:
        """Run a deep research task via Parallel Task API.

        Parallel's deep research performs multi-step web exploration across
        authoritative sources. Construct detailed, specific queries for best
        results — Parallel handles the actual research orchestration.

        Args:
            query: Detailed research query (up to 15k chars). Be specific
                   about what you want to learn, what context matters, and
                   what format the answer should take.
            processor: Tier — "pro" ($0.10/req, exploratory), "ultra" ($0.30/req,
                       extensive), "pro-fast" or "ultra-fast" for 2-5x speed.
            output_type: "text" for markdown report, "auto" for structured JSON,
                         "json" for custom schema.
            json_schema: Required if output_type="json". JSON schema for output.
            poll_interval: Seconds between status checks.
            max_wait: Max seconds to wait for completion.

        Returns:
            DeepResearchResult with findings, citations, and confidence levels.
        """
        proc = processor or self.default_processor

        # Build task spec
        if output_type == "json" and json_schema:
            output_schema = {"type": "json", "json_schema": json_schema}
        elif output_type == "text":
            output_schema = {"type": "text"}
        else:
            output_schema = {"type": "auto"}

        data = {
            "input": query,
            "processor": proc,
            "task_spec": {"output_schema": output_schema},
        }

        try:
            # Create the task run
            create_resp = self._request(TASK_CREATE_URL, data, timeout=30)
            run_id = create_resp.get("run_id") or create_resp.get("id", "")
            log.info("Parallel deep research started: run_id=%s processor=%s",
                     run_id, proc)

            if not run_id:
                return DeepResearchResult(
                    run_id="", status="failed", processor=proc,
                    error=f"No run_id in response: {create_resp}",
                )

            # Poll the /status endpoint (non-blocking) until the task is
            # finished, then fetch /result. Earlier versions of this client
            # GET'd /result with a huge timeout which turned it into a
            # 10-minute blocking call and starved the research loop.
            status_url = TASK_STATUS_URL.format(run_id=run_id)
            result_url = TASK_RESULT_URL.format(run_id=run_id)
            start = time.monotonic()

            while time.monotonic() - start < max_wait:
                try:
                    status_resp = self._get(status_url, timeout=30)
                except Exception as e:
                    log.debug("Deep research %s status poll error: %s",
                              run_id, e)
                    time.sleep(poll_interval)
                    continue

                status = (status_resp.get("status") or "").lower()
                if status in ("completed", "done", "succeeded", "success"):
                    try:
                        result_resp = self._get(result_url, timeout=60)
                    except Exception as e:
                        return DeepResearchResult(
                            run_id=run_id, status="failed", processor=proc,
                            error=f"result fetch failed: {e}",
                        )
                    return self._parse_task_result(run_id, proc, result_resp)
                if status in ("failed", "error", "cancelled", "canceled"):
                    return DeepResearchResult(
                        run_id=run_id, status="failed", processor=proc,
                        error=status_resp.get("error") or f"status={status}",
                    )
                # queued / running — keep polling
                time.sleep(poll_interval)

            return DeepResearchResult(
                run_id=run_id, status="timeout", processor=proc,
                error=f"Timed out after {max_wait}s",
            )

        except urllib.error.HTTPError as e:
            body = e.read().decode()[:500]
            log.error("Parallel task HTTP error %d: %s", e.code, body)
            return DeepResearchResult(
                run_id="", status="failed", processor=proc,
                error=f"HTTP {e.code}: {body}",
            )
        except Exception as e:
            log.error("Parallel task error: %s", e)
            return DeepResearchResult(
                run_id="", status="failed", processor=proc,
                error=str(e),
            )

    def _parse_task_result(self, run_id: str, processor: str,
                           resp: dict) -> DeepResearchResult:
        """Parse a completed task result response."""
        status = resp.get("status", "completed")
        output = resp.get("output")
        content = resp.get("content")
        basis = resp.get("basis", [])

        # output can be a string (text mode) or dict (auto/json mode).
        # When dict, Parallel nests the text in "content"/"text"/"markdown"
        # and may also put citations under "basis".
        output_str: Optional[str] = None
        content_dict: Optional[dict] = None
        if isinstance(output, str):
            output_str = output
        elif isinstance(output, dict):
            content_dict = output
            for key in ("content", "text", "markdown", "output", "answer", "summary"):
                v = output.get(key)
                if isinstance(v, str) and v.strip():
                    output_str = v
                    break
            if output_str is None:
                # Last resort: serialize the dict so downstream code still
                # has *something* to store rather than dropping silently.
                try:
                    output_str = json.dumps(output, indent=2)[:20000]
                except Exception:
                    pass
            inner_basis = output.get("basis")
            if isinstance(inner_basis, list) and not basis:
                basis = inner_basis
        if isinstance(content, dict):
            content_dict = content
            if output_str is None:
                for key in ("content", "text", "markdown", "output", "answer", "summary"):
                    v = content.get(key)
                    if isinstance(v, str) and v.strip():
                        output_str = v
                        break
        elif isinstance(content, str) and output_str is None:
            output_str = content

        if output_str is None and not content_dict:
            log.warning("Parallel result %s: no output/content found. keys=%s",
                        run_id, list(resp.keys()))

        return DeepResearchResult(
            run_id=run_id,
            status=status,
            processor=processor,
            output=output_str,
            content=content_dict,
            basis=basis if isinstance(basis, list) else [],
        )

    # ── Convenience: research with search context ─────────────────────

    def search_then_research(
        self,
        topic: str,
        search_queries: list[str],
        research_query: str,
        processor: Optional[str] = None,
    ) -> DeepResearchResult:
        """Two-phase research: quick search for context, then deep research.

        1. Run Search API to find relevant URLs and excerpts
        2. Include search findings as context in a deep research Task

        This gives the deep research grounding from fresh web results.
        """
        # Phase 1: Search
        results = self.search(
            objective=topic,
            queries=search_queries,
            mode="fast",
        )

        # Build context from search results
        context_parts = []
        for r in results[:8]:  # Top 8 results
            excerpt_text = "\n".join(r.excerpts[:2])[:2000]
            context_parts.append(
                f"Source: {r.title} ({r.url})\n{excerpt_text}"
            )
        search_context = "\n\n---\n\n".join(context_parts)

        # Phase 2: Deep research with search context
        augmented_query = (
            f"{research_query}\n\n"
            f"--- Preliminary search findings for additional context ---\n\n"
            f"{search_context[:8000]}"
        )

        return self.deep_research(
            query=augmented_query,
            processor=processor,
        )
