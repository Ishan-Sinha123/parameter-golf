"""Research agent: polls for new PRs/papers, evaluates them, proposes experiments.

From the handwritten notes:
1. Constantly poll for new PRs that claim to beat SOTA. Critically evaluate:
   - Does this violate any rules?
   - Does this scale well, or is it completely novel?
   - Are there interesting papers to review that could relate?
   - Posted findings/blogs that look exciting & could offer
     an interesting new research direction?
2. If so, propose new experiments to run & run them.
3. System should have ability for steering by me & it should be high
   priority with option to stop current experiment or deploy.

Integration with Parallel Web Systems API:
- Constructs detailed, multi-faceted queries for deep web research
- Parallel handles the actual research orchestration across the web
- Results stored in local knowledge base for future reference
- Knowledge base queried before proposing ideas to avoid duplication

PR evaluation pipeline:
- Fetches ALL open PRs (these are active proposals from competitors)
- Extracts techniques, reported metrics, and approach details
- Checks knowledge base: have we tried this? what were results?
- Uses Parallel deep research to investigate novel techniques
- Stores all evaluations in knowledge base for future queries
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import AutoResearchConfig
from ..db.models import IdeaSource, IdeaStatus, EventType
from ..db.registry import Registry
from ..db.knowledge import KnowledgeBase
from ..ideas.tracker import IdeaTracker
from ..tracing import tracer
from .parallel import ParallelClient, SearchResult, DeepResearchResult

log = logging.getLogger(__name__)


# ── Technique extraction ──────────────────────────────────────────────

KNOWN_TECHNIQUES = [
    "swiglu", "geglu", "rope", "alibi", "muon", "lion", "adam", "adamw",
    "sophia", "gptq", "awq", "int4", "int6", "int8", "fp8", "fp4",
    "quantization", "pruning", "sparsity", "distillation",
    "moe", "mixture of experts", "flash attention", "ring attention",
    "grouped query attention", "gqa", "multi-query attention", "mqa",
    "sliding window", "kv cache", "rotary embedding",
    "test-time training", "ttt", "test-time compute",
    "layer norm", "rms norm", "deep norm",
    "cosine schedule", "warmup", "weight decay",
    "gradient accumulation", "gradient checkpointing",
    "lora", "qlora", "knowledge distillation",
    "speculative decoding", "token merging",
    "zlib", "lzma", "entropy coding", "arithmetic coding",
    "byte pair encoding", "bpe", "unigram",
    "curriculum learning", "data mixing",
]


def extract_techniques(text: str) -> list[str]:
    """Extract known ML techniques mentioned in text."""
    text_lower = text.lower()
    found = []
    for tech in KNOWN_TECHNIQUES:
        if tech in text_lower:
            found.append(tech)
    return list(set(found))  # deduplicate


@dataclass
class PRInfo:
    number: int
    title: str
    author: str
    body: str
    url: str
    created_at: str
    state: str = "open"
    labels: list[str] = field(default_factory=list)
    val_bpb: Optional[float] = None
    artifact_mb: Optional[float] = None
    techniques: list[str] = field(default_factory=list)
    diff_summary: str = ""  # First 2000 chars of diff for analysis


@dataclass
class PaperInfo:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    published: str
    categories: list[str] = field(default_factory=list)


class ResearchAgent:
    """Polls external sources, evaluates proposals, runs deep web research."""

    def __init__(self, config: AutoResearchConfig, registry: Registry,
                 ideas: IdeaTracker, knowledge: KnowledgeBase,
                 claude=None):
        self.config = config
        self.registry = registry
        self.ideas = ideas
        self.kb = knowledge
        self.claude = claude  # optional ClaudeRunner for assess_pr etc.
        self._stop_event = threading.Event()
        self._seen_prs: set[int] = set()
        self._seen_papers: set[str] = set()

        # Mission statement — prepended to every Parallel query so deep
        # research is framed in terms of the Parameter Golf objective.
        self.mission = config.load_mission()
        if self.mission:
            log.info("Mission loaded from %s (%d chars)",
                     config.mission_file, len(self.mission))
        else:
            log.warning("No mission statement found at %s", config.mission_file)

        # Parallel client (None if no API key)
        self.parallel: Optional[ParallelClient] = None
        if config.parallel_api_key:
            self.parallel = ParallelClient(
                api_key=config.parallel_api_key,
                default_processor=config.parallel_default_processor,
            )
            log.info("Parallel Web Systems API enabled (processor=%s)",
                     config.parallel_default_processor)

        # Pending async deep-research runs awaiting completion.
        # Structure: run_id -> dict(query, tags, pr_number?, idea_id?,
        # topic, submitted_at, processor)
        self._pending_research: dict[str, dict] = {}
        self._pending_lock = threading.Lock()

    # ── Main Loop ──────────────────────────────────────────────────────

    def run(self):
        """Polling loop (blocking)."""
        log.info("Research agent starting, poll interval=%dm",
                 self.config.poll_interval_m)
        self._load_seen()

        # Background thread: finish pending async deep-research runs
        # every 60s (the main poll cycle runs every 30min, which is too
        # slow for 3-5min Parallel tasks).
        def _pending_loop():
            while not self._stop_event.is_set():
                try:
                    self._poll_pending_research()
                except Exception as e:
                    log.exception("pending research poll error: %s", e)
                self._stop_event.wait(60.0)
        t = threading.Thread(target=_pending_loop, daemon=True,
                             name="parallel-finisher")
        t.start()

        while not self._stop_event.is_set():
            try:
                self._poll_cycle()
            except Exception as e:
                log.exception("Research poll error: %s", e)
            self._stop_event.wait(self.config.poll_interval_m * 60)

    def stop(self):
        self._stop_event.set()

    def _poll_pending_research(self):
        """Check pending Parallel deep-research runs. For completed ones,
        fetch results, store in KB, and emit a completion span."""
        if not self.parallel:
            return
        with self._pending_lock:
            run_ids = list(self._pending_research.keys())
        if not run_ids:
            return
        log.info("Polling %d pending deep-research runs", len(run_ids))
        for run_id in run_ids:
            with self._pending_lock:
                meta = self._pending_research.get(run_id)
            if meta is None:
                continue
            # Give up after 30 minutes — these will just be lost.
            age = time.time() - meta["submitted_at"]
            try:
                result = self.parallel.fetch_deep_research(
                    run_id, processor=meta.get("processor", "pro"))
            except Exception as e:
                log.warning("poll deep_research %s error: %s", run_id, e)
                continue
            if result.status in ("completed", "done") and not result.output:
                # Parallel says done but we couldn't extract text. Drop
                # from pending to avoid an infinite loop and log loudly.
                log.warning("Deep research %s completed but output empty "
                            "(content=%s) — dropping", run_id,
                            "dict" if result.content else "none")
                tracer.record(
                    kind="deep_research",
                    name=f"{meta['topic'][:60]} [empty]",
                    entity=("pr", str(meta["pr_number"])) if meta.get("pr_number") else None,
                    started_at=meta["submitted_at"],
                    ended_at=time.time(),
                    status="error",
                    attrs={"run_id": run_id, "processor": meta.get("processor")},
                    error="completed with empty output",
                )
                with self._pending_lock:
                    self._pending_research.pop(run_id, None)
                continue
            if result.status in ("completed", "done") and result.output:
                self.kb.store_web_research(
                    run_id=run_id,
                    query=meta["query"],
                    findings=result.output,
                    citations=result.basis,
                    tags=meta.get("tags") or [],
                )
                self.registry.emit_event(
                    EventType.DEEP_RESEARCH, "research", run_id,
                    {"pr_number": meta.get("pr_number"),
                     "idea_id": meta.get("idea_id"),
                     "topic": meta["topic"],
                     "findings_length": len(result.output)},
                )
                # Emit a completed span back-dated to submit time
                tracer.record(
                    kind="deep_research",
                    name=f"{meta['topic'][:60]} [done]",
                    entity=("pr", str(meta["pr_number"])) if meta.get("pr_number") else None,
                    started_at=meta["submitted_at"],
                    ended_at=time.time(),
                    status="ok",
                    attrs={
                        "run_id": run_id,
                        "findings_chars": len(result.output),
                        "citations": len(result.basis or []),
                        "processor": meta.get("processor"),
                    },
                )
                log.info("Deep research %s completed: %d chars",
                         run_id, len(result.output))
                with self._pending_lock:
                    self._pending_research.pop(run_id, None)
            elif result.status == "failed":
                log.warning("Deep research %s failed: %s",
                            run_id, result.error)
                tracer.record(
                    kind="deep_research",
                    name=f"{meta['topic'][:60]} [failed]",
                    entity=("pr", str(meta["pr_number"])) if meta.get("pr_number") else None,
                    started_at=meta["submitted_at"],
                    ended_at=time.time(),
                    status="error",
                    attrs={"run_id": run_id, "processor": meta.get("processor")},
                    error=(result.error or "")[:500],
                )
                with self._pending_lock:
                    self._pending_research.pop(run_id, None)
            elif age > 1800:
                log.warning("Deep research %s abandoned after %ds",
                            run_id, int(age))
                tracer.record(
                    kind="deep_research",
                    name=f"{meta['topic'][:60]} [abandoned]",
                    entity=("pr", str(meta["pr_number"])) if meta.get("pr_number") else None,
                    started_at=meta["submitted_at"],
                    ended_at=time.time(),
                    status="timeout",
                    attrs={"run_id": run_id, "age_s": int(age)},
                    error="exceeded 30 minute max age",
                )
                with self._pending_lock:
                    self._pending_research.pop(run_id, None)
            # else: still queued/running, check again next cycle

    def _poll_cycle(self):
        """One polling cycle: PRs, papers, web research, records."""
        log.info("Research poll cycle starting")

        with tracer.span("poll_cycle", name="research_poll") as root:
            # 0. Finish any pending async deep-research runs from
            # previous cycles so findings land in the KB.
            with tracer.span("poll_pending_research",
                             name="parallel_async_finisher"):
                self._poll_pending_research()

            # 1. Poll ALL open GitHub PRs — evaluate each for novelty
            with tracer.span("fetch_prs", name="gh pr list"):
                prs = self._fetch_open_prs()
            new_prs = 0
            for pr in prs:
                if pr.number in self._seen_prs:
                    continue
                self._seen_prs.add(pr.number)
                new_prs += 1
                with tracer.span("pr_eval", name=f"PR#{pr.number}",
                                 entity=("pr", str(pr.number))) as s:
                    s.set("title", pr.title[:120])
                    s.set("author", pr.author)
                    self._evaluate_and_store_pr(pr)

            # 2. Poll arxiv papers
            with tracer.span("fetch_papers", name="arxiv"):
                papers = self._fetch_recent_papers()
            new_papers = 0
            for paper in papers:
                if paper.arxiv_id in self._seen_papers:
                    continue
                self._seen_papers.add(paper.arxiv_id)
                new_papers += 1
                with tracer.span("paper_eval", name=paper.title[:80],
                                 entity=("paper", paper.arxiv_id)) as s:
                    evaluation = self._evaluate_paper(paper)
                    s.set("worth", evaluation["worth_exploring"])
                    if evaluation["worth_exploring"]:
                        self._propose_idea_from_paper(paper, evaluation)
                self.registry.emit_event(
                    EventType.PAPER_FOUND, "research", paper.arxiv_id,
                    {"title": paper.title,
                     "worth": evaluation["worth_exploring"]},
                )

            # 3. Run periodic deep web research on promising topics
            if self.parallel:
                with tracer.span("web_research_cycle",
                                 name="proactive_web_research"):
                    self._run_web_research_cycle()

            # 4. Mine local records
            with tracer.span("mine_records", name="records/"):
                self._mine_records()

            # 5. Check recently-merged PRs that touch records/ and fork them
            try:
                with tracer.span("poll_merged_prs", name="merged-records"):
                    self._poll_merged_prs()
            except Exception as e:
                log.warning("_poll_merged_prs failed: %s", e)

            root.set("new_prs", new_prs)
            root.set("new_papers", new_papers)

        log.info("Research poll cycle done: %d new PRs, %d new papers",
                 new_prs, new_papers)

    # ── GitHub PR Polling (ALL open PRs) ──────────────────────────────

    def _fetch_open_prs(self) -> list[PRInfo]:
        """Fetch ALL open PRs — these are active proposals from competitors."""
        try:
            r = subprocess.run(
                ["gh", "pr", "list", "--repo", self.config.github_repo,
                 "--state", "open", "--limit", "100",
                 "--json", "number,title,author,body,url,createdAt,labels"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                log.warning("gh pr list failed: %s", r.stderr[:200])
                return []

            data = json.loads(r.stdout)
            prs = []
            for item in data:
                body = item.get("body", "")[:4000]
                pr = PRInfo(
                    number=item["number"],
                    title=item["title"],
                    author=item.get("author", {}).get("login", "unknown"),
                    body=body,
                    url=item["url"],
                    created_at=item.get("createdAt", ""),
                    state="open",
                    labels=[la.get("name", "") for la in item.get("labels", [])],
                )
                # Extract val_bpb from body
                bpb_match = re.search(r"val_bpb[:\s]+([\d.]+)", pr.body)
                if bpb_match:
                    pr.val_bpb = float(bpb_match.group(1))
                # Extract artifact size
                mb_match = re.search(r"(\d+\.?\d*)\s*MB", pr.body)
                if mb_match:
                    pr.artifact_mb = float(mb_match.group(1))
                # Extract techniques
                pr.techniques = extract_techniques(pr.title + " " + body)
                prs.append(pr)

            log.info("Fetched %d open PRs from %s", len(prs),
                     self.config.github_repo)
            return prs
        except Exception as e:
            log.warning("PR fetch error: %s", e)
            return []

    def _fetch_pr_diff_summary(self, pr_number: int) -> str:
        """Fetch the diff of a PR for deeper analysis."""
        try:
            r = subprocess.run(
                ["gh", "pr", "diff", str(pr_number),
                 "--repo", self.config.github_repo],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                return ""
            # Return first 3000 chars of diff (enough to see key changes)
            return r.stdout[:3000]
        except Exception:
            return ""

    def _evaluate_and_store_pr(self, pr: PRInfo):
        """Evaluate a PR for novelty and store the evaluation in knowledge base.

        Pipeline:
        1. Extract techniques from PR title + body
        2. Check knowledge base: have we tried these techniques?
        3. Check for rule violations
        4. Assess novelty and potential
        5. If novel + promising, do deep web research via Parallel
        6. Store evaluation in knowledge base
        7. Propose idea if worth exploring
        """
        body_lower = (pr.title + " " + pr.body).lower()

        # ── Step 1: Rule violation check ──
        violates = self._check_rule_violations(body_lower)
        if violates:
            evaluation_text = f"RULE VIOLATION: {violates}"
            self.kb.store_pr_evaluation(
                pr_number=pr.number, pr_title=pr.title, author=pr.author,
                evaluation=evaluation_text, techniques=pr.techniques,
                reported_bpb=pr.val_bpb, url=pr.url,
            )
            self.registry.emit_event(
                EventType.PR_EVALUATED, "research", str(pr.number),
                {"title": pr.title, "verdict": "rule_violation",
                 "detail": violates},
            )
            log.info("PR#%d: rule violation — %s", pr.number, violates)
            return

        # ── Step 2: Check knowledge base for prior experiments ──
        prior_knowledge = ""
        for tech in pr.techniques[:5]:
            prior = self.kb.get_experiment_context(tech, limit=3)
            if "No previous experiments" not in prior:
                prior_knowledge += prior + "\n"

        # Also check if we've already evaluated similar techniques
        web_knowledge = ""
        for tech in pr.techniques[:3]:
            web = self.kb.get_web_context(tech, limit=2)
            if "No web research" not in web:
                web_knowledge += web + "\n"

        # ── Step 3: Novelty assessment ──
        novelty_score, novelty_detail = self._assess_novelty(
            pr, prior_knowledge, web_knowledge)

        # ── Step 4: Deep research for novel + promising PRs ──
        web_findings = ""
        if novelty_score >= 2 and self.parallel:
            web_findings = self._deep_research_pr(pr)

        # ── Step 5: Build full evaluation ──
        evaluation_parts = [
            f"Techniques detected: {', '.join(pr.techniques) or 'none identified'}",
            f"Reported BPB: {pr.val_bpb or 'not reported'}",
            f"Artifact size: {pr.artifact_mb or 'not reported'} MB",
            f"Novelty score: {novelty_score}/5",
            f"Novelty assessment: {novelty_detail}",
        ]
        if prior_knowledge:
            evaluation_parts.append(f"\nPrior experiment context:\n{prior_knowledge}")
        if web_findings:
            evaluation_parts.append(f"\nWeb research findings:\n{web_findings[:2000]}")

        evaluation_text = "\n".join(evaluation_parts)

        # ── Step 6: Store in knowledge base ──
        self.kb.store_pr_evaluation(
            pr_number=pr.number, pr_title=pr.title, author=pr.author,
            evaluation=evaluation_text, techniques=pr.techniques,
            reported_bpb=pr.val_bpb, url=pr.url,
        )

        worth_exploring = novelty_score >= 3 or (
            pr.val_bpb is not None and pr.val_bpb < 1.15
        )

        self.registry.emit_event(
            EventType.PR_EVALUATED, "research", str(pr.number),
            {"title": pr.title, "author": pr.author,
             "techniques": pr.techniques,
             "novelty_score": novelty_score,
             "reported_bpb": pr.val_bpb,
             "worth_exploring": worth_exploring,
             "verdict": novelty_detail[:200]},
        )

        # ── Step 7: Propose idea if promising ──
        proposed_idea = None
        if worth_exploring:
            proposed_idea = self._propose_idea_from_pr(pr, {
                "worth_exploring": True,
                "novelty_score": novelty_score,
                "critique": novelty_detail,
                "web_findings": web_findings[:1000],
                "prior_knowledge": prior_knowledge[:500],
            })

        log.info("PR#%d evaluated: novelty=%d/5 worth=%s techniques=%s",
                 pr.number, novelty_score, worth_exploring,
                 pr.techniques[:5])

        # ── Step 8: Delegate structured assessment to Claude ──
        # Rule-based scoring is the first cut; the Claude assess_pr task
        # is the deeper read that returns a structured recommendation
        # (reproduce / stack_on_best / ignore / implement_clone). We fire
        # this async so poll cycles aren't blocked by Claude latency.
        # Pass the just-created idea id so the downstream compose step
        # attaches the experiment to it instead of creating a fresh one.
        if (self.claude is not None
                and self.config.claude_auto_assess_pr
                and worth_exploring):
            try:
                self._spawn_claude_pr_assessment(
                    pr,
                    existing_idea_id=proposed_idea.id if proposed_idea else None,
                )
            except Exception as e:
                log.warning("assess_pr spawn failed for PR#%d: %s",
                            pr.number, e)

    def _spawn_claude_pr_assessment(self, pr: PRInfo,
                                     existing_idea_id: str | None = None):
        """Fire assess_pr on a background thread + react to the result.

        If `existing_idea_id` is provided, the downstream compose step will
        attach the composed recipe + experiment to that idea instead of
        creating a fresh one (used by the orphan backfill path).
        """
        from ..claude import build_task

        def _on_complete(result):
            if not result.success or not result.parsed:
                log.warning("assess_pr PR#%d returned no parsed result: %s",
                            pr.number, result.error or "(none)")
                return
            self._apply_pr_assessment(
                pr, result.parsed, existing_idea_id=existing_idea_id)

        spec = build_task(
            "assess_pr",
            config=self.config, registry=self.registry,
            pr_number=pr.number, pr_url=pr.url, repo=self.config.github_repo,
        )
        t = self.claude.run_async(spec, on_complete=_on_complete)
        log.info("PR#%d: dispatched assess_pr task to Claude", pr.number)
        return t

    def _apply_pr_assessment(self, pr: PRInfo, assessment: dict,
                              existing_idea_id: str | None = None):
        """React to a Claude assess_pr result.

        Depending on the `recommendation` field, this either:
        - stack_on_best: compose a stacked recipe on the current best and
          auto-queue a screening experiment (if claude_auto_implement is
          off we still create the recipe but leave the experiment queued
          for a human to flip to screening)
        - reproduce: record the PR as an idea marked is_reproduction
        - ignore: just emit an event, no further action
        - implement_clone: open an implement_technique task (guarded)

        If `existing_idea_id` is passed, the compose path reuses that
        idea row rather than creating a new one.
        """
        recommendation = assessment.get("recommendation", "ignore")
        technique = assessment.get("technique", "") or pr.title
        env_overrides = assessment.get("env_overrides") or {}
        new_feature = (assessment.get("new_feature_name")
                        or f"pr{pr.number}_{technique[:20]}")
        novelty = assessment.get("novelty", "incremental")

        self.registry.emit_event(
            EventType.PR_EVALUATED, "research", str(pr.number),
            {"source": "claude_assess_pr", "recommendation": recommendation,
             "novelty": novelty, "composable": assessment.get("composable"),
             "technique": technique[:200]},
        )

        if recommendation in ("reproduce", "stack_on_best"):
            self._compose_and_queue_from_pr(
                pr=pr, assessment=assessment,
                new_feature=new_feature, env_overrides=env_overrides,
                is_reproduction=(recommendation == "reproduce"),
                existing_idea_id=existing_idea_id,
            )
        elif existing_idea_id and recommendation == "ignore":
            # Backfill: Claude said "ignore" — mark the orphan rejected so
            # it doesn't clutter the dashboard indefinitely.
            self.ideas.reject_idea(
                existing_idea_id,
                f"claude assess_pr: ignore — {assessment.get('notes','')[:200]}",
            )
        elif recommendation == "implement_clone":
            if not self.config.claude_auto_implement:
                log.info("PR#%d: implement_clone suggested but "
                         "claude_auto_implement=false, skipping", pr.number)
                return
            self._dispatch_implement_technique(pr, assessment)
        # else: ignore — the event is already logged

    def _compose_and_queue_from_pr(self, pr: PRInfo, assessment: dict,
                                    new_feature: str,
                                    env_overrides: dict,
                                    is_reproduction: bool,
                                    existing_idea_id: str | None = None):
        """Stack the PR's env overrides onto the current best recipe.

        When `existing_idea_id` is set we attach the new experiment to
        that idea row instead of creating a fresh one — used by the
        orphan-idea backfill path.
        """
        from ..db.recipes import RecipeStore
        from ..db.models import ExperimentStatus, ExperimentCategory
        store = RecipeStore(self.registry, self.config.abs_recipes_dir)
        base = store.current_best()
        if base is None:
            log.warning("PR#%d: no current_best_baseline, skipping compose",
                        pr.number)
            return

        recipe = store.compose(
            base=base,
            new_features=[new_feature],
            added_env_overrides=env_overrides,
            name=f"pr{pr.number}_{new_feature}"[:60],
            description=(
                f"From PR #{pr.number} ({pr.title[:80]}). "
                f"Recommendation: {assessment.get('recommendation')}, "
                f"novelty: {assessment.get('novelty')}. "
                + (assessment.get("notes", "")[:200])
            ),
        )
        log.info("PR#%d: composed recipe %s", pr.number, recipe.id)

        # Reuse the existing orphan idea if provided, otherwise create new.
        if existing_idea_id:
            idea = self.registry.get_idea(existing_idea_id)
            if idea is None:
                log.warning("PR#%d: existing_idea_id %s not found, creating new",
                            pr.number, existing_idea_id)
            else:
                self.ideas.approve_idea(
                    idea.id, "auto-approved by assess_pr (backfill)")
        if not existing_idea_id or self.registry.get_idea(existing_idea_id) is None:
            idea = self.ideas.create_idea(
                title=f"PR#{pr.number}: {pr.title[:80]}",
                hypothesis=(assessment.get("notes") or pr.title)[:500],
                source=IdeaSource.GITHUB_PR,
                source_ref=pr.url,
                priority=3,
                tags=(pr.techniques or []) + ["pr_ingestion"],
                notes=f"Auto-ingested from {pr.url}",
            )
            self.ideas.approve_idea(idea.id, "auto-approved by assess_pr")
        exp = self.ideas.create_experiment(
            idea_id=idea.id,
            name=f"PR#{pr.number} {new_feature}"[:60],
            env_overrides=recipe.env_overrides,
            category=ExperimentCategory.ARCHITECTURE,
            hypothesis=assessment.get("notes", "") or "",
            stages=["screen", "gate"],
            priority=3,
        )
        self.registry.set_experiment_recipe(exp.id, recipe.id)
        self.registry.set_experiment_source(
            exp.id, source_ref=f"PR#{pr.number}",
            is_reproduction=is_reproduction,
        )
        # Auto-queue only if the global auto-deploy flag is on
        if getattr(self.config, "claude_auto_implement", False):
            self.registry.update_experiment_status(
                exp.id, ExperimentStatus.QUEUED,
                reason="auto-queued from PR assessment",
            )
            log.info("PR#%d: queued experiment %s", pr.number, exp.id)
        else:
            log.info("PR#%d: experiment %s created (not auto-queued)",
                     pr.number, exp.id)

    def backfill_orphan_pr_ideas(self, limit: int = 100) -> int:
        """Re-dispatch Claude assess_pr for github_pr ideas with no experiments.

        Used to recover from a prior run where assess_pr tasks crashed or
        were interrupted before they could compose + queue experiments,
        leaving orphan ideas behind in the dashboard.

        Covers both PROPOSED (pre-approval) and APPROVED (idea was
        auto-approved on creation but the Claude callback never fired
        because the worker restarted mid-flight) — both states end up
        with zero experiments and a non-running pipeline.
        """
        from ..db.models import IdeaStatus, IdeaSource
        if self.claude is None:
            log.warning("backfill: no claude runner available")
            return 0

        orphans: list = []
        for status in (IdeaStatus.PROPOSED, IdeaStatus.APPROVED):
            orphans.extend(
                i for i in self.registry.list_ideas(status)
                if i.source == IdeaSource.GITHUB_PR
            )
        # Filter to those with zero experiments
        orphans = [
            i for i in orphans
            if not self.registry.list_experiments(idea_id=i.id)
        ]
        log.info("backfill: found %d orphan PR ideas", len(orphans))

        dispatched = 0
        threads = []
        for idea in orphans[:limit]:
            pr_num = self._extract_pr_number(idea)
            if pr_num is None:
                log.warning("backfill: could not extract PR# from %s (ref=%s)",
                            idea.id, idea.source_ref)
                continue
            pr = self._fetch_single_pr(pr_num)
            if pr is None:
                log.warning("backfill: gh pr view failed for PR#%d", pr_num)
                continue
            try:
                t = self._spawn_claude_pr_assessment(
                    pr, existing_idea_id=idea.id)
                if t is not None:
                    threads.append(t)
                dispatched += 1
            except Exception as e:
                log.warning("backfill: dispatch failed for PR#%d: %s",
                            pr_num, e)
        log.info("backfill: dispatched %d assess_pr tasks", dispatched)
        self._backfill_threads = threads  # expose for CLI wait loop
        return dispatched

    def _extract_pr_number(self, idea) -> int | None:
        """Pull PR number out of an idea's source_ref URL or id slug."""
        import re as _re
        if idea.source_ref:
            m = _re.search(r"/pull/(\d+)", idea.source_ref)
            if m:
                return int(m.group(1))
        m = _re.search(r"pr_(\d+)", idea.id or "")
        if m:
            return int(m.group(1))
        return None

    def _fetch_single_pr(self, pr_number: int) -> PRInfo | None:
        """Fetch a single PR by number (used by backfill)."""
        try:
            r = subprocess.run(
                ["gh", "pr", "view", str(pr_number),
                 "--repo", self.config.github_repo,
                 "--json", "number,title,author,body,url,createdAt,labels,state"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                return None
            item = json.loads(r.stdout)
            body = (item.get("body") or "")[:4000]
            pr = PRInfo(
                number=item["number"],
                title=item["title"],
                author=(item.get("author") or {}).get("login", "unknown"),
                body=body,
                url=item["url"],
                created_at=item.get("createdAt", ""),
                state=item.get("state", "open").lower(),
                labels=[la.get("name", "") for la in item.get("labels", [])],
            )
            bpb_match = re.search(r"val_bpb[:\s]+([\d.]+)", pr.body)
            if bpb_match:
                pr.val_bpb = float(bpb_match.group(1))
            mb_match = re.search(r"(\d+\.?\d*)\s*MB", pr.body)
            if mb_match:
                pr.artifact_mb = float(mb_match.group(1))
            pr.techniques = extract_techniques(pr.title + " " + body)
            return pr
        except Exception as e:
            log.warning("fetch_single_pr(%d) failed: %s", pr_number, e)
            return None

    def _dispatch_implement_technique(self, pr: PRInfo, assessment: dict):
        """Open an implement_technique task for a novel PR."""
        from ..claude import build_task
        slug = (assessment.get("new_feature_name")
                or f"pr{pr.number}").lower().replace(" ", "_")
        env_flag = f"USE_{slug.upper()}"
        spec = build_task(
            "implement_technique",
            config=self.config, registry=self.registry,
            technique_name=assessment.get("technique", pr.title),
            hypothesis=pr.title,
            research_summary=assessment.get("notes", ""),
            env_flag=env_flag,
        )
        self.claude.run_async(spec)
        log.info("PR#%d: dispatched implement_technique (flag=%s)",
                 pr.number, env_flag)

    def _check_rule_violations(self, text: str) -> str:
        """Check for competition rule violations. Returns violation description or empty string."""
        violations = []
        if all(k in text for k in ("validation", "training", "data")):
            if "leak" in text or "overfit" in text:
                violations.append("Possible data leakage (validation data in training)")
        if all(k in text for k in ("network", "eval", "download")):
            violations.append("Possible network access during evaluation")
        if "pretrained" in text and "weight" in text:
            if "distill" not in text:
                violations.append("Possible use of pretrained weights")
        return "; ".join(violations)

    def _assess_novelty(self, pr: PRInfo, prior_experiments: str,
                        web_knowledge: str) -> tuple[int, str]:
        """Assess how novel a PR's approach is. Returns (score 1-5, detail).

        1 = We've already tried this exact thing
        2 = Similar to something we've tried, marginal difference
        3 = Interesting variation we haven't fully explored
        4 = Novel technique with strong theoretical backing
        5 = Completely novel approach, very promising
        """
        score = 3  # Start neutral
        details = []

        # Check if techniques overlap with our experiments
        if prior_experiments and "No previous experiments" not in prior_experiments:
            # We've tried something related
            score -= 1
            details.append("We have prior experiments in this area")
            # But if their BPB is significantly better, bump it back up
            if pr.val_bpb and pr.val_bpb < 1.10:
                score += 2
                details.append(f"But their BPB={pr.val_bpb} is very strong")
            elif pr.val_bpb and pr.val_bpb < 1.15:
                score += 1
                details.append(f"Their BPB={pr.val_bpb} is competitive")
        else:
            # We haven't tried this area
            score += 1
            details.append("Novel area — no prior experiments found")

        # Check technique count and novelty
        if len(pr.techniques) >= 4:
            score = min(score + 1, 5)
            details.append(f"Rich approach with {len(pr.techniques)} techniques")

        # Check for specific high-value signals
        high_value = ["moe", "test-time training", "ttt", "curriculum learning",
                      "speculative decoding", "token merging", "flash attention"]
        hv_found = [t for t in pr.techniques if t in high_value]
        if hv_found:
            score = min(score + 1, 5)
            details.append(f"High-value techniques: {', '.join(hv_found)}")

        # Already-known results from web
        if web_knowledge and "No web research" not in web_knowledge:
            details.append("We have web research context on these techniques")

        score = max(1, min(5, score))
        return score, "; ".join(details)

    def _deep_research_pr(self, pr: PRInfo) -> str:
        """Use Parallel to do deep web research on a PR's techniques.

        Constructs a detailed query so Parallel can do thorough research.
        """
        if not self.parallel:
            return ""

        # Build a detailed research query
        techniques_str = ", ".join(pr.techniques[:8]) if pr.techniques else "general optimization"
        mission_preamble = (
            f"# Mission context\n{self.mission}\n\n# Research request\n"
            if self.mission else ""
        )
        query = (
            f"{mission_preamble}"
            f"A competitor (PR#{pr.number} by {pr.author}) proposes the following "
            f"approach. Evaluate it against the mission above.\n\n"
            f"Title: {pr.title}\n"
            f"Techniques used: {techniques_str}\n"
            f"Reported BPB: {pr.val_bpb or 'not reported'}\n"
            f"Summary: {pr.body[:1500]}\n\n"
            f"Please research the following:\n"
            f"1. What is the current state-of-the-art for these techniques "
            f"({techniques_str}) applied to small language models (46M-200M params)?\n"
            f"2. What papers or blog posts discuss optimal configurations for "
            f"these techniques? What hyperparameters work best?\n"
            f"3. Are there known failure modes or limitations when combining "
            f"these techniques for compression/BPB optimization?\n"
            f"4. What improvements could we make beyond what this PR proposes?\n"
            f"5. Are there recent (2025-2026) advances in any of these areas "
            f"that could give us an edge?\n\n"
            f"Focus on practical, actionable findings with specific numbers "
            f"and configurations where possible."
        )

        try:
            # Fire-and-forget: Parallel's pro tier takes 3-5 min per task,
            # so blocking PR eval on it starves the whole research loop
            # (80 PRs × 300s = ~7 hours/cycle). Submit the task, record a
            # pending span, and let _poll_pending_research() finish the
            # span + write to KB when the result arrives.
            with tracer.span("deep_research_submit",
                             name=f"PR#{pr.number} {techniques_str[:60]}",
                             entity=("pr", str(pr.number))) as s:
                s.set("processor",
                      self.config.parallel_default_processor)
                s.set("query_len", len(query))
                result = self.parallel.submit_deep_research(
                    query=query,
                    processor=self.config.parallel_default_processor,
                    output_type="text",
                )
                s.set("run_id", result.run_id)
                s.set("submit_status", result.status)
                if result.error:
                    s.set("error_msg", result.error[:200])

            if result.run_id and result.status == "submitted":
                with self._pending_lock:
                    self._pending_research[result.run_id] = {
                        "query": f"PR#{pr.number} techniques: {techniques_str}",
                        "tags": pr.techniques,
                        "pr_number": pr.number,
                        "idea_id": None,
                        "topic": techniques_str[:100],
                        "submitted_at": time.time(),
                        "processor": result.processor,
                    }
                log.info("Queued deep_research PR#%d run_id=%s",
                         pr.number, result.run_id)
            else:
                log.warning("Deep research submit PR#%d failed: %s",
                            pr.number, result.error)
            # Always return "" — the PR eval doesn't block on findings.
            # Later PR evals will pick up accumulated web_research context
            # from the KB automatically.
            return ""
        except Exception as e:
            log.warning("Deep research submit PR#%d exception: %s",
                        pr.number, e)
            return ""

    # ── Web Research Cycle ─────────────────────────────────────────────

    def _run_web_research_cycle(self):
        """Proactively research topics based on current experiment landscape.

        Looks at what we're working on and what's promising, then does
        targeted deep research to find improvements.
        """
        if not self.parallel:
            return

        # Get active ideas to understand current research directions
        active_ideas = self.registry.list_ideas(IdeaStatus.ACTIVE)
        approved_ideas = self.registry.list_ideas(IdeaStatus.APPROVED)
        all_ideas = active_ideas + approved_ideas

        if not all_ideas:
            # No current research — do broad survey
            self._broad_web_research()
            return

        # Research the most promising active direction
        for idea in sorted(all_ideas, key=lambda i: i.priority, reverse=True)[:2]:
            # Check if we already have recent web research on this
            existing = self.kb.search(idea.title, source_type="web_research", limit=1)
            if existing:
                continue  # Already researched recently

            self._targeted_web_research(idea)

    def _broad_web_research(self):
        """Broad survey: what's new in small LM training and compression?"""
        if not self.parallel:
            return

        mission_preamble = (
            f"# Mission context\n{self.mission}\n\n# Research request\n"
            if self.mission else ""
        )
        query = (
            f"{mission_preamble}"
            "What are the latest techniques (2025-2026) for training small "
            "language models (46M-200M parameters) to achieve the best possible "
            "bits-per-byte (BPB) score on text compression benchmarks?\n\n"
            "Given the mission above, research the following areas:\n"
            "1. Architecture innovations: What activation functions, attention "
            "mechanisms, normalization techniques work best at this scale?\n"
            "2. Training recipes: Optimal learning rates, schedulers, warmup "
            "periods, weight decay for small models?\n"
            "3. Quantization-aware training: How to train models that maintain "
            "quality after int6 quantization? What's the expected BPB gap?\n"
            "4. Data efficiency: Best practices for maximizing learning from "
            "limited training data for compression tasks?\n"
            "5. Any recent breakthroughs or surprising findings?\n\n"
            "Provide specific numbers, configurations, and paper references "
            "where available."
        )

        try:
            with tracer.span("deep_research",
                             name="broad_survey") as s:
                s.set("processor", "pro")
                result = self.parallel.deep_research(
                    query=query, processor="pro", output_type="text",
                )
                s.set("run_id", result.run_id)
                s.set("result_status", result.status)
            if result.output:
                self.kb.store_web_research(
                    run_id=result.run_id,
                    query="Broad survey: small LM training + compression 2025-2026",
                    findings=result.output,
                    citations=result.basis,
                    tags=["survey", "broad", "small-lm", "compression"],
                )
                self.registry.emit_event(
                    EventType.DEEP_RESEARCH, "research", result.run_id,
                    {"topic": "broad_survey", "processor": result.processor,
                     "findings_length": len(result.output)},
                )
                log.info("Broad web research completed: %d chars", len(result.output))

                # Check findings for ideas worth proposing
                self._extract_ideas_from_research(result.output, "broad_survey")
        except Exception as e:
            log.warning("Broad web research failed: %s", e)

    def _targeted_web_research(self, idea):
        """Deep research on a specific idea's techniques."""
        if not self.parallel:
            return

        # Get experiment context
        exp_context = self.kb.get_experiment_context(idea.title, limit=5)

        mission_preamble = (
            f"# Mission context\n{self.mission}\n\n# Research request\n"
            if self.mission else ""
        )
        query = (
            f"{mission_preamble}"
            f"Deep research on: {idea.title}\n\n"
            f"Hypothesis: {idea.hypothesis}\n\n"
            f"Our current experiment results in this area:\n{exp_context}\n\n"
            f"Please research:\n"
            f"1. What specific configurations and hyperparameters have been shown "
            f"to work best for this technique at our model scale?\n"
            f"2. What are common failure modes and how to avoid them?\n"
            f"3. Are there recent papers or implementations that push the "
            f"state-of-the-art for this approach?\n"
            f"4. What complementary techniques could amplify the effect?\n"
            f"5. What specific next experiments should we try based on the "
            f"latest findings?\n\n"
            f"Be specific with numbers, configurations, and citations."
        )

        try:
            with tracer.span("deep_research",
                             name=f"targeted:{idea.title[:60]}",
                             entity=("idea", idea.id)) as s:
                s.set("processor",
                      self.config.parallel_default_processor)
                result = self.parallel.deep_research(
                    query=query,
                    processor=self.config.parallel_default_processor,
                    output_type="text",
                )
                s.set("run_id", result.run_id)
                s.set("result_status", result.status)
            if result.output:
                tags = idea.tags if hasattr(idea, 'tags') else []
                self.kb.store_web_research(
                    run_id=result.run_id,
                    query=f"Targeted: {idea.title}",
                    findings=result.output,
                    citations=result.basis,
                    tags=tags,
                )
                self.registry.emit_event(
                    EventType.DEEP_RESEARCH, "research", result.run_id,
                    {"idea_id": idea.id, "topic": idea.title[:100],
                     "processor": result.processor,
                     "findings_length": len(result.output)},
                )
                log.info("Targeted research for '%s': %d chars",
                         idea.title[:40], len(result.output))
        except Exception as e:
            log.warning("Targeted research for '%s' failed: %s",
                        idea.title[:40], e)

    def _extract_ideas_from_research(self, findings: str, source_ref: str):
        """Extract actionable experiment ideas from web research findings."""
        # Look for technique mentions that we haven't tried
        techniques = extract_techniques(findings)
        for tech in techniques[:5]:
            # Check if we already have experiments or ideas for this
            existing_exps = self.kb.search(tech, source_type="experiment", limit=1)
            existing_ideas = self.registry.list_ideas()
            already_proposed = any(
                tech.lower() in (i.title + " " + i.hypothesis).lower()
                for i in existing_ideas
            )
            if not existing_exps and not already_proposed:
                # New technique from web research — propose it
                self.ideas.create_idea(
                    title=f"Web research: {tech.title()}",
                    hypothesis=f"Web research suggests {tech} may improve BPB. "
                               f"Found in recent literature survey.",
                    source=IdeaSource.WEB_RESEARCH,
                    source_ref=source_ref,
                    priority=2,
                    tags=["from-web-research", tech, "needs-evaluation"],
                    notes=f"Extracted from web research findings.\n\n"
                          f"Context: {findings[:500]}",
                )

    # ── Parallel Search (Quick) ────────────────────────────────────────

    def web_search(self, query: str, search_terms: Optional[list[str]] = None
                   ) -> list[SearchResult]:
        """Run a quick Parallel Search. Returns search results."""
        if not self.parallel:
            return []

        terms = search_terms or [query]
        results = self.parallel.search(
            objective=query,
            queries=terms,
            mode="fast",
        )
        self.registry.emit_event(
            EventType.WEB_SEARCH, "research", "",
            {"query": query, "result_count": len(results)},
        )
        return results

    def deep_research(self, query: str,
                      processor: Optional[str] = None) -> DeepResearchResult:
        """Run a deep research query via Parallel Task API.

        Construct detailed, specific queries — Parallel will do the heavy
        lifting of multi-step web exploration.
        """
        if not self.parallel:
            return DeepResearchResult(
                run_id="", status="disabled", processor="none",
                error="Parallel API not configured",
            )

        result = self.parallel.deep_research(
            query=query,
            processor=processor or self.config.parallel_default_processor,
            output_type="text",
        )

        # Store findings in knowledge base
        if result.output:
            self.kb.store_web_research(
                run_id=result.run_id,
                query=query[:200],
                findings=result.output,
                citations=result.basis,
                tags=extract_techniques(query),
            )

        return result

    # ── Arxiv Polling ──────────────────────────────────────────────────

    def _fetch_recent_papers(self) -> list[PaperInfo]:
        """Fetch recent arxiv papers via API."""
        import urllib.request
        import urllib.parse
        import xml.etree.ElementTree as ET

        query_terms = [
            "language model quantization",
            "efficient transformer training",
            "test-time training",
            "model compression",
        ]
        papers = []

        for query in query_terms[:2]:  # Limit to avoid rate limits
            try:
                params = urllib.parse.urlencode({
                    "search_query": f"all:{query}",
                    "start": 0, "max_results": 5,
                    "sortBy": "submittedDate", "sortOrder": "descending",
                })
                url = f"http://export.arxiv.org/api/query?{params}"
                req = urllib.request.Request(
                    url, headers={"User-Agent": "parameter-golf-research/2.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    xml_data = resp.read().decode()

                root = ET.fromstring(xml_data)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns):
                    arxiv_id = entry.find("atom:id", ns).text.split("/")[-1]
                    papers.append(PaperInfo(
                        arxiv_id=arxiv_id,
                        title=entry.find("atom:title", ns).text.strip(),
                        authors=[a.find("atom:name", ns).text
                                 for a in entry.findall("atom:author", ns)],
                        abstract=entry.find("atom:summary", ns).text.strip()[:1000],
                        url=entry.find("atom:id", ns).text,
                        published=entry.find("atom:published", ns).text,
                        categories=[c.get("term")
                                    for c in entry.findall(
                                        "{http://arxiv.org/schemas/atom}category")],
                    ))
                time.sleep(3)  # Be polite to arxiv
            except Exception as e:
                log.warning("Arxiv fetch error for '%s': %s", query, e)

        return papers

    def _evaluate_paper(self, paper: PaperInfo) -> dict:
        """Evaluate if a paper is relevant to our work."""
        evaluation = {"worth_exploring": False, "critique": ""}

        text = (paper.title + " " + paper.abstract).lower()
        relevance_keywords = [
            "quantization", "int4", "int6", "int8", "gptq",
            "small language model", "efficient training",
            "test-time", "knowledge distillation",
            "model compression", "pruning",
            "bpb", "bits per byte", "perplexity",
        ]
        relevance = sum(1 for kw in relevance_keywords if kw in text)

        if relevance >= 2:
            evaluation["worth_exploring"] = True
            evaluation["critique"] = (
                f"Relevant to parameter-golf: {relevance} keyword matches. "
                f"Abstract mentions: {paper.abstract[:200]}"
            )

        return evaluation

    # ── Idea Proposals ─────────────────────────────────────────────────

    def _propose_idea_from_pr(self, pr: PRInfo, evaluation: dict):
        """Create an idea from a promising PR, enriched with web research.

        Returns the created Idea so the caller can wire its id into the
        downstream Claude assess_pr task — otherwise the task creates a
        fresh idea and the original one stays orphaned at PROPOSED.
        """
        notes_parts = [
            f"val_bpb={pr.val_bpb}",
            f"Techniques: {', '.join(pr.techniques)}",
            f"Novelty: {evaluation.get('novelty_score', '?')}/5",
        ]
        if evaluation.get("prior_knowledge"):
            notes_parts.append(f"\nPrior experiments:\n{evaluation['prior_knowledge']}")
        if evaluation.get("web_findings"):
            notes_parts.append(f"\nWeb research:\n{evaluation['web_findings']}")
        notes_parts.append(f"\nPR body:\n{pr.body[:500]}")

        return self.ideas.create_idea(
            title=f"PR#{pr.number}: {pr.title[:50]}",
            hypothesis=(
                f"Techniques from PR#{pr.number} by {pr.author} may improve our BPB. "
                f"{evaluation.get('critique', '')}"
            ),
            source=IdeaSource.GITHUB_PR,
            source_ref=pr.url,
            priority=3 if pr.val_bpb and pr.val_bpb < 1.10 else 2,
            tags=["from-pr", "needs-evaluation"] + pr.techniques[:5],
            notes="\n".join(notes_parts),
        )

    def _propose_idea_from_paper(self, paper: PaperInfo, evaluation: dict):
        """Create an idea from a promising paper."""
        self.ideas.create_idea(
            title=f"Paper: {paper.title[:50]}",
            hypothesis=f"Methods from {paper.title} may be applicable. "
                        f"{evaluation['critique']}",
            source=IdeaSource.PAPER,
            source_ref=paper.url,
            priority=2,
            tags=["from-paper", "needs-evaluation"],
            notes=f"Authors: {', '.join(paper.authors[:3])}\n\n"
                  f"{paper.abstract[:500]}",
        )

    # ── Merged PR Polling ──────────────────────────────────────────────

    def _poll_merged_prs(self):
        """Scan recently-merged PRs for new records/ submissions and fork them.

        This is the remote counterpart to `_mine_records`: when a SOTA PR
        lands upstream but we haven't pulled it locally yet, `gh pr list
        --state merged` surfaces it and we `git fetch` to make the record
        files available before running sota_fork.sync_from_records.
        """
        try:
            r = subprocess.run(
                ["gh", "pr", "list", "--repo", self.config.github_repo,
                 "--state", "merged", "--limit", "20",
                 "--search", "records/track_10min_16mb in:path",
                 "--json", "number,title,mergedAt,files"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0:
                return
            merged = json.loads(r.stdout or "[]")
        except Exception as e:
            log.debug("gh pr list --merged failed: %s", e)
            return

        seen_key = "_seen_merged_prs"
        if not hasattr(self, seen_key):
            setattr(self, seen_key, set())
        seen: set[int] = getattr(self, seen_key)

        touches_records = False
        for item in merged:
            num = item.get("number")
            if num in seen:
                continue
            seen.add(num)
            files = item.get("files") or []
            if any("records/track_10min_16mb" in (f.get("path") or "")
                   for f in files):
                touches_records = True
                log.info("Merged PR#%s touches records/ — will re-sync baselines",
                         num)

        if not touches_records:
            return

        # Fetch + pull main so local records/ includes the newly merged files.
        repo_root = Path(self.config.workspace_dir).parent
        try:
            subprocess.run(
                ["git", "fetch", "origin", "main"],
                cwd=repo_root, capture_output=True, timeout=30,
            )
            # Non-destructive: update records/ tree from main without
            # touching the current branch.
            subprocess.run(
                ["git", "checkout", "origin/main", "--", "records/"],
                cwd=repo_root, capture_output=True, timeout=15,
            )
        except Exception as e:
            log.warning("git fetch/checkout records/ failed: %s", e)

        # Now re-run the local records sync — which will pick up anything new.
        self._mine_records()

    # ── Local Records Mining ───────────────────────────────────────────

    def _mine_records(self):
        """Scan local records/ for ideas (from parameter-golf repo)."""
        records_dir = Path(self.config.workspace_dir).parent / "records"
        if not records_dir.exists():
            return

        # Fork any SOTA records that beat our current_best_baseline pointer.
        # This is idempotent — existing recipes dedupe by feature hash.
        try:
            from ..db.recipes import RecipeStore
            from . import sota_fork
            store = RecipeStore(self.registry, self.config.abs_recipes_dir)
            repo_root = Path(self.config.workspace_dir).parent
            sota_fork.sync_from_records(
                records_dir / "track_10min_16mb",
                recipes_store=store,
                repo_root=repo_root,
            )
        except Exception as e:
            log.warning("sota_fork.sync_from_records failed: %s", e)

        for track_dir in records_dir.iterdir():
            if not track_dir.is_dir():
                continue
            for submission_dir in track_dir.iterdir():
                if not submission_dir.is_dir():
                    continue
                submission_json = submission_dir / "submission.json"
                if not submission_json.exists():
                    continue

                try:
                    with open(submission_json) as f:
                        sub = json.loads(f.read())
                    val_bpb = sub.get("val_bpb")
                    if val_bpb and val_bpb < 1.15:
                        source_ref = f"records/{track_dir.name}/{submission_dir.name}"
                        if source_ref not in self._seen_papers:
                            self._seen_papers.add(source_ref)
                            self.registry.add_sota_entry(
                                source="records", source_ref=source_ref,
                                val_bpb=val_bpb,
                                artifact_mb=sub.get("bytes_total", 0) / 1e6,
                                technique=submission_dir.name,
                            )
                            if (self.claude is not None
                                    and getattr(self.config,
                                                "claude_auto_reproduce", False)):
                                try:
                                    self._spawn_reproduce_record(
                                        submission_dir, source_ref, sub,
                                    )
                                except Exception as e:
                                    log.warning(
                                        "reproduce_record spawn failed for %s: %s",
                                        source_ref, e,
                                    )
                except Exception:
                    pass

    def _spawn_reproduce_record(self, submission_dir: Path,
                                 source_ref: str, submission: dict):
        """Fire reproduce_record on a background thread + react to the result."""
        from ..claude import build_task

        def _on_complete(result):
            if not result.success or not result.parsed:
                log.warning("reproduce_record %s returned no parsed result: %s",
                            source_ref, result.error or "(none)")
                return
            self._apply_reproduce_record(source_ref, submission, result.parsed)

        spec = build_task(
            "reproduce_record",
            config=self.config, registry=self.registry,
            record_path=str(submission_dir),
        )
        self.claude.run_async(spec, on_complete=_on_complete)
        log.info("record %s: dispatched reproduce_record task to Claude",
                 source_ref)

    def _apply_reproduce_record(self, source_ref: str, submission: dict,
                                  result: dict):
        """React to a Claude reproduce_record result.

        Composes a recipe from the proposal and creates a reproduction
        experiment (marked is_reproduction=True).
        """
        from ..db.recipes import RecipeStore
        from ..db.models import ExperimentStatus, ExperimentCategory

        proposal = result.get("recipe_proposal") or {}
        confidence = result.get("confidence", "unknown")

        self.registry.emit_event(
            EventType.PAPER_FOUND, "research", source_ref,
            {"source": "claude_reproduce_record",
             "confidence": confidence,
             "claimed_bpb": result.get("claimed_bpb"),
             "notes": (result.get("notes") or "")[:500]},
        )

        if confidence == "unreproducible" or not proposal:
            log.info("record %s: reproduce_record returned unreproducible",
                     source_ref)
            return

        store = RecipeStore(self.registry, self.config.abs_recipes_dir)
        base = store.current_best()
        if base is None:
            log.warning("record %s: no current_best_baseline, skipping compose",
                        source_ref)
            return

        features = proposal.get("features") or []
        env_overrides = proposal.get("env_overrides") or {}
        name = (proposal.get("name")
                or f"repro_{submission.get('name', source_ref)[:40]}")
        description = (
            f"Reproduction of {source_ref}. "
            f"Confidence: {confidence}. "
            f"Claimed bpb: {result.get('claimed_bpb')}. "
            + (proposal.get("description", "")[:200])
        )

        recipe = store.compose(
            base=base,
            new_features=features,
            added_env_overrides=env_overrides,
            name=name[:60],
            description=description,
        )
        log.info("record %s: composed recipe %s", source_ref, recipe.id)

        idea = self.ideas.create_idea(
            title=f"Repro: {source_ref}"[:80],
            hypothesis=(result.get("notes") or source_ref)[:500],
            source=IdeaSource.RECORD_MINING,
            source_ref=source_ref,
            priority=2,
            tags=["records", "reproduction"],
            notes=f"Auto-ingested from {source_ref}",
        )
        self.ideas.approve_idea(idea.id, "auto-approved by reproduce_record")
        exp = self.ideas.create_experiment(
            idea_id=idea.id,
            name=name[:60],
            env_overrides=recipe.env_overrides,
            category=ExperimentCategory.ARCHITECTURE,
            hypothesis=(result.get("notes") or "")[:500],
            stages=["screen", "gate"],
            priority=2,
        )
        self.registry.set_experiment_recipe(exp.id, recipe.id)
        self.registry.set_experiment_source(
            exp.id, source_ref=source_ref, is_reproduction=True,
        )
        if getattr(self.config, "claude_auto_implement", False):
            self.registry.update_experiment_status(
                exp.id, ExperimentStatus.QUEUED,
                reason="auto-queued from reproduce_record",
            )
            log.info("record %s: queued reproduction experiment %s",
                     source_ref, exp.id)
        else:
            log.info("record %s: reproduction experiment %s created "
                     "(not auto-queued)", source_ref, exp.id)

    def _load_seen(self):
        """Load already-processed items from events table."""
        events = self.registry.list_events(entity_type="research", limit=1000)
        for ev in events:
            if ev["event_type"] in ("pr_found", "pr_evaluated"):
                try:
                    self._seen_prs.add(int(ev["entity_id"]))
                except (ValueError, TypeError):
                    pass
            elif ev["event_type"] == "paper_found":
                self._seen_papers.add(ev["entity_id"])

    # ── Manual Triggers ────────────────────────────────────────────────

    def poll_now(self) -> dict:
        """Trigger an immediate poll cycle. Returns summary."""
        prs = self._fetch_open_prs()
        papers = self._fetch_recent_papers()
        new_prs = [p for p in prs if p.number not in self._seen_prs]
        new_papers = [p for p in papers if p.arxiv_id not in self._seen_papers]
        return {
            "total_prs": len(prs),
            "new_prs": len(new_prs),
            "total_papers": len(papers),
            "new_papers": len(new_papers),
            "pr_titles": [p.title for p in new_prs[:10]],
            "paper_titles": [p.title for p in new_papers[:10]],
            "parallel_enabled": self.parallel is not None,
            "knowledge_entries": self.kb.count(),
        }

    def get_knowledge_summary(self) -> dict:
        """Get knowledge base stats and recent entries."""
        return {
            "total_entries": self.kb.count(),
            "experiments": self.kb.count("experiment"),
            "web_research": self.kb.count("web_research"),
            "pr_evaluations": self.kb.count("pr_evaluation"),
            "papers": self.kb.count("paper"),
            "recent": [
                {"id": e.id, "type": e.source_type, "title": e.title,
                 "created_at": e.created_at}
                for e in self.kb.recent(limit=10)
            ],
        }

    def query_knowledge(self, query: str,
                        source_type: Optional[str] = None) -> list[dict]:
        """Query the knowledge base. Returns matching entries."""
        entries = self.kb.search(query, source_type=source_type, limit=20)
        return [
            {"id": e.id, "type": e.source_type, "title": e.title,
             "content": e.content[:500], "tags": e.tags,
             "metadata": e.metadata, "relevance": e.relevance_score,
             "created_at": e.created_at}
            for e in entries
        ]
