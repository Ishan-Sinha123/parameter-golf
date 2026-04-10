"""Git operations for experiment lifecycle management.

All git calls go through _run_git() which wraps subprocess.run and provides
consistent error handling and logging.
"""

from __future__ import annotations

__all__ = ["GitManager", "GitError"]

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

logger = logging.getLogger(__name__)

MAIN_BRANCH = "main"
EXP_BRANCH_PREFIX = "exp"


class GitError(Exception):
    """Raised when a git subprocess command fails."""


class GitManager:
    """Wrap git subprocess calls for experiment lifecycle management.

    Args:
        repo_path: Path to the git repository root. If not a git repo,
                   ``git init`` will be called automatically.
    """

    def __init__(self, repo_path: Path) -> None:
        self._repo = repo_path.resolve()
        if not self._repo.exists():
            raise GitError(f"Repository path does not exist: {self._repo}")
        if not self._is_git_repo():
            logger.info("No git repo at %s — running git init", self._repo)
            self._run_git("init")
            self._run_git("commit", "--allow-empty", "-m", "Initial empty commit")
        self._default_branch = self._detect_default_branch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_experiment_branch(self, exp_id: str) -> str:
        """Create and checkout a branch for *exp_id*.

        If the branch already exists, returns the branch name without error.

        Returns:
            The branch name (``exp/<exp_id>``).
        """
        branch = f"{EXP_BRANCH_PREFIX}/{exp_id}"
        self._ensure_clean_working_tree()
        self._ensure_on_main()

        existing = self._run_git("branch", "--list", branch).strip()
        if existing:
            logger.info("Branch %s already exists — skipping creation", branch)
        else:
            self._run_git("checkout", "-b", branch)
            logger.info("Created branch %s", branch)
            self._run_git("checkout", MAIN_BRANCH)

        return branch

    def commit_experiment_results(
        self,
        exp_id: str,
        status: str,
        metrics: Dict[str, float],
    ) -> str:
        """Stage all changes and commit them to the experiment branch.

        Args:
            exp_id: Experiment identifier.
            status: Human-readable status string (e.g. "screen_passed").
            metrics: Dict of metric name → value to embed in the commit message.

        Returns:
            The commit hash (short, 8 chars).
        """
        branch = f"{EXP_BRANCH_PREFIX}/{exp_id}"
        stashed = self._stash_if_dirty()
        try:
            self._run_git("checkout", branch)
            self._run_git("add", "-A")
            metrics_json = json.dumps(metrics, sort_keys=True)
            message = (
                f"[{exp_id}] {status}\n\n"
                f"metrics: {metrics_json}"
            )
            self._run_git("commit", "--allow-empty", "-m", message)
            commit_hash = self._run_git("rev-parse", "--short=8", "HEAD").strip()
            logger.info(
                "Committed results for %s on branch %s (%s)",
                exp_id,
                branch,
                commit_hash,
            )
        finally:
            self._run_git("checkout", MAIN_BRANCH)
            if stashed:
                self._run_git("stash", "pop")
        return commit_hash

    def tag_promotion(
        self,
        exp_id: str,
        metrics: Dict[str, float],
    ) -> str:
        """Create an annotated tag on the experiment branch HEAD.

        Args:
            exp_id: Experiment identifier.
            metrics: Metrics to embed in the tag annotation.

        Returns:
            The tag name (``promoted/<exp_id>``).
        """
        tag = f"promoted/{exp_id}"
        branch = f"{EXP_BRANCH_PREFIX}/{exp_id}"
        metrics_json = json.dumps(metrics, sort_keys=True)
        annotation = f"Promoted {exp_id}\n\nmetrics: {metrics_json}"

        stashed = self._stash_if_dirty()
        try:
            branch_sha = self._run_git(
                "rev-parse", f"refs/heads/{branch}"
            ).strip()
            existing_tags = self._run_git("tag", "--list", tag).strip()
            if existing_tags:
                logger.warning("Tag %s already exists — skipping", tag)
            else:
                self._run_git(
                    "tag", "-a", tag, branch_sha, "-m", annotation
                )
                logger.info("Tagged %s as %s", exp_id, tag)
        finally:
            if stashed:
                self._run_git("stash", "pop")
        return tag

    def get_experiment_lineage(self, exp_id: str) -> List[str]:
        """Return the list of commit hashes on the experiment branch.

        The list is ordered from oldest to newest.

        Returns:
            List of full commit hashes.
        """
        branch = f"{EXP_BRANCH_PREFIX}/{exp_id}"
        existing = self._run_git("branch", "--list", branch).strip()
        if not existing:
            logger.warning("Branch %s does not exist", branch)
            return []
        log_output = self._run_git(
            "log", "--pretty=format:%H", "--reverse", branch
        )
        return [line.strip() for line in log_output.splitlines() if line.strip()]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_git(self, *args: str) -> str:
        """Run a git command in the repo directory and return stdout.

        Raises:
            GitError: if the command exits with a non-zero status.
        """
        cmd = ["git", *args]
        logger.debug("Running: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=self._repo,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise GitError(
                f"git command failed: {' '.join(cmd)}\n"
                f"stderr: {result.stderr.strip()}"
            )
        return result.stdout

    def _is_git_repo(self) -> bool:
        try:
            self._run_git("rev-parse", "--git-dir")
            return True
        except GitError:
            return False

    def _current_branch(self) -> Optional[str]:
        try:
            return self._run_git("rev-parse", "--abbrev-ref", "HEAD").strip()
        except GitError:
            return None

    def _is_detached_head(self) -> bool:
        return self._current_branch() == "HEAD"

    def _ensure_on_main(self) -> None:
        """Checkout main (or create it) if we are in detached HEAD state."""
        if self._is_detached_head():
            logger.warning("Detached HEAD detected — checking out %s", MAIN_BRANCH)
            try:
                self._run_git("checkout", MAIN_BRANCH)
            except GitError:
                self._run_git("checkout", "-b", MAIN_BRANCH)

    def _is_working_tree_dirty(self) -> bool:
        status = self._run_git("status", "--porcelain").strip()
        return bool(status)

    def _stash_if_dirty(self) -> bool:
        if self._is_working_tree_dirty():
            logger.info("Working tree dirty — stashing changes")
            self._run_git("stash", "push", "--include-untracked", "-m", "autoresearch-stash")
            return True
        return False

    def _ensure_clean_working_tree(self) -> None:
        """Stash dirty working tree before branch operations."""
        self._stash_if_dirty()
