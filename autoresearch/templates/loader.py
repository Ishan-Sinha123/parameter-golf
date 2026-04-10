"""YAML template loader with {{ placeholder }} substitution.

No Jinja2 — substitution is done via simple string replacement.
"""

from __future__ import annotations

__all__ = ["TemplateEngine"]

import logging
import re
from pathlib import Path
from typing import Dict

import yaml

from autoresearch.templates.types import AutoResearchError
from autoresearch.templates.types import Category
from autoresearch.templates.types import PLACEHOLDER_RE
from autoresearch.templates.types import Priority
from autoresearch.templates.types import TemplateConfig
from autoresearch.templates.types import TemplateError

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent


class TemplateEngine:
    """Load YAML templates and render them by substituting {{ PLACEHOLDER }} tokens."""

    def __init__(self, templates_dir: Path = TEMPLATES_DIR) -> None:
        self._dir = templates_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, name: str) -> str:
        """Return the raw template text for *name* (without .yaml extension)."""
        path = self._dir / f"{name}.yaml"
        if not path.exists():
            raise TemplateError(f"Template not found: {path}")
        text = path.read_text(encoding="utf-8")
        logger.debug("Loaded template %s (%d chars)", name, len(text))
        return text

    def render(self, name: str, params: Dict[str, str]) -> TemplateConfig:
        """Load *name* template, substitute *params*, and return a TemplateConfig.

        Raises:
            TemplateError: if any {{ }} placeholders remain after substitution.
        """
        raw = self.load(name)
        rendered = self._substitute(raw, params)
        self._validate_no_remaining_placeholders(rendered)
        return self._parse(rendered)

    def render_text(self, raw: str, params: Dict[str, str]) -> str:
        """Substitute *params* into *raw* template text and validate completeness."""
        rendered = self._substitute(raw, params)
        self._validate_no_remaining_placeholders(rendered)
        return rendered

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _substitute(self, text: str, params: Dict[str, str]) -> str:
        for placeholder, value in params.items():
            token = "{{" + placeholder + "}}"
            text = text.replace(token, value)
            # Also handle {{ PLACEHOLDER }} with surrounding spaces
            token_spaced = "{{ " + placeholder + " }}"
            text = text.replace(token_spaced, value)
        return text

    def _validate_no_remaining_placeholders(self, text: str) -> None:
        remaining = PLACEHOLDER_RE.findall(text)
        if remaining:
            raise TemplateError(
                f"Unresolved placeholders after substitution: {remaining}"
            )

    def _parse(self, text: str) -> TemplateConfig:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise TemplateError(f"YAML parse error: {exc}") from exc

        if not isinstance(data, dict):
            raise TemplateError("Template YAML must be a mapping at the top level")

        try:
            return TemplateConfig(
                id=str(data["id"]),
                name=str(data["name"]),
                parent=str(data.get("parent", "")),
                hypothesis=str(data.get("hypothesis", "")),
                category=Category(data.get("category", "hyperparameter")),
                priority=Priority(data.get("priority", "normal")),
                env_overrides={
                    str(k): str(v)
                    for k, v in data.get("env_overrides", {}).items()
                },
                stages=list(data.get("stages", ["screen", "gate"])),
                reject_if_worse_by=float(data.get("reject_if_worse_by", 0.05)),
                raw_text=text,
            )
        except (KeyError, ValueError) as exc:
            raise TemplateError(f"Invalid template structure: {exc}") from exc
