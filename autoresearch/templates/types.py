"""Template-specific types and exceptions for autoresearch."""

from __future__ import annotations

__all__ = [
    "AutoResearchError",
    "TemplateError",
    "Category",
    "Priority",
    "TemplateConfig",
]

import re
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional


class AutoResearchError(Exception):
    """Base exception for all autoresearch errors."""


class TemplateError(AutoResearchError):
    """Raised when a template cannot be loaded or rendered."""


class Category(str, Enum):
    ARCHITECTURE = "architecture"
    HYPERPARAMETER = "hyperparameter"
    EVALUATION = "evaluation"
    TTT = "ttt"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


PLACEHOLDER_RE = re.compile(r"\{\{[^}]*\}\}")


@dataclass(frozen=True)
class TemplateConfig:
    """Immutable representation of a parsed experiment template.

    All fields mirror the canonical YAML template format.
    """

    id: str
    name: str
    parent: str
    hypothesis: str
    category: Category
    priority: Priority
    env_overrides: Dict[str, str]
    stages: List[str]
    reject_if_worse_by: float
    raw_text: str = field(compare=False, repr=False)

    def has_unresolved_placeholders(self) -> bool:
        """Return True if any {{ }} placeholders remain in any string field."""
        for value in (self.id, self.name, self.parent, self.hypothesis):
            if PLACEHOLDER_RE.search(value):
                return True
        for v in self.env_overrides.values():
            if PLACEHOLDER_RE.search(v):
                return True
        return False
