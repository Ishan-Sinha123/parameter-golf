"""Template system for autoresearch."""

from autoresearch.templates.loader import TemplateEngine
from autoresearch.templates.types import AutoResearchError
from autoresearch.templates.types import Category
from autoresearch.templates.types import Priority
from autoresearch.templates.types import TemplateConfig
from autoresearch.templates.types import TemplateError

__all__ = [
    "AutoResearchError",
    "Category",
    "Priority",
    "TemplateConfig",
    "TemplateEngine",
    "TemplateError",
]
