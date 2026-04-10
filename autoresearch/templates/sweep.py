"""Sweep generator: template × param values → N YAML configs.

CLI usage::

    python -m autoresearch.templates.sweep \\
        --template architecture \\
        --param NUM_LAYERS \\
        --values 6,7,8,9 \\
        --parent exp_000_baseline \\
        --prefix arch_layers

Generates autoresearch/queue/arch_layers_001.yaml ... arch_layers_004.yaml
"""

from __future__ import annotations

__all__ = ["generate_sweep"]

import argparse
import logging
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

QUEUE_DIR = Path(__file__).parent.parent / "queue"
TEMPLATES_DIR = Path(__file__).parent


def generate_sweep(
    template: str,
    param: str,
    values: List[str],
    parent: str,
    prefix: str,
    queue_dir: Path = QUEUE_DIR,
    templates_dir: Path = TEMPLATES_DIR,
) -> List[Path]:
    """Generate one YAML config per value, writing to *queue_dir*.

    Args:
        template: Template name (without .yaml), e.g. "architecture".
        param: The placeholder name to vary, e.g. "NUM_LAYERS".
        values: List of string values for the parameter.
        parent: Parent experiment ID, e.g. "exp_000_baseline".
        prefix: Output file prefix, e.g. "arch_layers".
        queue_dir: Directory to write generated configs into.
        templates_dir: Directory containing template YAML files.

    Returns:
        List of Paths to the generated YAML files.

    Raises:
        FileNotFoundError: if the template YAML does not exist.
    """
    template_path = templates_dir / f"{template}.yaml"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    raw = template_path.read_text(encoding="utf-8")
    queue_dir.mkdir(parents=True, exist_ok=True)

    generated: List[Path] = []
    for idx, value in enumerate(values, start=1):
        out_path = queue_dir / f"{prefix}_{idx:03d}.yaml"
        rendered = _render_single(raw, param, value, parent, prefix, idx)
        out_path.write_text(rendered, encoding="utf-8")
        logger.info("Generated %s (param %s=%s)", out_path.name, param, value)
        generated.append(out_path)

    return generated


def _render_single(
    raw: str,
    param: str,
    value: str,
    parent: str,
    prefix: str,
    idx: int,
) -> str:
    """Substitute the sweep parameter and required metadata placeholders."""
    exp_id = f"{prefix}_{idx:03d}"
    replacements = {
        param: value,
        "EXP_ID": exp_id,
        "EXP_NAME": f"{prefix} {param}={value}",
        "PARENT_ID": parent,
        "PRIORITY": "normal",
    }
    text = raw
    for placeholder, replacement in replacements.items():
        text = text.replace("{{" + placeholder + "}}", replacement)
        text = text.replace("{{ " + placeholder + " }}", replacement)
    return text


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a parameter sweep from a template."
    )
    parser.add_argument("--template", required=True, help="Template name (no .yaml)")
    parser.add_argument(
        "--param", required=True, help="Placeholder name to vary (e.g. NUM_LAYERS)"
    )
    parser.add_argument(
        "--values",
        required=True,
        help="Comma-separated values for the parameter (e.g. 6,7,8,9)",
    )
    parser.add_argument(
        "--parent", required=True, help="Parent experiment ID (e.g. exp_000_baseline)"
    )
    parser.add_argument(
        "--prefix", required=True, help="Output file prefix (e.g. arch_layers)"
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    values = [v.strip() for v in args.values.split(",") if v.strip()]
    paths = generate_sweep(
        template=args.template,
        param=args.param,
        values=values,
        parent=args.parent,
        prefix=args.prefix,
    )
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
