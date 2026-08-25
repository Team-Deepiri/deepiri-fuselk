"""Path confinement helpers for local data-root I/O.

CodeQL flags joins of caller-supplied roots with relative segments as
path-injection. Resolving under a base and checking ``is_relative_to``
is the recognized sanitizer for those sinks.
"""

from __future__ import annotations

from pathlib import Path


def under_root(root: Path | str, *parts: str | Path) -> Path:
    """Join ``parts`` under ``root`` and refuse any escape outside it."""
    base = Path(root).resolve()
    target = base.joinpath(*[str(p) for p in parts]).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"Path escapes data root {base}: {target}")
    return target


def resolve_within(root: Path | str, path: Path | str) -> Path:
    """Resolve ``path`` and require it to stay under ``root``."""
    base = Path(root).resolve()
    target = Path(path).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"Path escapes data root {base}: {target}")
    return target
