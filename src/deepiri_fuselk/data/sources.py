"""Load the fuselk data source catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_CATALOG_PATH = Path(__file__).resolve().parents[3] / "config" / "data_sources.yaml"


@dataclass
class DataSource:
    id: str
    name: str
    tier: str
    device: str
    format: str
    fuselk_modules: list[str] = field(default_factory=list)
    homepage: str = ""
    license: str = ""
    download: dict[str, str] = field(default_factory=dict)
    ingest_path: str = ""
    normalized_path: str = ""
    notes: str = ""
    signals: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)


@dataclass
class FeedbackLoop:
    name: str
    in_: str
    through: str
    out: str


def load_catalog(path: Path | None = None) -> tuple[list[DataSource], list[FeedbackLoop]]:
    """Parse data/sources.yaml."""
    p = path or _CATALOG_PATH
    raw: dict[str, Any] = yaml.safe_load(p.read_text())
    sources = [
        DataSource(
            id=s["id"],
            name=s["name"],
            tier=s["tier"],
            device=s["device"],
            format=s["format"],
            fuselk_modules=s.get("fuselk_modules", []),
            homepage=s.get("homepage", ""),
            license=s.get("license", ""),
            download=s.get("download", {}),
            ingest_path=s.get("ingest_path", ""),
            normalized_path=s.get("normalized_path", ""),
            notes=(s.get("notes") or "").strip(),
            signals=s.get("signals", []),
            env=s.get("env", []),
        )
        for s in raw.get("sources", [])
    ]
    loops = [
        FeedbackLoop(
            name=fb["name"],
            in_=fb["in"],
            through=fb["through"],
            out=fb["out"],
        )
        for fb in raw.get("feedback_loops", [])
    ]
    return sources, loops


def get_source(source_id: str, path: Path | None = None) -> DataSource | None:
    sources, _ = load_catalog(path)
    for src in sources:
        if src.id == source_id:
            return src
    return None
