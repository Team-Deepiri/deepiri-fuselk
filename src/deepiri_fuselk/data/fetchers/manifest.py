"""Fetch manifest tracking for .fuselk-data/."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class FetchRecord:
    source_id: str
    status: str
    fetched_at: str
    files: list[str] = field(default_factory=list)
    shots_written: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class Manifest:
    version: str = "1"
    records: list[FetchRecord] = field(default_factory=list)

    def upsert(self, record: FetchRecord) -> None:
        self.records = [r for r in self.records if r.source_id != record.source_id]
        self.records.append(record)

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "records": [asdict(r) for r in self.records]}


def manifest_path(root: Path) -> Path:
    return root / "manifest.json"


def load_manifest(root: Path) -> Manifest:
    path = manifest_path(root)
    if not path.exists():
        return Manifest()
    data = json.loads(path.read_text())
    records = [FetchRecord(**r) for r in data.get("records", [])]
    return Manifest(version=data.get("version", "1"), records=records)


def save_manifest(root: Path, manifest: Manifest) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = manifest_path(root)
    path.write_text(json.dumps(manifest.to_dict(), indent=2))
    return path


def now_iso() -> str:
    return datetime.now(UTC).isoformat()
