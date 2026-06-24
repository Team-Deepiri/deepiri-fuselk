"""Load experiment catalog from YAML registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ExperimentEntry:
    id: str
    name: str
    category: str
    status: str
    module: str
    description: str
    runner: str | None = None


def registry_path() -> Path:
    return Path(__file__).resolve().parents[3] / "experiments" / "registry.yaml"


def load_registry(path: Path | None = None) -> list[ExperimentEntry]:
    path = path or registry_path()
    data = yaml.safe_load(path.read_text())
    entries = []
    for item in data.get("experiments", []):
        entries.append(
            ExperimentEntry(
                id=item["id"],
                name=item["name"],
                category=item["category"],
                status=item["status"],
                module=item["module"],
                description=item["description"],
                runner=item.get("runner"),
            )
        )
    return entries


def get_experiment(exp_id: str, path: Path | None = None) -> ExperimentEntry | None:
    for entry in load_registry(path):
        if entry.id == exp_id:
            return entry
    return None
