"""Data fetch orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from deepiri_fuselk.data.fetchers.manifest import FetchRecord, load_manifest, save_manifest
from deepiri_fuselk.data.fetchers.odl import fetch_odl
from deepiri_fuselk.data.fetchers.synthetic import fetch_synthetic

FETCHERS: dict[str, Callable[..., FetchRecord]] = {
    "odl": fetch_odl,
    "synthetic": fetch_synthetic,
}

PUBLIC_DEFAULTS = ("synthetic", "odl")


def run_fetch(
    root: Path,
    sources: list[str] | None = None,
    *,
    force: bool = False,
    n_shots: int = 100,
    grid_size: int = 32,
    max_odl_discharges: int = 50,
) -> dict[str, FetchRecord]:
    """Fetch and normalize all requested sources into .fuselk-data/."""
    selected = sources or list(PUBLIC_DEFAULTS)
    manifest = load_manifest(root)
    results: dict[str, FetchRecord] = {}

    for source_id in selected:
        if source_id not in FETCHERS:
            raise ValueError(f"Unknown source '{source_id}'. Available: {sorted(FETCHERS)}")
        if source_id == "odl":
            record = fetch_odl(root, force=force, max_discharges=max_odl_discharges)
        elif source_id == "synthetic":
            record = fetch_synthetic(root, n_shots=n_shots, grid_size=grid_size)
        else:
            record = FETCHERS[source_id](root)
        manifest.upsert(record)
        results[source_id] = record

    save_manifest(root, manifest)
    _write_catalog_snapshot(root)
    return results


def _write_catalog_snapshot(root: Path) -> None:
    import shutil

    from deepiri_fuselk.data.sources import _CATALOG_PATH

    catalog_dir = root / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_CATALOG_PATH, catalog_dir / "sources.yaml")
