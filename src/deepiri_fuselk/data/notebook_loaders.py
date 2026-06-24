"""Helpers for notebooks — load shots and corpus from the fetch pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from deepiri_fuselk.data.fetchers import run_fetch
from deepiri_fuselk.data.fetchers.manifest import load_manifest
from deepiri_fuselk.data.imas_loader import IMASShot, load_imas_hdf5
from deepiri_fuselk.sim.synthetic_data_gen import SyntheticShot


def resolve_repo_root(start: Path | None = None) -> Path:
    """Find repository root (contains src/deepiri_fuselk)."""
    cwd = (start or Path.cwd()).resolve()
    if (cwd / "src" / "deepiri_fuselk").exists():
        return cwd
    if (cwd.parent / "src" / "deepiri_fuselk").exists():
        return cwd.parent
    return cwd


def resolve_data_root(repo: Path | None = None) -> Path:
    """Locate .fuselk-data/ relative to repo or cwd."""
    root = resolve_repo_root(repo)
    for candidate in (root / ".fuselk-data", root.parent / ".fuselk-data", Path(".fuselk-data")):
        if candidate.exists():
            return candidate.resolve()
    return (root / ".fuselk-data").resolve()


def ensure_fetched_data(
    root: Path | None = None,
    *,
    n_shots: int = 100,
    max_odl: int = 50,
    force: bool = False,
) -> Path:
    """Run public fetch if manifest or corpus is missing."""
    data_root = root or resolve_data_root()
    manifest = load_manifest(data_root)
    corpus = data_root / "corpus" / "elm_corpus.npz"
    needs = force or not manifest.records or not corpus.exists()
    if needs:
        data_root.mkdir(parents=True, exist_ok=True)
        run_fetch(data_root, force=force, n_shots=n_shots, max_odl_discharges=max_odl)
    return data_root


def list_shots(data_root: Path, *, source: str | None = None) -> list[Path]:
    """List HDF5 shot archives under .fuselk-data/shots/."""
    shots_dir = data_root / "shots"
    if not shots_dir.exists():
        return []
    paths = sorted(shots_dir.glob("*.h5"))
    if source == "cmod":
        return [p for p in paths if p.name.startswith("CMOD_")]
    if source == "synthetic":
        return [p for p in paths if p.name.startswith("SYN")]
    return paths


def load_odl_meta(path: str | Path) -> dict[str, Any] | None:
    """Read ODL density-limit labels from a normalized C-Mod shot."""
    with h5py.File(path, "r") as f:
        if "odl_meta" not in f:
            return None
        meta = f["odl_meta"]
        return {
            "density_limit_phase": np.array(meta["density_limit_phase"]),
            "density": np.array(meta["density"]),
            "discharge_id": str(f.attrs.get("discharge_id", "")),
            "elm_event_rate": float(f.attrs.get("elm_event_rate", 0.0)),
            "time": np.array(f["time"]) if "time" in f else None,
        }


def imas_to_synthetic_shot(shot: IMASShot) -> SyntheticShot:
    """Convert a fetched IMAS shot into HELIX-compatible ECE diagnostics."""
    size = shot.heat_field.shape[0]
    angles = np.linspace(0, 2 * np.pi, size, endpoint=False)
    raw = shot.heat_field.mean(axis=1).astype(np.float64)
    peak = float(np.max(shot.heat_field))
    std = float(np.std(shot.heat_field))
    island_amplitude = min(1.2, max(0.05, peak / max(std, 1e-9) * 0.12))
    return SyntheticShot(
        heat_field=shot.heat_field.astype(np.float64),
        raw_signal=raw,
        angles=angles,
        island_amplitude=island_amplitude,
    )


def load_corpus_arrays(data_root: Path) -> dict[str, Any]:
    """Load elm_corpus.npz produced by `fuselk data fetch`."""
    path = data_root / "corpus" / "elm_corpus.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {path}. Run: python scripts/fetch_data.py --all"
        )
    data = np.load(path)
    labels = np.array(data["labels"], dtype=np.float64)
    return {
        "features": np.array(data["features"], dtype=np.float64),
        "labels": labels,
        "elm_rate": float(data["elm_rate"][0]) if "elm_rate" in data else float(labels.mean()),
        "grid_size": int(data["grid_size"][0]) if "grid_size" in data else 32,
        "path": path,
    }


def manifest_summary(data_root: Path) -> list[dict[str, Any]]:
    """Human-readable manifest rows for notebook display."""
    manifest = load_manifest(data_root)
    return [
        {
            "source": r.source_id,
            "status": r.status,
            "shots": r.shots_written,
            "fetched_at": r.fetched_at,
            "details": r.details,
        }
        for r in manifest.records
    ]


def load_shot_by_id(data_root: Path, shot_id: str) -> IMASShot:
    """Load shot from .fuselk-data/shots/<shot_id>.h5."""
    path = data_root / "shots" / f"{shot_id}.h5"
    if not path.exists():
        path = data_root / "shots" / shot_id
    return load_imas_hdf5(path)
