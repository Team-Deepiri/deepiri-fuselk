"""Tests for data fetch pipeline and normalization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from deepiri_fuselk.data.fetchers import run_fetch
from deepiri_fuselk.data.fetchers.manifest import load_manifest
from deepiri_fuselk.data.imas_loader import load_imas_hdf5
from deepiri_fuselk.data.normalize import odl_csv_to_shots
from deepiri_fuselk.data.sources import get_source, load_catalog

FIXTURE_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "odl_sample.csv"


def test_catalog_loads():
    sources, loops = load_catalog()
    assert len(sources) >= 5
    assert any(s.id == "odl" for s in sources)
    assert len(loops) >= 3
    odl = get_source("odl")
    assert odl is not None
    assert odl.tier == "public"


def test_odl_normalize_fixture(tmp_path: Path):
    out = odl_csv_to_shots(FIXTURE_CSV, tmp_path / "shots", max_discharges=10, grid_size=16)
    assert len(out) == 2
    shot = load_imas_hdf5(out[0])
    assert shot.device == "Alcator C-Mod"
    assert shot.heat_field.shape == (16, 16)
    assert shot.shot_id.startswith("CMOD_")


def test_fetch_synthetic_offline(tmp_path: Path):
    results = run_fetch(tmp_path, ["synthetic"], n_shots=12, grid_size=16)
    assert results["synthetic"].status == "ok"
    assert results["synthetic"].shots_written == 12
    corpus = tmp_path / "corpus" / "elm_corpus.npz"
    assert corpus.exists()
    data = np.load(corpus)
    assert data["features"].shape[0] == 12
    manifest = load_manifest(tmp_path)
    assert any(r.source_id == "synthetic" for r in manifest.records)


def test_fetch_manifest_written(tmp_path: Path):
    run_fetch(tmp_path, ["synthetic"], n_shots=5, grid_size=16)
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text())
    assert payload["records"][0]["source_id"] == "synthetic"
