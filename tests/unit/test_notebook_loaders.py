"""Tests for notebook data loader helpers."""

from __future__ import annotations

from pathlib import Path

from deepiri_fuselk.data.fetchers import run_fetch
from deepiri_fuselk.data.imas_loader import load_imas_hdf5
from deepiri_fuselk.data.notebook_loaders import (
    ensure_fetched_data,
    imas_to_synthetic_shot,
    list_shots,
    load_corpus_arrays,
    load_odl_meta,
    load_shot_by_id,
    manifest_summary,
)


def test_notebook_loaders_on_fetched_data(tmp_path: Path):
    run_fetch(tmp_path, ["synthetic", "odl"], n_shots=8, grid_size=16, max_odl_discharges=3)
    root = ensure_fetched_data(tmp_path)

    assert list_shots(root, source="cmod")
    assert list_shots(root, source="synthetic")

    corpus = load_corpus_arrays(root)
    assert corpus["features"].shape[0] == corpus["labels"].shape[0]

    cmod_path = list_shots(root, source="cmod")[0]
    shot = load_imas_hdf5(cmod_path)
    ece = imas_to_synthetic_shot(shot)
    assert ece.heat_field.shape == shot.heat_field.shape
    assert len(ece.raw_signal) == ece.heat_field.shape[0]

    meta = load_odl_meta(cmod_path)
    assert meta is not None
    assert meta["density_limit_phase"].size > 0

    summary = manifest_summary(root)
    assert len(summary) == 2

    loaded = load_shot_by_id(root, shot.shot_id)
    assert loaded.shot_id == shot.shot_id
