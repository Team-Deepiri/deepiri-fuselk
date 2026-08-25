"""SOLPS edge ingest and Venturi coupling tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from deepiri_fuselk.control.venturi_controller import VenturiController
from deepiri_fuselk.sim.solps_ingest import (
    export_solps_hdf5,
    load_solps_hdf5,
    synthetic_solps_edge,
)
from deepiri_fuselk.sim.solps_wrapper import SOLPSConfig, SOLPSWrapper


def test_synthetic_solps_edge_shapes():
    edge = synthetic_solps_edge(grid_size=24, q_peak_mw=10.0, seed=1)
    assert edge.heat_flux_2d.shape == (24, 24)
    assert edge.peak_heat() > 0
    assert edge.ne[0] > edge.ne[-1]


def test_solps_hdf5_roundtrip(tmp_path: Path):
    edge = synthetic_solps_edge(grid_size=16, seed=2)
    path = export_solps_hdf5(edge, tmp_path / "edge.h5")
    loaded = load_solps_hdf5(path)
    assert np.allclose(loaded.ne, edge.ne)
    assert np.allclose(loaded.heat_flux_2d, edge.heat_flux_2d)


def test_wrapper_feeds_venturi():
    wrap = SOLPSWrapper(SOLPSConfig(grid_points=16))
    wrap.load_edge(q_peak_mw=12.0, seed=0)
    heat = wrap.divertor_heat_map(16)
    assert heat.shape == (16, 16)
    st = VenturiController(engineering_limit=10.0).step(heat, elm_probability=0.4)
    assert isinstance(st.reward, float)


def test_wrapper_binary_not_linked_by_default():
    assert SOLPSWrapper().available() is False
    assert SOLPSWrapper().ingest_available() is True
