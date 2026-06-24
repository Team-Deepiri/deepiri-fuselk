"""Integration tests for digital twin."""

import numpy as np
import pytest
from deepiri_fuselk.data.imas_loader import export_imas_hdf5, load_imas_hdf5, synthetic_imas_shot
from deepiri_fuselk.sim.digital_twin import DigitalTwin
from deepiri_fuselk.viz.dashboard.app import create_app


@pytest.mark.slow
def test_digital_twin_100_steps_no_nan():
    twin = DigitalTwin(grid_size=16)
    twin.reset()
    for _ in range(100):
        state = twin.step()
        assert not np.isnan(state.heat_variance)
        assert not np.isnan(state.elm_probability)


def test_digital_twin_20_steps():
    twin = DigitalTwin(grid_size=16)
    twin.reset()
    for _ in range(20):
        state = twin.step()
        assert state.helix_snr >= 0


def test_imas_hdf5_roundtrip(tmp_path):
    shot = synthetic_imas_shot("RT001", size=16)
    path = tmp_path / "shot.h5"
    export_imas_hdf5(shot, path)
    loaded = load_imas_hdf5(path)
    assert loaded.shot_id == "RT001"
    assert loaded.heat_field.shape == (16, 16)


def test_dashboard_app_creates():
    app = create_app()
    assert app.layout is not None
