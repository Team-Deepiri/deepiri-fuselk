"""Integration tests for digital twin."""

import numpy as np
from deepiri_fuselk.sim.digital_twin import DigitalTwin
from deepiri_fuselk.sim.synthetic_data_gen import generate_ece_shot
from deepiri_fuselk.viz.dashboard.app import create_app


def test_digital_twin_100_steps_no_nan():
    twin = DigitalTwin(grid_size=16)
    twin.reset()
    for _ in range(100):
        state = twin.step()
        assert not np.isnan(state.heat_variance)
        assert not np.isnan(state.elm_probability)


def test_synthetic_shot_shapes():
    shot = generate_ece_shot(16, seed=99)
    assert shot.heat_field.shape == (16, 16)
    assert len(shot.raw_signal) == 16


def test_dashboard_app_creates():
    app = create_app()
    assert app.layout is not None
