"""Tests for oil-water PDE system."""

import numpy as np
import pytest
from deepiri_fuselk.physics.pde_solver import solve_oil_water_steady
from deepiri_fuselk.physics.pde_system import PDEParameters, peclet_criterion


def test_solve_oil_water_converges():
    result = solve_oil_water_steady(n_grid=32, max_iter=500)
    assert result.state.n_p[0] == pytest.approx(1.0, abs=0.1)
    assert result.state.n_v[-1] == pytest.approx(1.0, abs=0.1)
    assert np.isfinite(result.residual)
    assert result.residual < 5.0


def test_peclet_criterion_positive():
    params = PDEParameters(v_v=0.5, D_T=0.02)
    pe = peclet_criterion(params)
    assert pe > 0


def test_tritium_non_negative():
    result = solve_oil_water_steady(n_grid=32, max_iter=200)
    assert np.all(result.state.n_T >= 0)
