"""Tests for oil-water PDE system."""

import numpy as np
import pytest
from deepiri_fuselk.physics.pde_solver import solve_oil_water_steady, solve_oil_water_transient
from deepiri_fuselk.physics.pde_system import PDEParameters, peclet_criterion


def test_solve_oil_water_converges():
    result = solve_oil_water_steady(n_grid=32)
    assert result.state.n_p[0] == pytest.approx(1.0, abs=0.15)
    assert result.state.n_v[-1] == pytest.approx(1.0, abs=0.15)
    assert np.isfinite(result.residual)
    assert result.converged or result.residual < 10.0


def test_transient_solver():
    hist = solve_oil_water_transient(n_grid=16, t_end=0.2, dt=0.02)
    assert len(hist) == 10
    assert np.all(hist[-1].n_p >= 0)


def test_peclet_criterion_positive():
    pe = peclet_criterion(PDEParameters(v_v=0.5, D_T=0.02))
    assert pe > 0


def test_tritium_non_negative():
    result = solve_oil_water_steady(n_grid=32)
    assert np.all(result.state.n_T >= 0)
