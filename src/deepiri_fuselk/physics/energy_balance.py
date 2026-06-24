"""Energy balance computations for plasma and vapor phases."""

from __future__ import annotations

import numpy as np

from deepiri_fuselk.physics.pde_system import (
    PDEParameters,
    PDEState,
    bremsstrahlung_power,
    radiative_cooling,
)


def plasma_energy_residual(state: PDEState, params: PDEParameters) -> np.ndarray:
    """Residual of plasma energy balance equation."""
    dx = state.x[1] - state.x[0]
    cond = params.kappa_p * np.gradient(np.gradient(state.T_p, dx), dx)
    brem = bremsstrahlung_power(state.n_p, state.T_p)
    exchange = params.alpha * state.n_p * state.n_v * state.T_p
    return cond - brem - exchange


def vapor_energy_residual(state: PDEState, params: PDEParameters) -> np.ndarray:
    """Residual of vapor energy balance equation."""
    dx = state.x[1] - state.x[0]
    cond = params.kappa_v * np.gradient(np.gradient(state.T_v, dx), dx)
    conv = params.v_v * np.gradient(state.T_v, dx)
    exchange = params.alpha * state.n_p * state.n_v * state.T_p
    rad = radiative_cooling(state.n_v, state.T_v)
    return cond - conv + exchange - rad


def total_heat_flux_to_wall(state: PDEState, params: PDEParameters) -> float:
    """Integrated heat flux reaching the wall."""
    dx = state.x[1] - state.x[0]
    q = params.kappa_v * np.gradient(state.T_v, dx)
    return float(q[-1])
