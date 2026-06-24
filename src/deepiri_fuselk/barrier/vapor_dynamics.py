"""Vapor barrier dynamics and phase interface tracking."""

from __future__ import annotations

import numpy as np

from deepiri_fuselk.physics.pde_system import PDEParameters, PDEState, interface_thickness


def find_interface(state: PDEState) -> float:
    """Locate x_I where n_p ≈ n_v."""
    diff = np.abs(state.n_p - state.n_v)
    idx = int(np.argmin(diff))
    return float(state.x[idx])


def pressure_balance_residual(state: PDEState, B: float = 5.0, mu0: float = 4e-7 * np.pi) -> float:
    """Residual of P_magnetic + P_plasma = P_vapor at interface."""
    x_i = find_interface(state)
    idx = int(np.argmin(np.abs(state.x - x_i)))
    p_mag = B**2 / (2 * mu0)
    p_plasma = state.n_p[idx]
    p_vapor = state.n_v[idx]
    return float(p_mag + p_plasma - p_vapor)


def barrier_stability(params: PDEParameters) -> bool:
    """True if interface thickness is small enough for sharp phase separation."""
    return interface_thickness(params) < 1.0
