"""Oil-water phase separation dynamics."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from deepiri_fuselk.physics.pde_system import PDEParameters, interface_thickness


def phase_separation_ode(t: float, y: np.ndarray, params: PDEParameters) -> list[float]:
    """Coupled 1D oil-water reaction-diffusion (spatially averaged modes)."""
    n_p, n_v = y
    alpha = params.alpha
    return [
        -alpha * n_p * n_v,
        alpha * n_p * n_v - params.v_v * 0.1 * n_v,
    ]


def simulate_phase_evolution(
    t_span: tuple[float, float] = (0, 10),
    params: PDEParameters | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate phase interface evolution."""
    params = params or PDEParameters()
    sol = solve_ivp(
        phase_separation_ode,
        t_span,
        [params.n0, params.n_wall * 0.1],
        args=(params,),
        dense_output=True,
    )
    return sol.t, sol.y


def oil_water_separated(params: PDEParameters) -> bool:
    """True when interface thickness indicates sharp separation."""
    return interface_thickness(params) < 1.0
