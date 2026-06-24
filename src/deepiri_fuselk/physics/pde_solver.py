"""1D finite-difference solver for the oil-water coupled PDE system."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from deepiri_fuselk.physics.pde_system import PDEParameters, PDEState, tritium_source


@dataclass
class SolverResult:
    state: PDEState
    converged: bool
    iterations: int
    residual: float


def solve_oil_water_steady(
    n_grid: int = 64,
    length: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-6,
    params: PDEParameters | None = None,
) -> SolverResult:
    """Relaxation solve for steady-state plasma/vapor/tritium profiles."""
    params = params or PDEParameters()
    x = np.linspace(0.0, length, n_grid)
    dx = x[1] - x[0]

    n_p = np.clip(params.n0 * (1.0 - x / length), 0.0, params.n0)
    n_v = np.clip(params.n_wall * (x / length), 0.0, params.n_wall)
    n_T = np.zeros(n_grid)
    T_p = np.clip(1.0 - 0.8 * (x / length), 0.01, 1.0)
    T_v = np.clip(0.2 + 0.3 * (x / length), 0.01, 1.0)

    relax = 0.05
    residual = float("inf")

    for iteration in range(1, max_iter + 1):
        n_p_old = n_p.copy()
        n_v_old = n_v.copy()

        # Plasma diffusion with weak reaction coupling
        d2n_p = np.zeros(n_grid)
        d2n_p[1:-1] = (n_p[2:] - 2 * n_p[1:-1] + n_p[:-2]) / dx**2
        reaction_p = params.alpha * n_p * n_v
        n_p = np.clip(n_p + relax * (params.D_p * d2n_p - reaction_p), 0.0, params.n0)
        n_p[0] = params.n0
        n_p[-1] = 0.0

        # Vapor diffusion + advection + reaction
        d2n_v = np.zeros(n_grid)
        d2n_v[1:-1] = (n_v[2:] - 2 * n_v[1:-1] + n_v[:-2]) / dx**2
        dn_v_dx = np.zeros(n_grid)
        dn_v_dx[1:-1] = (n_v[2:] - n_v[:-2]) / (2 * dx)
        reaction_v = params.alpha * n_p * n_v
        n_v = np.clip(
            n_v + relax * (params.D_v * d2n_v - params.v_v * dn_v_dx + reaction_v),
            0.0,
            params.n_wall,
        )
        n_v[0] = 0.0
        n_v[-1] = params.n_wall

        # Tritium advection-diffusion with breeding source
        state_tmp = PDEState(x=x, n_p=n_p, n_v=n_v, n_T=n_T, T_p=T_p, T_v=T_v)
        source = np.maximum(tritium_source(params, state_tmp), 0.0)
        lower = -(params.D_T / dx**2 - params.v_v / (2 * dx)) * np.ones(n_grid - 1)
        main = (2 * params.D_T / dx**2) * np.ones(n_grid)
        upper = -(params.D_T / dx**2 + params.v_v / (2 * dx)) * np.ones(n_grid - 1)
        A = diags([lower, main, upper], [-1, 0, 1], format="csc")
        n_T = np.maximum(spsolve(A, source), 0.0)

        residual = float(np.max(np.abs(n_p - n_p_old)) + np.max(np.abs(n_v - n_v_old)))
        if not np.isfinite(residual):
            residual = float("inf")
            break
        if residual < tol:
            state = PDEState(x=x, n_p=n_p, n_v=n_v, n_T=n_T, T_p=T_p, T_v=T_v)
            return SolverResult(
                state=state, converged=True, iterations=iteration, residual=residual
            )

    state = PDEState(x=x, n_p=n_p, n_v=n_v, n_T=n_T, T_p=T_p, T_v=T_v)
    if not np.isfinite(residual):
        residual = 1.0
    return SolverResult(
        state=state, converged=residual < tol, iterations=max_iter, residual=residual
    )
