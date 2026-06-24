"""Newton solver for coupled oil-water steady-state system."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from deepiri_fuselk.physics.pde_system import PDEParameters, PDEState, tritium_source
from deepiri_fuselk.physics.pde_wellposedness import WellposednessReport, verify_wellposedness


@dataclass
class SolverResult:
    state: PDEState
    converged: bool
    iterations: int
    residual: float
    wellposedness: WellposednessReport | None = None


def _build_laplacian_matrix(n: int, dx: float) -> np.ndarray:
    off = np.ones(n - 1) / dx**2
    main = -2.0 * np.ones(n) / dx**2
    return np.array(np.diag(main) + np.diag(off, 1) + np.diag(off, -1))


def _residuals(
    y: np.ndarray,
    x: np.ndarray,
    dx: float,
    params: PDEParameters,
    lap: np.ndarray,
) -> np.ndarray:
    """Coupled residual vector for [n_p interior, n_v interior]."""
    n = len(x)
    n_p = np.zeros(n)
    n_v = np.zeros(n)
    interior = slice(1, -1)
    n_p[0], n_p[-1] = params.n0, 0.0
    n_v[0], n_v[-1] = 0.0, params.n_wall
    n_p[interior] = y[: n - 2]
    n_v[interior] = y[n - 2 :]

    d2n_p = lap @ n_p
    d2n_v = lap @ n_v
    dn_v_dx = np.gradient(n_v, dx)
    react = params.alpha * n_p * n_v

    res_p = params.D_p * d2n_p - react
    res_v = params.D_v * d2n_v - params.v_v * dn_v_dx + react

    return np.concatenate([res_p[interior], res_v[interior]])


def _solve_tritium(
    n_p: np.ndarray, n_v: np.ndarray, x: np.ndarray, params: PDEParameters
) -> np.ndarray:
    n = len(x)
    dx = x[1] - x[0]
    state = PDEState(x=x, n_p=n_p, n_v=n_v, n_T=np.zeros(n), T_p=np.zeros(n), T_v=np.zeros(n))
    source = np.maximum(tritium_source(params, state), 0.0)
    lower = -(params.D_T / dx**2 - params.v_v / (2 * dx)) * np.ones(n - 1)
    main = (2 * params.D_T / dx**2) * np.ones(n)
    upper = -(params.D_T / dx**2 + params.v_v / (2 * dx)) * np.ones(n - 1)
    A = diags([lower, main, upper], [-1, 0, 1], format="csc")
    return np.maximum(spsolve(A, source), 0.0)


def solve_oil_water_steady(
    n_grid: int = 64,
    length: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-8,
    params: PDEParameters | None = None,
) -> SolverResult:
    """Newton solve for steady-state plasma/vapor; tritium via linear sub-solve."""
    params = params or PDEParameters.certified()
    wellposedness = verify_wellposedness(params)
    if not wellposedness.steady_uniqueness:
        params = PDEParameters.certified()
        wellposedness = verify_wellposedness(params)
    x = np.linspace(0.0, length, n_grid)
    dx = x[1] - x[0]
    lap = _build_laplacian_matrix(n_grid, dx)

    n_p_init = params.n0 * (1.0 - x / length)
    n_v_init = params.n_wall * (x / length)
    y0 = np.concatenate([n_p_init[1:-1], n_v_init[1:-1]])

    result = root(
        _residuals,
        y0,
        args=(x, dx, params, lap),
        method="hybr",
        options={"xtol": tol, "maxfev": max_iter * n_grid},
    )

    n = n_grid
    n_p = np.zeros(n)
    n_v = np.zeros(n)
    n_p[0], n_p[-1] = params.n0, 0.0
    n_v[0], n_v[-1] = 0.0, params.n_wall
    n_p[1:-1] = np.clip(result.x[: n - 2], 0, params.n0)
    n_v[1:-1] = np.clip(result.x[n - 2 :], 0, params.n_wall)
    n_T = _solve_tritium(n_p, n_v, x, params)
    T_p = np.clip(1.0 - 0.8 * (x / length), 0.01, 1.0)
    T_v = np.clip(0.2 + 0.3 * (x / length), 0.01, 1.0)

    state = PDEState(x=x, n_p=n_p, n_v=n_v, n_T=n_T, T_p=T_p, T_v=T_v)
    residual = float(np.linalg.norm(result.fun, ord=np.inf)) if result.fun is not None else 1.0
    return SolverResult(
        state=state,
        converged=result.success,
        iterations=getattr(result, "nfev", max_iter),
        residual=residual,
        wellposedness=wellposedness,
    )


def solve_oil_water_transient(
    n_grid: int = 64,
    length: float = 1.0,
    t_end: float = 1.0,
    dt: float = 0.01,
    params: PDEParameters | None = None,
) -> list[PDEState]:
    """Explicit time-stepping for transient oil-water dynamics."""
    params = params or PDEParameters()
    x = np.linspace(0.0, length, n_grid)
    dx = x[1] - x[0]
    n_steps = int(t_end / dt)

    n_p = params.n0 * (1.0 - x / length)
    n_v = params.n_wall * (x / length) * 0.5
    n_T = np.zeros(n_grid)
    T_p = 1.0 - 0.8 * (x / length)
    T_v = 0.2 + 0.3 * (x / length)

    history: list[PDEState] = []
    for _ in range(n_steps):
        react = params.alpha * n_p * n_v
        d2n_p = np.zeros(n_grid)
        d2n_v = np.zeros(n_grid)
        d2n_p[1:-1] = (n_p[2:] - 2 * n_p[1:-1] + n_p[:-2]) / dx**2
        d2n_v[1:-1] = (n_v[2:] - 2 * n_v[1:-1] + n_v[:-2]) / dx**2
        dn_v_dx = np.gradient(n_v, dx)

        n_p = np.clip(n_p + dt * (params.D_p * d2n_p - react), 0, params.n0)
        n_v = np.clip(
            n_v + dt * (params.D_v * d2n_v - params.v_v * dn_v_dx + react), 0, params.n_wall
        )
        n_p[0], n_p[-1] = params.n0, 0.0
        n_v[0], n_v[-1] = 0.0, params.n_wall

        n_T = _solve_tritium(n_p, n_v, x, params)
        history.append(
            PDEState(x=x, n_p=n_p.copy(), n_v=n_v.copy(), n_T=n_T.copy(), T_p=T_p, T_v=T_v)
        )
    return history
