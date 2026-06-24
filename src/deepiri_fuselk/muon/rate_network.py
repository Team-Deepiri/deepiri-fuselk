"""Muon catalytic fusion rate network."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

BREAKEVEN_FUSIONS = 284.0


@dataclass
class RateNetworkParams:
    lambda_form: float = 1e8
    lambda_transfer: float = 1e7
    lambda_fusion: float = 1e9
    lambda_mu: float = 4.5e5
    omega_0: float = 0.01
    R_col: float = 0.35
    R_photon: float = 0.2
    R_proton: float = 0.15
    S_source: float = 1e6


@dataclass
class RateNetworkResult:
    t: np.ndarray
    populations: np.ndarray
    fusions_per_muon: float
    breakeven: bool


def effective_sticking(params: RateNetworkParams) -> float:
    """Effective alpha-sticking after external stripping."""
    R_strip = params.R_col + params.R_photon + params.R_proton
    return params.omega_0 * (1.0 - params.R_col) * (1.0 - R_strip)


def rate_equations(t: float, y: np.ndarray, params: RateNetworkParams) -> list[float]:
    """ODE system: [N_mu, N_dmu, N_dtmu, N_alpha_mu]."""
    n_mu, n_dmu, n_dtmu, n_alpha = y
    R_strip = params.R_col + params.R_photon + params.R_proton
    dn_mu = (
        params.S_source + R_strip * n_alpha - params.lambda_form * n_mu - params.lambda_mu * n_mu
    )
    dn_dmu = params.lambda_form * n_mu - params.lambda_transfer * n_dmu
    dn_dtmu = params.lambda_transfer * n_dmu - params.lambda_fusion * n_dtmu
    dn_alpha = (
        params.lambda_fusion * n_dtmu * params.omega_0 - R_strip * n_alpha - params.R_col * n_alpha
    )
    return [dn_mu, dn_dmu, dn_dtmu, dn_alpha]


def run_rate_network(
    t_span: tuple[float, float] = (0.0, 1e-5),
    params: RateNetworkParams | None = None,
) -> RateNetworkResult:
    """Integrate muon population dynamics."""
    params = params or RateNetworkParams()
    y0 = [0.0, 0.0, 0.0, 0.0]
    sol = solve_ivp(
        rate_equations,
        t_span,
        y0,
        args=(params,),
        method="RK45",
        max_step=1e-7,
    )
    dt = np.diff(sol.t)
    fusion_events = params.lambda_fusion * sol.y[2][:-1] * dt
    total_fusions = float(np.sum(fusion_events))
    muons_introduced = params.S_source * (t_span[1] - t_span[0])
    fusions_per_muon = total_fusions / max(muons_introduced, 1.0)
    return RateNetworkResult(
        t=sol.t,
        populations=sol.y,
        fusions_per_muon=fusions_per_muon,
        breakeven=fusions_per_muon >= BREAKEVEN_FUSIONS,
    )
