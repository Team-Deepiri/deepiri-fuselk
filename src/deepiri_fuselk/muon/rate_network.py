"""Full muon catalytic rate network with extended population tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

BREAKEVEN_FUSIONS = 284.0


@dataclass
class RateNetworkParams:
    lambda_form: float = 1e8
    lambda_transfer: float = 1e7
    lambda_fusion: float = 1e9
    lambda_mu: float = 4.5e5
    lambda_tmu: float = 1e7
    omega_0: float = 0.008
    R_col: float = 0.35
    R_photon: float = 0.25
    R_proton: float = 0.18
    S_source: float = 1e6
    dt_mu: float = 2.2e-6


@dataclass
class RateNetworkResult:
    t: np.ndarray
    populations: np.ndarray
    fusions_per_muon: float
    breakeven: bool
    effective_sticking: float
    strip_rate: float
    population_names: list[str] = field(
        default_factory=lambda: ["N_mu", "N_dmu", "N_tmu", "N_dtmu", "N_alpha_mu"]
    )


def strip_rate(params: RateNetworkParams) -> float:
    return params.R_col + params.R_photon + params.R_proton


def effective_sticking(params: RateNetworkParams) -> float:
    R = strip_rate(params)
    return params.omega_0 * max(0.0, 1.0 - R)


def rate_equations(t: float, y: np.ndarray, params: RateNetworkParams) -> list[float]:
    """Extended ODE: free muon, dμ, tμ, dtμ molecule, stuck (αμ)+."""
    n_mu, n_dmu, n_tmu, n_dtmu, n_alpha = y
    R = strip_rate(params)
    dn_mu = params.S_source + R * n_alpha - params.lambda_form * n_mu - n_mu / params.dt_mu
    dn_dmu = params.lambda_form * n_mu - params.lambda_transfer * n_dmu
    dn_tmu = params.lambda_transfer * n_dmu - params.lambda_tmu * n_tmu
    dn_dtmu = params.lambda_tmu * n_tmu + params.lambda_transfer * n_dmu * 0.1 - params.lambda_fusion * n_dtmu
    dn_alpha = params.lambda_fusion * n_dtmu * params.omega_0 - R * n_alpha
    return [dn_mu, dn_dmu, dn_tmu, dn_dtmu, dn_alpha]


def run_rate_network(
    t_span: tuple[float, float] = (0.0, 5e-5),
    params: RateNetworkParams | None = None,
    y0: list[float] | None = None,
) -> RateNetworkResult:
    params = params or RateNetworkParams()
    initial = y0 or [0.0, 0.0, 0.0, 0.0, 0.0]
    sol = solve_ivp(
        rate_equations,
        t_span,
        initial,
        args=(params,),
        method="BDF",
        max_step=1e-7,
        rtol=1e-8,
        atol=1e-10,
    )
    dt = np.diff(sol.t)
    fusion_rate = params.lambda_fusion * sol.y[3][:-1]
    total_fusions = float(np.sum(fusion_rate * dt))
    muons = params.S_source * (t_span[1] - t_span[0])
    fpm = total_fusions / max(muons, 1.0)
    eff = effective_sticking(params)
    return RateNetworkResult(
        t=sol.t,
        populations=sol.y,
        fusions_per_muon=fpm,
        breakeven=fpm >= BREAKEVEN_FUSIONS,
        effective_sticking=eff,
        strip_rate=strip_rate(params),
    )
