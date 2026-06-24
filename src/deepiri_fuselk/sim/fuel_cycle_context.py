"""Shared oil-water PDE + muon trifecta state for reactor and fusion cell."""

from __future__ import annotations

from dataclasses import dataclass

from deepiri_fuselk.muon.stripping_orchestrator import (
    StrippingTrifectaResult,
    run_stripping_trifecta,
)
from deepiri_fuselk.physics.pde_solver import SolverResult, solve_oil_water_steady
from deepiri_fuselk.physics.pde_system import PDEParameters


@dataclass
class FuelCycleContext:
    """Single solve of barrier PDE and muon stack — shared across orchestrators."""

    pde_params: PDEParameters
    pde: SolverResult
    muon_trifecta: StrippingTrifectaResult

    @property
    def muon_fpm(self) -> float:
        return self.muon_trifecta.projected_fpm


def build_fuel_cycle_context(
    grid_size: int,
    *,
    params: PDEParameters | None = None,
) -> FuelCycleContext:
    pde_params = params or PDEParameters.certified()
    pde = solve_oil_water_steady(n_grid=grid_size, params=pde_params)
    muon = run_stripping_trifecta()
    return FuelCycleContext(pde_params=pde_params, pde=pde, muon_trifecta=muon)
