"""Physics subpackage."""

from deepiri_fuselk.physics.energy_balance import (
    plasma_energy_residual,
    total_heat_flux_to_wall,
    vapor_energy_residual,
)
from deepiri_fuselk.physics.pde_solver import (
    SolverResult,
    solve_oil_water_steady,
    solve_oil_water_transient,
)
from deepiri_fuselk.physics.pde_system import (
    PDEParameters,
    PDEState,
    interface_thickness,
    peclet_criterion,
)

__all__ = [
    "PDEParameters",
    "PDEState",
    "SolverResult",
    "interface_thickness",
    "peclet_criterion",
    "plasma_energy_residual",
    "solve_oil_water_steady",
    "solve_oil_water_transient",
    "total_heat_flux_to_wall",
    "vapor_energy_residual",
]
