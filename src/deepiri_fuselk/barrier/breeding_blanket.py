"""Oil-water vapor barrier and lithium breeding blanket models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepiri_fuselk.physics.pde_system import PDEParameters, PDEState


@dataclass
class BreedingResult:
    tritium_breeding_ratio: float
    extraction_efficiency: float
    outward_flux: float


def tritium_breeding_ratio(
    state: PDEState, params: PDEParameters, consumption: float = 1.0
) -> float:
    """Compute TBR = tritium produced / tritium consumed."""
    production = float(np.trapz(params.sigma_li * params.phi_n * state.n_v, state.x))
    return production / max(consumption, 1e-12)


def vapor_extraction_velocity(params: PDEParameters) -> float:
    """Required vapor injection speed for outward tritium sweep."""
    from deepiri_fuselk.physics.pde_system import interface_thickness

    delta = interface_thickness(params)
    return 1.1 * params.D_T / delta


def evaluate_breeding_blanket(state: PDEState, params: PDEParameters) -> BreedingResult:
    """Evaluate blanket performance metrics."""
    tbr = tritium_breeding_ratio(state, params)
    v_required = vapor_extraction_velocity(params)
    extraction = min(1.0, params.v_v / v_required)
    outward = float(state.n_T[-1])
    return BreedingResult(
        tritium_breeding_ratio=tbr,
        extraction_efficiency=extraction,
        outward_flux=outward,
    )
