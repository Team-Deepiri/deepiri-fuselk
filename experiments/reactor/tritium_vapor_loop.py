"""Reactor experiment: tritium vapor extraction loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepiri_fuselk.barrier.breeding_blanket import evaluate_breeding_blanket
from deepiri_fuselk.physics.pde_solver import solve_oil_water_steady
from deepiri_fuselk.physics.pde_system import PDEParameters, peclet_criterion


@dataclass
class TritiumLoopResult:
    tbr: float
    peclet: float
    extraction_ok: bool
    outward_flux: float


def run_tritium_extraction_sweep(
    v_v_range: tuple[float, float] = (0.1, 2.0),
    n_steps: int = 20,
) -> list[TritiumLoopResult]:
    """Sweep vapor injection velocity for tritium extraction viability."""
    results = []
    for v_v in np.linspace(v_v_range[0], v_v_range[1], n_steps):
        params = PDEParameters(v_v=float(v_v))
        solved = solve_oil_water_steady(n_grid=32, max_iter=300, params=params)
        br = evaluate_breeding_blanket(solved.state, params)
        pe = peclet_criterion(params)
        results.append(
            TritiumLoopResult(
                tbr=br.tritium_breeding_ratio,
                peclet=pe,
                extraction_ok=pe > 1.0,
                outward_flux=br.outward_flux,
            )
        )
    return results
