"""Lab experiment: brine wall coating simulation."""

from __future__ import annotations

import numpy as np

from deepiri_fuselk.barrier.heat_exhaust import BrineCoatingResult, evaluate_brine_coating


def run_brine_sweep(
    salinity_range: tuple[float, float] = (10, 50),
    n_steps: int = 10,
) -> list[BrineCoatingResult]:
    """Sweep salinity parameter space for coating viability."""
    results = []
    for sal in np.linspace(salinity_range[0], salinity_range[1], n_steps):
        results.append(evaluate_brine_coating(salinity_ppt=float(sal)))
    return results


def best_salinity(results: list[BrineCoatingResult]) -> float | None:
    viable = [r for r in results if r.viable]
    if not viable:
        return None
    return float(max(viable, key=lambda r: r.effective_heat_reduction).effective_heat_reduction)
