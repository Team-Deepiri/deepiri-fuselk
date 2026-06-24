"""Plasma traffic router — divertor heat flux state extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrafficState:
    heat_flux_map: np.ndarray
    peak_flux: float
    variance: float
    peak_location: tuple[int, int]


class PlasmaTrafficRouter:
    """Extract real-time divertor traffic state from heat flux maps."""

    def __init__(self, engineering_limit: float = 10.0) -> None:
        self.engineering_limit = engineering_limit

    def route(self, heat_flux: np.ndarray) -> TrafficState:
        peak = float(np.max(heat_flux))
        variance = float(np.var(heat_flux))
        iy, ix = np.unravel_index(int(np.argmax(heat_flux)), heat_flux.shape)
        return TrafficState(
            heat_flux_map=heat_flux,
            peak_flux=peak,
            variance=variance,
            peak_location=(int(iy), int(ix)),
        )

    def congestion_ratio(self, state: TrafficState) -> float:
        return state.peak_flux / self.engineering_limit
