"""SOLPS-ITER / BOUT++ wrapper stubs for high-fidelity coupling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SOLPSConfig:
    grid_points: int = 64
    dt: float = 1e-5
    diffusion_coeff: float = 0.5


@dataclass
class SOLPSResult:
    density: np.ndarray
    temperature: np.ndarray
    heat_flux: np.ndarray


class SOLPSWrapper:
    """Stub wrapper for SOLPS-ITER edge plasma code."""

    def __init__(self, config: SOLPSConfig | None = None) -> None:
        self.config = config or SOLPSConfig()

    def run_step(self, boundary_condition: np.ndarray | None = None) -> SOLPSResult:
        n = self.config.grid_points
        x = np.linspace(0, 1, n)
        ne = 1e19 * np.exp(-3 * x)
        Te = 100 * (1 - x)
        q = self.config.diffusion_coeff * np.gradient(Te)
        if boundary_condition is not None:
            ne = ne * (1 + 0.1 * boundary_condition[:n])
        return SOLPSResult(density=ne, temperature=Te, heat_flux=np.abs(q))

    def available(self) -> bool:
        return False  # Set True when real SOLPS binary is linked
