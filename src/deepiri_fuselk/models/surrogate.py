"""Reduced-order plasma surrogate for fast digital twin stepping."""

from __future__ import annotations

import numpy as np


class PlasmaSurrogate:
    """Fast surrogate mapping control actions to profile evolution."""

    def __init__(self, n_radial: int = 32) -> None:
        self.n_radial = n_radial
        self.rho = np.linspace(0, 1, n_radial)
        self.ne = 1e19 * (1 - self.rho)
        self.Te = 5e3 * (1 - 0.8 * self.rho)

    def step(
        self, sweep_velocity: float, rmp_phase: float, dt: float = 1e-3
    ) -> dict[str, np.ndarray]:
        """Evolve profiles under control action."""
        perturbation = 0.01 * sweep_velocity * np.sin(rmp_phase * 2 * np.pi * self.rho)
        self.ne = np.clip(self.ne * (1 + perturbation), 0, None)
        self.Te = np.clip(self.Te * (1 - 0.005 * sweep_velocity), 100, None)
        return {"ne": self.ne.copy(), "Te": self.Te.copy(), "rho": self.rho.copy()}
