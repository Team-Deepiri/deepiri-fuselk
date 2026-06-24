"""Domain randomization for robust RL training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepiri_fuselk.physics.pde_system import PDEParameters


@dataclass
class RandomizedDomain:
    params: PDEParameters
    q0: float
    q95: float
    rotation_hz: float
    noise_level: float


def randomize_domain(seed: int | None = None) -> RandomizedDomain:
    """Sample randomized plasma domain parameters."""
    rng = np.random.default_rng(seed)
    return RandomizedDomain(
        params=PDEParameters(
            D_p=rng.uniform(0.01, 0.1),
            D_v=rng.uniform(0.05, 0.2),
            v_v=rng.uniform(0.2, 1.0),
            alpha=rng.uniform(0.5, 2.0),
        ),
        q0=rng.uniform(0.8, 1.5),
        q95=rng.uniform(2.5, 5.0),
        rotation_hz=rng.uniform(1e3, 1e4),
        noise_level=rng.uniform(0.1, 0.5),
    )
