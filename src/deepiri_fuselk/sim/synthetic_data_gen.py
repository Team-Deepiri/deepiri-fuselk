"""Synthetic diagnostic shot generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticShot:
    heat_field: np.ndarray
    raw_signal: np.ndarray
    angles: np.ndarray
    island_amplitude: float


def generate_ece_shot(
    size: int = 32,
    seed: int | None = None,
    island_amplitude: float = 0.5,
) -> SyntheticShot:
    """Generate synthetic ECE data with helical island precursor + turbulence noise."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(grid, grid)
    phase = rng.uniform(0, 2 * np.pi)
    island = island_amplitude * np.exp(
        -((X - 0.3 * np.cos(phase)) ** 2 + (Y - 0.3 * np.sin(phase)) ** 2) / 0.05
    )
    noise = 0.3 * rng.standard_normal((size, size))
    heat = np.maximum(island + noise, 0.0)
    angles = np.linspace(0, 2 * np.pi, size)
    raw = heat.mean(axis=1) + 0.1 * rng.standard_normal(size)
    return SyntheticShot(
        heat_field=heat.astype(np.float64),
        raw_signal=raw.astype(np.float64),
        angles=angles,
        island_amplitude=island_amplitude,
    )
