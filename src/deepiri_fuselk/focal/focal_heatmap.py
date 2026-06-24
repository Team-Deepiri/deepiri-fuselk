"""Focal singularity heat map generation."""

from __future__ import annotations

import numpy as np

from deepiri_fuselk.helix.helical_quadtree import HQRMResult
from deepiri_fuselk.helix.kalman_tracker import PhaseLockedTracker


def focal_heatmap(
    raw_signal: np.ndarray,
    angles: np.ndarray,
    tracker: PhaseLockedTracker | None = None,
    size: int = 32,
) -> np.ndarray:
    """Generate noise-reduced focal heat map centered on tracked O-point."""
    tracker = tracker or PhaseLockedTracker()
    tracker.predict(dt=1e-4)
    center_val = tracker.sample_at_phase(raw_signal, angles)

    grid = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(grid, grid)
    r2 = X**2 + Y**2
    base = center_val * np.exp(-r2 / 0.3)
    signal_grid = np.resize(raw_signal, size)
    heatmap = base + 0.1 * np.outer(signal_grid, signal_grid)
    return heatmap.astype(np.float64)


def singularity_gradient(heatmap: np.ndarray) -> tuple[float, float]:
    """Compute gradient direction at heatmap peak (fracture vector)."""
    iy, ix = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    gy, gx = np.gradient(heatmap)
    return float(gx[iy, ix]), float(gy[iy, ix])


def from_hqrm(hqrm: HQRMResult, size: int = 32) -> np.ndarray:
    """Build focal map from HQRM O-point lock."""
    grid = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(grid, grid)
    ox, oy = hqrm.o_point
    r2 = (X - ox) ** 2 + (Y - oy) ** 2
    amplitude = 1.0 / max(hqrm.heat_variance, 1e-6)
    return amplitude * np.exp(-r2 / 0.2)
