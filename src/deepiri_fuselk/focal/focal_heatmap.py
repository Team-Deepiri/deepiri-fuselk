"""Focal singularity heat map generation with lock-in denoising."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from deepiri_fuselk.focal.lockin_amplifier import lockin_demodulate, subtract_incoherent_noise
from deepiri_fuselk.helix.kalman_tracker import PhaseLockedTracker

if TYPE_CHECKING:
    from deepiri_fuselk.helix.helical_quadtree import HQRMResult


def focal_heatmap(
    raw_signal: np.ndarray,
    angles: np.ndarray,
    tracker: PhaseLockedTracker | None = None,
    size: int = 32,
) -> np.ndarray:
    """Generate noise-reduced focal heat map via lock-in + Gaussian O-point kernel."""
    tracker = tracker or PhaseLockedTracker()
    tracker.predict(dt=1e-4)

    cleaned = subtract_incoherent_noise(raw_signal, angles, tracker.state.omega)
    amp, _, _ = lockin_demodulate(cleaned, tracker.state.theta, angles)
    center_val = tracker.sample_at_phase(cleaned, angles)

    grid = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(grid, grid)
    r2 = (X - 0.3 * np.cos(tracker.state.theta)) ** 2 + (
        Y - 0.3 * np.sin(tracker.state.theta)
    ) ** 2
    base = max(abs(center_val), amp) * np.exp(-r2 / 0.25)
    signal_grid = np.resize(cleaned, size)
    return (base + 0.05 * np.outer(signal_grid, signal_grid)).astype(np.float64)


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
