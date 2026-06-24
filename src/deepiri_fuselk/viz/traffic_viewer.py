"""Traffic flow visualization helpers."""

from __future__ import annotations

import numpy as np


def traffic_arrows(heat_flux: np.ndarray, threshold: float = 0.5) -> list[dict]:
    """Generate arrow descriptors for divertor traffic visualization."""
    arrows = []
    h, w = heat_flux.shape
    for iy in range(0, h, 4):
        for ix in range(0, w, 4):
            val = heat_flux[iy, ix]
            if val > threshold:
                arrows.append(
                    {
                        "x": ix,
                        "y": iy,
                        "dx": 0,
                        "dy": -val,
                        "magnitude": float(val),
                    }
                )
    return arrows
