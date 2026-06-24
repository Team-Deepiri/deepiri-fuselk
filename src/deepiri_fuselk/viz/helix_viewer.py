"""HELIX viewer utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepiri_fuselk.helix.helical_quadtree import HQRMResult


@dataclass
class HelixView:
    o_point: tuple[float, float]
    rotation_deg: float
    converged: bool
    reticle_html: str


def build_helix_view(hqrm: HQRMResult, rotation_hz: float = 5000.0) -> HelixView:
    """Build HELIX viewer state with O-point reticle."""
    ox, oy = hqrm.o_point
    reticle = (
        f'<div style="position:absolute;left:{ox}%;top:{oy}%;'
        f'width:12px;height:12px;border:2px solid red;border-radius:50%"></div>'
    )
    return HelixView(
        o_point=hqrm.o_point,
        rotation_deg=float(rotation_hz * 360 / 1e6),
        converged=hqrm.converged,
        reticle_html=reticle,
    )


def torus_cross_section(n: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Generate tokamak cross-section mesh for Three.js embedding."""
    theta = np.linspace(0, 2 * np.pi, n)
    R0, a = 3.0, 1.0
    R = R0 + a * np.cos(theta)
    Z = a * np.sin(theta)
    return R, Z
