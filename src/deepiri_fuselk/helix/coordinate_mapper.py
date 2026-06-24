"""Boozer / field-line coordinate mapping."""

from __future__ import annotations

import numpy as np


def q_profile(r: np.ndarray, q0: float = 1.0, q95: float = 3.5, a: float = 1.0) -> np.ndarray:
    """Monotonic safety factor profile q(r)."""
    return q0 + (q95 - q0) * (r / a) ** 2


def boozer_map(
    R: np.ndarray,
    Z: np.ndarray,
    r_minor: float = 0.5,
    q0: float = 1.0,
    q95: float = 3.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Map (R, Z) poloidal plane to (theta, phi) Boozer-like coordinates."""
    r = np.sqrt(R**2 + Z**2)
    theta = np.arctan2(Z, R)
    q = q_profile(r, q0=q0, q95=q95, a=r_minor)
    phi = theta / np.maximum(q, 1e-6)
    return theta, phi


def field_line_pitch(R: np.ndarray, Z: np.ndarray, q0: float = 1.0, q95: float = 3.5) -> np.ndarray:
    """Local field line pitch angle for HQRM rotation."""
    r = np.sqrt(R**2 + Z**2)
    q = q_profile(r, q0=q0, q95=q95)
    return np.arctan2(1.0, q)
