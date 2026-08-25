"""Runaway / disruption filament geometry from field-line pitch.

Phase 9 of PORT_VIEW_RADIANCE_ROADMAP: disruption visuals follow the
device's magnetic pitch, not a generic radial flash.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepiri_fuselk.devices.profile import DeviceProfile
from deepiri_fuselk.helix.coordinate_mapper import field_line_pitch, q_profile


@dataclass(frozen=True)
class DisruptionFilament:
    """Direction and length of a runaway-electron-like beam in port-view space."""

    pitch_rad: float
    direction: tuple[float, float, float]
    length: float
    q95: float
    device: str

    def to_dict(self) -> dict:
        return {
            "pitch_rad": self.pitch_rad,
            "direction": list(self.direction),
            "length": self.length,
            "q95": self.q95,
            "device": self.device,
        }


def midplane_pitch_rad(
    device: DeviceProfile,
    *,
    q0: float = 1.0,
    q95: float | None = None,
) -> float:
    """Field-line pitch at mid-radius for the active device."""
    a = max(device.minor_radius_m, 1e-3)
    q_edge = float(q95) if q95 is not None else 3.0 + 0.5 * device.aspect_ratio
    r = np.array([0.5 * a])
    z = np.zeros(1)
    # Normalize to unit minor radius so pitch depends on q-shape, not metres.
    pitch = field_line_pitch(r / a, z, q0=q0, q95=q_edge, a=1.0)
    return float(pitch[0])


def disruption_filament(
    device: DeviceProfile,
    *,
    q95: float | None = None,
    fracture_vector: tuple[float, float] | None = None,
) -> DisruptionFilament:
    """
    Build a filament aligned with field-line pitch.

    ``fracture_vector`` (HELIX) biases the horizontal aim when present;
    otherwise the beam follows (cos θ, sin θ, 0.15) in view coordinates.
    """
    pitch = midplane_pitch_rad(device, q95=q95)
    q_edge = (
        float(q95)
        if q95 is not None
        else float(
            q_profile(np.array([1.0]), q0=1.0, q95=3.0 + 0.5 * device.aspect_ratio, a=1.0)[0]
        )
    )
    # Toroidal-ish beam: pitch tilts poloidal vs toroidal components.
    fx, fy = (0.0, 0.0)
    if fracture_vector is not None:
        fx, fy = float(fracture_vector[0]), float(fracture_vector[1])
        norm = max(np.hypot(fx, fy), 1e-9)
        fx, fy = fx / norm, fy / norm
    # High-field / high-q devices → shallower pitch (more toroidal).
    dx = np.cos(pitch) + 0.25 * fx
    dy = np.sin(pitch) * 0.6 + 0.15 * fy
    dz = 0.35 + 0.1 * (device.max_bt_t / 5.0)
    vec = np.array([dx, dy, dz], dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    vec /= max(norm, 1e-9)
    # Length scales with minor radius so ITER filaments read longer than DIII-D.
    length = float(1.2 + 0.8 * device.minor_radius_m)
    return DisruptionFilament(
        pitch_rad=pitch,
        direction=(float(vec[0]), float(vec[1]), float(vec[2])),
        length=length,
        q95=q_edge,
        device=device.name,
    )
