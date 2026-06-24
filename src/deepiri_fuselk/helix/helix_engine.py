"""Unified HELIX Engine pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepiri_fuselk.focal.focal_heatmap import focal_heatmap, from_hqrm, singularity_gradient
from deepiri_fuselk.focal.lockin_amplifier import subtract_incoherent_noise
from deepiri_fuselk.focal.spiral_attention import apply_spiral_attention
from deepiri_fuselk.helix.coordinate_mapper import boozer_map, field_line_pitch
from deepiri_fuselk.helix.helical_quadtree import HQRMResult, run_hqrm
from deepiri_fuselk.helix.kalman_tracker import PhaseLockedTracker


@dataclass
class HelixResult:
    """Complete HELIX pipeline output."""

    raw_heatmap: np.ndarray
    focal_map: np.ndarray
    hqrm: HQRMResult
    o_point: tuple[float, float]
    fracture_vector: tuple[float, float]
    elm_probability: float
    rotation_hz: float
    phase_locked_snr: float


class HelixEngine:
    """
    Helical Extraction & Locked-In Isotropy eXclusion.

    Pipeline:
      1. Phase-locked O-point tracking (Kalman)
      2. Boozer coordinate unwrap
      3. HQRM 7x7 kernel lock
      4. Spiral attention denoising
      5. Focal singularity heat map
    """

    def __init__(
        self,
        rotation_hz: float = 5000.0,
        shear_threshold: float = 0.3,
        variance_threshold: float = 0.07,
    ) -> None:
        self.tracker = PhaseLockedTracker(omega_init=rotation_hz)
        self.rotation_hz = rotation_hz
        self.shear_threshold = shear_threshold
        self.variance_threshold = variance_threshold
        self._snr_history: list[float] = []

    def process(
        self,
        heat_field: np.ndarray,
        raw_signal: np.ndarray,
        angles: np.ndarray,
        dt: float = 1e-4,
    ) -> HelixResult:
        """Run full HELIX denoising and focal mapping pipeline."""
        self.tracker.predict(dt)
        cleaned = subtract_incoherent_noise(raw_signal, angles, self.tracker.state.omega)
        peak_idx = int(np.argmax(np.abs(cleaned)))
        self.tracker.update(
            float(angles[peak_idx]),
            float(angles[peak_idx % len(angles)] * 0.5),
            float(np.max(heat_field)),
        )
        est_omega = self.tracker.estimate_rotation_from_signal(cleaned, angles)
        self.tracker.x[1] = 0.9 * self.tracker.x[1] + 0.1 * est_omega
        center = self.tracker.sample_at_phase(cleaned, angles)
        noise_floor = float(np.std(raw_signal))
        snr = abs(center) / max(noise_floor, 1e-9)
        self._snr_history.append(snr)

        # Boozer unwrap for pitch
        n = heat_field.shape[0]
        grid = np.linspace(-1, 1, n)
        R, Z = np.meshgrid(grid, grid)
        _theta, _phi = boozer_map(R.ravel(), Z.ravel())
        pitch = float(np.mean(field_line_pitch(R.ravel(), Z.ravel())))

        # HQRM lock
        hqrm = run_hqrm(
            heat_field,
            shear_threshold=self.shear_threshold,
            variance_threshold=self.variance_threshold,
        )

        # Spiral attention + focal map
        attended = apply_spiral_attention(heat_field, pitch=pitch)
        focal = (
            from_hqrm(hqrm, size=n)
            if hqrm.converged
            else focal_heatmap(raw_signal, angles, self.tracker, size=n)
        )
        focal = 0.7 * focal + 0.3 * attended

        fracture = singularity_gradient(focal)
        elm_p = min(1.0, hqrm.heat_variance * 2.0 + (1.0 / max(snr, 0.1)) * 0.1)

        return HelixResult(
            raw_heatmap=heat_field,
            focal_map=focal,
            hqrm=hqrm,
            o_point=hqrm.o_point,
            fracture_vector=fracture,
            elm_probability=elm_p,
            rotation_hz=self.rotation_hz,
            phase_locked_snr=snr,
        )

    @property
    def mean_snr_gain(self) -> float:
        if len(self._snr_history) < 2:
            return 1.0
        return float(np.mean(self._snr_history))
