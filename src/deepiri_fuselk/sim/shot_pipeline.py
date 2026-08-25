"""Per-shot pipeline aligned with VISION.md: diagnostics → HELIX → control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepiri_fuselk.control.policy_runner import HybridControlResult, HybridPolicyRunner
from deepiri_fuselk.data.imas_loader import IMASShot
from deepiri_fuselk.data.notebook_loaders import imas_to_synthetic_shot
from deepiri_fuselk.helix.helix_engine import HelixEngine, HelixResult
from deepiri_fuselk.models.disruption_detector import DisruptionAssessment, DisruptionDetector
from deepiri_fuselk.sim.synthetic_data_gen import SyntheticShot, generate_ece_shot


@dataclass(frozen=True)
class ShotPipelineResult:
    """Single timestep through the live fusion control stack."""

    shot: SyntheticShot
    helix: HelixResult
    disruption: DisruptionAssessment
    control: HybridControlResult
    q_min: float
    beta_n: float
    recommended_action: str
    source: str = "synthetic"  # "synthetic" | "imas"


class ShotPipeline:
    """
    VISION §1–5 fast path: ECE/SXR → HELIX → disruption fuse → Venturi/RL.

    Shared by ReactorCell, DigitalTwin, and dashboard simulators.
    When ``imas`` is provided, HELIX runs on the archived heat field instead
    of generating a fresh synthetic ECE shot each step.
    """

    def __init__(
        self,
        helix: HelixEngine,
        detector: DisruptionDetector,
        hybrid: HybridPolicyRunner,
    ) -> None:
        self.helix = helix
        self.detector = detector
        self.hybrid = hybrid

    def process(
        self,
        *,
        grid_size: int,
        seed: int,
        q_profile_values: np.ndarray,
        te_profile_values: np.ndarray,
        rl_steps: int = 1,
        imas: IMASShot | None = None,
        heat_field: np.ndarray | None = None,
        island_amplitude: float | None = None,
    ) -> ShotPipelineResult:
        if imas is not None:
            shot = imas_to_synthetic_shot(imas)
            # Resize heat if grid_size differs from archive
            if shot.heat_field.shape[0] != grid_size:
                shot = _resize_shot(shot, grid_size)
            source = "imas"
            q_vals = np.array(imas.q_profile.values, dtype=np.float64)
            te_vals = np.array(imas.Te_profile.values, dtype=np.float64)
            if len(q_vals):
                q_profile_values = q_vals
            if len(te_vals):
                te_profile_values = te_vals
        elif heat_field is not None:
            size = heat_field.shape[0]
            angles = np.linspace(0, 2 * np.pi, size, endpoint=False)
            raw = heat_field.mean(axis=1).astype(np.float64)
            amp = island_amplitude if island_amplitude is not None else 0.5
            shot = SyntheticShot(
                heat_field=heat_field.astype(np.float64),
                raw_signal=raw,
                angles=angles,
                island_amplitude=amp,
            )
            if size != grid_size:
                shot = _resize_shot(shot, grid_size)
            source = "imas"
        else:
            amp = island_amplitude if island_amplitude is not None else 0.5
            shot = generate_ece_shot(grid_size, seed=seed, island_amplitude=amp)
            source = "synthetic"

        helix = self.helix.process(shot.heat_field, shot.raw_signal, shot.angles)

        q_min = float(np.min(q_profile_values)) if len(q_profile_values) else 2.0
        beta_n = float(np.mean(te_profile_values)) / 2000.0 if len(te_profile_values) else 2.0

        disruption = self.detector.assess(helix, q_min=q_min, beta_n=beta_n)
        control = self.hybrid.step(
            shot.heat_field,
            elm_probability=disruption.probability,
            rl_steps=rl_steps,
        )

        action = disruption.recommended_action
        if action == "pellet_inject":
            control.venturi.action.pellet_ready = True
        elif action == "gas_puff_radiate":
            control.venturi.action.gas_puff = 0.8

        return ShotPipelineResult(
            shot=shot,
            helix=helix,
            disruption=disruption,
            control=control,
            q_min=q_min,
            beta_n=beta_n,
            recommended_action=action,
            source=source,
        )


def _resize_shot(shot: SyntheticShot, grid_size: int) -> SyntheticShot:
    """Nearest-neighbor resize of heat field to match reactor grid."""
    from scipy.ndimage import zoom

    h = shot.heat_field
    if h.shape[0] == grid_size:
        return shot
    factor = grid_size / h.shape[0]
    resized = zoom(h, factor, order=1)
    angles = np.linspace(0, 2 * np.pi, grid_size, endpoint=False)
    raw = resized.mean(axis=1).astype(np.float64)
    return SyntheticShot(
        heat_field=resized.astype(np.float64),
        raw_signal=raw,
        angles=angles,
        island_amplitude=shot.island_amplitude,
    )
