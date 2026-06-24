"""Per-shot pipeline aligned with VISION.md: diagnostics → HELIX → control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepiri_fuselk.control.policy_runner import HybridControlResult, HybridPolicyRunner
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


class ShotPipeline:
    """
    VISION §1–5 fast path: ECE/SXR → HELIX → disruption fuse → Venturi/RL.

    Shared by ReactorCell, DigitalTwin, and dashboard simulators so diagnostics
    and actuation always refer to the same synthetic shot.
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
    ) -> ShotPipelineResult:
        shot = generate_ece_shot(grid_size, seed=seed)
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
        )
