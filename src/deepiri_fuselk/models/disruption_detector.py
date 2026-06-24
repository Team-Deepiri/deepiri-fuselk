"""Unified disruption precursor detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepiri_fuselk.helix.helix_engine import HelixResult
from deepiri_fuselk.models.elm_predictor import ELMPrediction, ELMPredictor
from deepiri_fuselk.sim.fusion_kpis import mhd_stability_margin


@dataclass
class DisruptionAssessment:
    probability: float
    elm: ELMPrediction
    mhd_risk: float
    helix_snr: float
    recommended_action: str
    time_to_disruption_ms: float


class DisruptionDetector:
    """
    Fuse HELIX diagnostics, trainable ELM predictor, and MHD stability margins.

    recommended_action drives Venturi / pellet / gas puff decisions.
    """

    def __init__(self, elm_model: ELMPredictor | None = None) -> None:
        self.elm = elm_model or ELMPredictor()

    def assess(
        self,
        helix: HelixResult,
        q_min: float = 2.5,
        beta_n: float = 2.0,
    ) -> DisruptionAssessment:
        elm_pred = self.elm.predict(helix.focal_map, rotation_hz=helix.rotation_hz)
        mhd = mhd_stability_margin(q_min, beta_n)
        helix_risk = min(
            1.0, helix.elm_probability + (1.0 / max(helix.phase_locked_snr, 0.1)) * 0.05
        )
        combined = float(np.clip(0.45 * elm_pred.probability + 0.35 * mhd + 0.2 * helix_risk, 0, 1))

        if combined > 0.85:
            action = "pellet_inject"
        elif combined > 0.65:
            action = "rmp_phase_shift"
        elif combined > 0.45:
            action = "gas_puff_radiate"
        elif combined > 0.25:
            action = "strike_point_sweep"
        else:
            action = "hold"

        ttd = min(elm_pred.time_to_elm_ms, max(0, (1 - combined) * 120))

        return DisruptionAssessment(
            probability=combined,
            elm=elm_pred,
            mhd_risk=mhd,
            helix_snr=helix.phase_locked_snr,
            recommended_action=action,
            time_to_disruption_ms=ttd,
        )
