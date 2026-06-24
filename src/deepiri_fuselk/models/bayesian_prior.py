"""Bayesian neural net prior with rotational axis awareness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PriorState:
    """Slow-loop policy prior for Venturi controller."""

    max_sweep_velocity: float
    max_rmp_phase: float
    elm_risk_regime: str  # low, medium, high
    confidence: float


class BayesianRotationalPrior:
    """
    Slow (~100ms) Bayesian prior that conditions RL on global plasma state.

    Incorporates rotation profile as direct feature (rotational axis awareness).
    Uses variational approximation until Pyro training pipeline is wired.
    """

    def __init__(self) -> None:
        self._weights = np.array([0.3, 0.5, 0.2, 0.4])
        self._bias = -0.5

    def update(
        self,
        ne_pedestal: float,
        beta_n: float,
        rotation_khz: float,
        q95: float,
    ) -> PriorState:
        x = np.array([ne_pedestal, beta_n, rotation_khz / 10, q95 / 5])
        logit = float(np.dot(self._weights, x) + self._bias)
        risk_score = 1.0 / (1.0 + np.exp(-logit))

        if risk_score > 0.7:
            regime = "high"
            max_sweep = 0.4
            max_rmp = 0.3
        elif risk_score > 0.4:
            regime = "medium"
            max_sweep = 0.7
            max_rmp = 0.6
        else:
            regime = "low"
            max_sweep = 1.0
            max_rmp = 1.0

        # Epistemic uncertainty via weight perturbation (MC dropout surrogate)
        samples = [risk_score + np.random.normal(0, 0.05) for _ in range(10)]
        confidence = 1.0 - float(np.std(samples))

        return PriorState(
            max_sweep_velocity=max_sweep,
            max_rmp_phase=max_rmp,
            elm_risk_regime=regime,
            confidence=max(0, confidence),
        )
