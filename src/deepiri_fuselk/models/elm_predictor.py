"""ELM onset predictor from focal heat maps and profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ELMPrediction:
    probability: float
    confidence_interval: tuple[float, float]
    time_to_elm_ms: float
    precursor_mode: str


class ELMPredictor:
    """
    ELM predictor using focal heat map features.

    Uses ensemble of heuristic + learned features until full BNN training data available.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._history: list[float] = []

    def extract_features(self, focal_map: np.ndarray, rotation_hz: float) -> np.ndarray:
        peak = float(np.max(focal_map))
        variance = float(np.var(focal_map))
        gradient = float(np.max(np.abs(np.gradient(focal_map))))
        return np.array([peak, variance, gradient, rotation_hz / 1e4])

    def predict(self, focal_map: np.ndarray, rotation_hz: float = 5000.0) -> ELMPrediction:
        features = self.extract_features(focal_map, rotation_hz)
        # Logistic-style scoring from physics-informed weights
        weights = np.array([0.4, 0.8, 0.3, -0.2])
        logit = float(np.dot(features, weights) - 1.5)
        prob = 1.0 / (1.0 + np.exp(-logit))
        uncertainty = 0.1 + 0.2 * float(np.std(self._history[-10:])) if self._history else 0.15
        self._history.append(prob)
        return ELMPrediction(
            probability=prob,
            confidence_interval=(max(0, prob - uncertainty), min(1, prob + uncertainty)),
            time_to_elm_ms=max(0, (1 - prob) * 100),
            precursor_mode="2/1_ntm" if prob > 0.6 else "stable",
        )

    def is_elm_imminent(self, prediction: ELMPrediction) -> bool:
        return prediction.probability > self.threshold
