"""Trainable ELM onset predictor with online logistic regression."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ELMPrediction:
    probability: float
    confidence_interval: tuple[float, float]
    time_to_elm_ms: float
    precursor_mode: str


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class ELMPredictor:
    """
    ELM predictor: physics-informed features + trainable logistic weights.

    Train on LabeledFrame corpus from sim.shot_corpus.generate_corpus().
    """

    FEATURE_NAMES = ("peak", "variance", "gradient", "width", "snr", "rotation")

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.weights = np.array([0.35, 0.7, 0.45, 0.5, 0.15, -0.2], dtype=np.float64)
        self.bias = -1.2
        self._feat_mean: np.ndarray | None = None
        self._feat_std: np.ndarray | None = None
        self._history: list[float] = []
        self._train_accuracy: float | None = None

    def extract_features(self, focal_map: np.ndarray, rotation_hz: float = 5000.0) -> np.ndarray:
        gy, gx = np.gradient(focal_map)
        grad_mag = float(np.max(np.sqrt(gx**2 + gy**2)))
        peak = float(np.max(focal_map))
        var = float(np.var(focal_map))
        width = float(np.sum(focal_map > 0.5 * max(peak, 1e-9)) / max(focal_map.size, 1))
        snr = peak / max(float(np.std(focal_map)), 1e-9)
        return np.array([peak, var, grad_mag, width, snr, rotation_hz / 1e4], dtype=np.float64)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self._feat_mean is None or self._feat_std is None:
            return X
        return (X - self._feat_mean) / self._feat_std

    def predict(self, focal_map: np.ndarray, rotation_hz: float = 5000.0) -> ELMPrediction:
        features = self.extract_features(focal_map, rotation_hz)
        if self._feat_mean is not None:
            features = self._normalize(features.reshape(1, -1))[0]
        logit = float(np.dot(self.weights, features) + self.bias)
        prob = float(_sigmoid(np.array([logit]))[0])
        uncertainty = 0.08 + 0.15 * float(np.std(self._history[-20:])) if self._history else 0.12
        self._history.append(prob)
        mode = "2/1_ntm" if prob > 0.65 else "1/1_kink" if prob > 0.45 else "stable"
        return ELMPrediction(
            probability=prob,
            confidence_interval=(max(0, prob - uncertainty), min(1, prob + uncertainty)),
            time_to_elm_ms=max(0, (1 - prob) * 80),
            precursor_mode=mode,
        )

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 800,
        lr: float = 0.1,
        l2: float = 1e-4,
    ) -> float:
        """Batch gradient descent logistic regression with feature normalization."""
        self._feat_mean = X.mean(axis=0)
        self._feat_std = X.std(axis=0) + 1e-9
        Xn = self._normalize(X)
        w = self.weights.copy()
        b = self.bias
        n = len(y)
        for _ in range(epochs):
            logits = Xn @ w + b
            preds = _sigmoid(logits)
            error = preds - y
            grad_w = (Xn.T @ error) / n + l2 * w
            grad_b = float(np.mean(error))
            w -= lr * grad_w
            b -= lr * grad_b
        self.weights = w
        self.bias = b
        acc = float(np.mean((_sigmoid(Xn @ w + b) >= 0.5) == (y >= 0.5)))
        self._train_accuracy = acc
        return acc

    def train_from_corpus(self, corpus) -> float:
        return self.train(corpus.features_matrix(), corpus.labels())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "threshold": self.threshold,
            "train_accuracy": self._train_accuracy,
            "feat_mean": self._feat_mean.tolist() if self._feat_mean is not None else None,
            "feat_std": self._feat_std.tolist() if self._feat_std is not None else None,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ELMPredictor:
        data = json.loads(Path(path).read_text())
        model = cls(threshold=data.get("threshold", 0.5))
        model.weights = np.array(data["weights"], dtype=np.float64)
        model.bias = float(data["bias"])
        model._train_accuracy = data.get("train_accuracy")
        if data.get("feat_mean") is not None:
            model._feat_mean = np.array(data["feat_mean"], dtype=np.float64)
            model._feat_std = np.array(data["feat_std"], dtype=np.float64)
        return model

    def is_elm_imminent(self, prediction: ELMPrediction) -> bool:
        return prediction.probability > self.threshold

    @property
    def train_accuracy(self) -> float | None:
        return self._train_accuracy
