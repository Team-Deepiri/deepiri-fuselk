"""Labeled synthetic shot corpus for ELM / disruption model training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepiri_fuselk.helix.helix_engine import HelixEngine
from deepiri_fuselk.sim.synthetic_data_gen import generate_ece_shot


@dataclass
class LabeledFrame:
    focal_map: np.ndarray
    features: np.ndarray
    elm_label: float  # 1 = ELM imminent, 0 = stable
    island_amplitude: float
    rotation_hz: float


@dataclass
class ShotCorpus:
    frames: list[LabeledFrame]
    elm_rate: float

    def features_matrix(self) -> np.ndarray:
        return np.array([f.features for f in self.frames])

    def labels(self) -> np.ndarray:
        return np.array([f.elm_label for f in self.frames])


def generate_corpus(
    n_shots: int = 200,
    grid_size: int = 32,
    seed: int = 0,
) -> ShotCorpus:
    """Build labeled dataset: high island amplitude -> ELM label."""
    rng = np.random.default_rng(seed)
    engine = HelixEngine()
    frames: list[LabeledFrame] = []

    for i in range(n_shots):
        amp = float(rng.uniform(0.05, 1.2))
        shot = generate_ece_shot(grid_size, seed=seed + i, island_amplitude=amp)
        result = engine.process(shot.heat_field, shot.raw_signal, shot.angles)
        label = 1.0 if amp > 0.55 else 0.0
        feat = _extract_training_features(
            result.focal_map, result.phase_locked_snr, result.rotation_hz
        )
        frames.append(
            LabeledFrame(
                focal_map=result.focal_map,
                features=feat,
                elm_label=label,
                island_amplitude=amp,
                rotation_hz=result.rotation_hz,
            )
        )

    elm_rate = float(np.mean([f.elm_label for f in frames]))
    return ShotCorpus(frames=frames, elm_rate=elm_rate)


def _extract_training_features(
    focal_map: np.ndarray,
    snr: float,
    rotation_hz: float,
) -> np.ndarray:
    gy, gx = np.gradient(focal_map)
    grad_mag = float(np.max(np.sqrt(gx**2 + gy**2)))
    peak = float(np.max(focal_map))
    var = float(np.var(focal_map))
    width = float(np.sum(focal_map > 0.5 * max(peak, 1e-9)) / max(focal_map.size, 1))
    return np.array([peak, var, grad_mag, width, snr, rotation_hz / 1e4], dtype=np.float64)
