"""Fusion reactor key performance indicators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FusionKPIs:
    """Standard metrics for fuselk reactor cell assessment."""

    tritium_breeding_ratio: float
    elm_free_fraction: float
    divertor_uniformity: float
    disruption_risk: float
    muon_gain: float
    helix_snr_mean: float
    venturi_mean_reward: float
    q_min: float
    beta_n: float

    def score(self) -> float:
        """Composite fusion progress score in [0, 1]."""
        tbr = min(1.0, self.tritium_breeding_ratio / 1.05)
        elm = self.elm_free_fraction
        div = self.divertor_uniformity
        dis = 1.0 - self.disruption_risk
        muon = min(1.0, self.muon_gain / 284.0)
        snr = min(1.0, self.helix_snr_mean / 5.0)
        return float(0.25 * tbr + 0.2 * elm + 0.15 * div + 0.2 * dis + 0.1 * muon + 0.1 * snr)


def divertor_uniformity(heat_flux: np.ndarray) -> float:
    """1 = perfectly uniform, 0 = single hotspot."""
    if heat_flux.size == 0:
        return 0.0
    peak = float(np.max(heat_flux))
    mean = float(np.mean(heat_flux))
    if peak < 1e-9:
        return 1.0
    return float(np.clip(mean / peak, 0.0, 1.0))


def elm_free_fraction(elm_probs: list[float], threshold: float = 0.5) -> float:
    if not elm_probs:
        return 1.0
    safe = sum(1 for p in elm_probs if p < threshold)
    return safe / len(elm_probs)


def mhd_stability_margin(q_min: float, beta_n: float, q_limit: float = 2.0, beta_limit: float = 3.0) -> float:
    """Disruption risk from q_min and normalized beta (0=safe, 1=imminent)."""
    q_risk = max(0.0, (q_limit - q_min) / q_limit) if q_min < q_limit else 0.0
    beta_risk = max(0.0, (beta_n - beta_limit * 0.8) / (beta_limit * 0.2))
    return float(np.clip(0.6 * q_risk + 0.4 * beta_risk, 0.0, 1.0))
