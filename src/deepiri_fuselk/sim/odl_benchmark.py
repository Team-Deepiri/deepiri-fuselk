"""Public ODL (C-Mod) disruption / density-limit benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from deepiri_fuselk.data.notebook_loaders import (
    ensure_fetched_data,
    list_shots,
    load_odl_meta,
    resolve_data_root,
)
from deepiri_fuselk.sim.shot_replay import ShotReplayer


@dataclass
class ODLShotScore:
    shot_id: str
    odl_label_rate: float
    mean_p_dis: float
    mean_snr: float
    final_score: float
    auc_proxy: float  # P_dis vs label agreement proxy in [0, 1]


@dataclass
class ODLBenchmarkReport:
    n_shots: int
    mean_auc_proxy: float
    mean_p_dis: float
    correlation_label_pdis: float
    shots: list[ODLShotScore] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_shots": self.n_shots,
            "mean_auc_proxy": self.mean_auc_proxy,
            "mean_p_dis": self.mean_p_dis,
            "correlation_label_pdis": self.correlation_label_pdis,
            "shots": [
                {
                    "shot_id": s.shot_id,
                    "odl_label_rate": s.odl_label_rate,
                    "mean_p_dis": s.mean_p_dis,
                    "mean_snr": s.mean_snr,
                    "final_score": s.final_score,
                    "auc_proxy": s.auc_proxy,
                }
                for s in self.shots
            ],
        }


def _auc_proxy(labels: np.ndarray, probs: np.ndarray) -> float:
    """Simple ranking agreement: mean P(pos) − mean P(neg), mapped to [0, 1]."""
    labels = np.asarray(labels, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    if labels.size == 0 or probs.size == 0:
        return 0.5
    pos = probs[labels >= 0.5]
    neg = probs[labels < 0.5]
    if pos.size == 0 or neg.size == 0:
        # Degenerate — fall back to absolute error vs mean label
        return float(1.0 - abs(float(probs.mean()) - float(labels.mean())))
    delta = float(pos.mean() - neg.mean())
    return float(np.clip(0.5 + 0.5 * delta, 0.0, 1.0))


def run_odl_benchmark(
    data_root: Path | None = None,
    *,
    max_shots: int = 12,
    steps_per_shot: int = 12,
    ensure_data: bool = True,
) -> ODLBenchmarkReport:
    """
    Scrub fetched C-Mod ODL shots through ShotReplayer and score P_dis vs labels.

    Designed as a public, credential-free regression target for the control stack.
    """
    root = data_root or resolve_data_root()
    if ensure_data:
        root = ensure_fetched_data(root, n_shots=50, max_odl=max(max_shots, 20))

    paths = list_shots(root, source="cmod")[:max_shots]
    if not paths:
        raise FileNotFoundError(
            f"No CMOD_*.h5 shots under {root}/shots. Run: python scripts/fetch_data.py --all"
        )

    replayer = ShotReplayer(grid_size=32)
    scores: list[ODLShotScore] = []
    label_rates: list[float] = []
    pdis_means: list[float] = []

    for path in paths:
        result = replayer.scrub(path=path, n_steps=steps_per_shot)
        meta = load_odl_meta(path)
        labels = meta["density_limit_phase"] if meta is not None else np.zeros(len(result.frames))
        # Align labels to scrub length
        if len(labels) != len(result.frames) and len(labels) > 0:
            idx = np.linspace(0, len(labels) - 1, len(result.frames)).astype(int)
            labels = labels[idx]
        probs = np.array([f.step.disruption.probability for f in result.frames])
        label_rate = float(result.odl_label_rate or 0.0)
        auc = _auc_proxy(np.asarray(labels, dtype=np.float64), probs)
        scores.append(
            ODLShotScore(
                shot_id=result.shot_id,
                odl_label_rate=label_rate,
                mean_p_dis=result.mean_disruption,
                mean_snr=result.mean_snr,
                final_score=result.final_score,
                auc_proxy=auc,
            )
        )
        label_rates.append(label_rate)
        pdis_means.append(result.mean_disruption)

    corr = 0.0
    if len(label_rates) >= 2 and float(np.std(label_rates)) > 1e-9:
        corr = float(np.corrcoef(label_rates, pdis_means)[0, 1])
        if np.isnan(corr):
            corr = 0.0

    return ODLBenchmarkReport(
        n_shots=len(scores),
        mean_auc_proxy=float(np.mean([s.auc_proxy for s in scores])),
        mean_p_dis=float(np.mean(pdis_means)) if pdis_means else 0.0,
        correlation_label_pdis=corr,
        shots=scores,
    )
