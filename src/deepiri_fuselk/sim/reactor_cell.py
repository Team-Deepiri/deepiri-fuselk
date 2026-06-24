"""Full fusion reactor cell — closed loop with KPI tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from deepiri_fuselk.barrier.breeding_blanket import tritium_breeding_ratio
from deepiri_fuselk.control.policy_runner import HybridPolicyRunner
from deepiri_fuselk.control.venturi_controller import VenturiController
from deepiri_fuselk.data.imas_loader import IMASShot, synthetic_imas_shot
from deepiri_fuselk.helix.helix_engine import HelixEngine
from deepiri_fuselk.models.disruption_detector import DisruptionAssessment, DisruptionDetector
from deepiri_fuselk.models.elm_predictor import ELMPredictor
from deepiri_fuselk.muon.rate_network import RateNetworkParams, run_rate_network
from deepiri_fuselk.physics.pde_solver import solve_oil_water_steady
from deepiri_fuselk.physics.pde_system import PDEParameters
from deepiri_fuselk.sim.fusion_kpis import FusionKPIs, divertor_uniformity, elm_free_fraction
from deepiri_fuselk.sim.synthetic_data_gen import generate_ece_shot


@dataclass
class ReactorStep:
    step: int
    kpis: FusionKPIs
    disruption: DisruptionAssessment
    heat_flux: np.ndarray
    action_taken: str


@dataclass
class ReactorRun:
    steps: list[ReactorStep] = field(default_factory=list)
    final_score: float = 0.0

    def to_report(self) -> dict:
        if not self.steps:
            return {"final_score": 0.0, "steps": 0}
        last = self.steps[-1].kpis
        return {
            "steps": len(self.steps),
            "final_score": self.final_score,
            "tritium_breeding_ratio": last.tritium_breeding_ratio,
            "elm_free_fraction": last.elm_free_fraction,
            "disruption_risk": last.disruption_risk,
            "muon_gain": last.muon_gain,
            "divertor_uniformity": last.divertor_uniformity,
        }


class ReactorCell:
    """
    Self-sustaining fusion cell simulation.

    Closed loop: diagnostics -> HELIX -> disruption detect -> Venturi/RL actuate -> KPI update.
    """

    def __init__(
        self,
        grid_size: int = 32,
        policy_path: Path | None = None,
        train_elm: bool = True,
    ) -> None:
        self.grid_size = grid_size
        self.helix = HelixEngine()
        self.elm = ELMPredictor()
        if train_elm:
            self._bootstrap_elm_model()
        self.detector = DisruptionDetector(self.elm)
        self.venturi = VenturiController()
        self.hybrid = HybridPolicyRunner(policy_path)
        self._pde = solve_oil_water_steady(n_grid=grid_size)
        self._muon = run_rate_network(
            params=RateNetworkParams(R_photon=0.4, R_proton=0.25),
            t_span=(0.0, 2e-5),
        )
        self.imas: IMASShot = synthetic_imas_shot(size=grid_size)
        self._step = 0
        self._elm_probs: list[float] = []
        self._rewards: list[float] = []
        self._snrs: list[float] = []

    def _bootstrap_elm_model(self, n_shots: int = 60) -> None:
        from deepiri_fuselk.sim.shot_corpus import generate_corpus

        corpus = generate_corpus(n_shots=n_shots, grid_size=self.grid_size, seed=42)
        self.elm.train_from_corpus(corpus)

    def reset(self, seed: int = 42) -> ReactorStep:
        self._step = 0
        self._elm_probs = []
        self._rewards = []
        self._snrs = []
        self.imas = synthetic_imas_shot(size=self.grid_size, seed=seed)
        return self.step(seed=seed)

    def step(self, seed: int | None = None) -> ReactorStep:
        self._step += 1
        s = seed if seed is not None else 42 + self._step
        shot = generate_ece_shot(self.grid_size, seed=s)

        helix = self.helix.process(shot.heat_field, shot.raw_signal, shot.angles)
        q_vals = np.array(self.imas.q_profile.values, dtype=np.float64)
        q_min = float(np.min(q_vals)) if len(q_vals) else 2.0
        beta_n = float(np.mean(self.imas.Te_profile.values)) / 2000.0

        disruption = self.detector.assess(helix, q_min=q_min, beta_n=beta_n)

        hybrid = self.hybrid.step(
            shot.heat_field,
            elm_probability=disruption.probability,
            rl_steps=1,
        )
        self._rewards.append(hybrid.venturi.reward)

        # Act on disruption recommendation
        action = disruption.recommended_action
        if action == "pellet_inject":
            hybrid.venturi.action.pellet_ready = True
        elif action == "gas_puff_radiate":
            hybrid.venturi.action.gas_puff = 0.8

        self._elm_probs.append(disruption.probability)
        self._snrs.append(helix.phase_locked_snr)

        tbr = tritium_breeding_ratio(self._pde.state, PDEParameters())
        kpis = FusionKPIs(
            tritium_breeding_ratio=tbr,
            elm_free_fraction=elm_free_fraction(self._elm_probs),
            divertor_uniformity=divertor_uniformity(hybrid.final_heat),
            disruption_risk=disruption.probability,
            muon_gain=self._muon.fusions_per_muon,
            helix_snr_mean=float(np.mean(self._snrs)),
            venturi_mean_reward=float(np.mean(self._rewards)),
            q_min=q_min,
            beta_n=beta_n,
        )

        rs = ReactorStep(
            step=self._step,
            kpis=kpis,
            disruption=disruption,
            heat_flux=hybrid.final_heat,
            action_taken=action,
        )
        return rs

    def run(self, n_steps: int = 100, seed: int = 42) -> ReactorRun:
        self.reset(seed=seed)
        run = ReactorRun()
        for i in range(n_steps):
            run.steps.append(self.step(seed=seed + i))
        run.final_score = run.steps[-1].kpis.score() if run.steps else 0.0
        return run

    def export_report(self, path: str | Path) -> dict:
        import json

        run = self.run(n_steps=50)
        report = run.to_report()
        Path(path).write_text(json.dumps(report, indent=2))
        return report
