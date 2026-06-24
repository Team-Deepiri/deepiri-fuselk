"""Full fusion reactor cell — closed loop with KPI tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from deepiri_fuselk.barrier.breeding_blanket import tritium_breeding_ratio
from deepiri_fuselk.control.policy_runner import HybridPolicyRunner
from deepiri_fuselk.data.imas_loader import IMASShot, synthetic_imas_shot
from deepiri_fuselk.helix.helix_engine import HelixEngine, HelixResult
from deepiri_fuselk.models.disruption_detector import DisruptionAssessment, DisruptionDetector
from deepiri_fuselk.models.elm_predictor import ELMPredictor
from deepiri_fuselk.sim.fuel_cycle_context import FuelCycleContext, build_fuel_cycle_context
from deepiri_fuselk.sim.fusion_kpis import FusionKPIs, RunningKPIAccumulator, divertor_uniformity
from deepiri_fuselk.sim.shot_pipeline import ShotPipeline

DEFAULT_ELM_MODEL = Path(".fuselk-data/models/elm_predictor.json")


@dataclass
class ReactorStep:
    step: int
    kpis: FusionKPIs
    disruption: DisruptionAssessment
    heat_flux: np.ndarray
    action_taken: str
    helix: HelixResult
    raw_heat: np.ndarray
    seed: int


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

    Closed loop: diagnostics → HELIX → disruption detect → Venturi/RL actuate → KPI update.
    """

    def __init__(
        self,
        grid_size: int = 32,
        policy_path: Path | None = None,
        train_elm: bool = False,
        elm_model_path: Path | None = None,
        fuel_cycle: FuelCycleContext | None = None,
    ) -> None:
        self.grid_size = grid_size
        self.helix = HelixEngine()
        self.elm = self._init_elm(train_elm, elm_model_path, grid_size)
        self.detector = DisruptionDetector(self.elm)
        self.hybrid = HybridPolicyRunner(policy_path)
        self._pipeline = ShotPipeline(self.helix, self.detector, self.hybrid)
        self._fuel_cycle = fuel_cycle or build_fuel_cycle_context(grid_size)
        self.imas: IMASShot = synthetic_imas_shot(size=grid_size)
        self._step = 0
        self._kpi_acc = RunningKPIAccumulator()

    @staticmethod
    def _init_elm(train_elm: bool, model_path: Path | None, grid_size: int) -> ELMPredictor:
        path = model_path or DEFAULT_ELM_MODEL
        if train_elm:
            from deepiri_fuselk.sim.shot_corpus import generate_corpus

            corpus = generate_corpus(n_shots=60, grid_size=grid_size, seed=42)
            model = ELMPredictor()
            model.train_from_corpus(corpus)
            return model
        if path.exists():
            return ELMPredictor.load(path)
        return ELMPredictor()

    def reset(self, seed: int = 42) -> ReactorStep:
        self._step = 0
        self._kpi_acc.reset()
        self.imas = synthetic_imas_shot(size=self.grid_size, seed=seed)
        self.hybrid.venturi.reset()
        return self.step(seed=seed)

    def step(self, seed: int | None = None) -> ReactorStep:
        self._step += 1
        s = seed if seed is not None else 42 + self._step

        result = self._pipeline.process(
            grid_size=self.grid_size,
            seed=s,
            q_profile_values=np.array(self.imas.q_profile.values, dtype=np.float64),
            te_profile_values=np.array(self.imas.Te_profile.values, dtype=np.float64),
        )

        self._kpi_acc.update(
            snr=result.helix.phase_locked_snr,
            reward=result.control.venturi.reward,
            elm_probability=result.disruption.probability,
        )

        tbr = tritium_breeding_ratio(
            self._fuel_cycle.pde.state,
            self._fuel_cycle.pde_params,
        )
        kpis = FusionKPIs(
            tritium_breeding_ratio=tbr,
            elm_free_fraction=self._kpi_acc.elm_free_fraction,
            divertor_uniformity=divertor_uniformity(result.control.final_heat),
            disruption_risk=result.disruption.probability,
            muon_gain=self._fuel_cycle.muon_fpm,
            helix_snr_mean=self._kpi_acc.helix_snr_mean,
            venturi_mean_reward=self._kpi_acc.venturi_mean_reward,
            q_min=result.q_min,
            beta_n=result.beta_n,
        )

        return ReactorStep(
            step=self._step,
            kpis=kpis,
            disruption=result.disruption,
            heat_flux=result.control.final_heat,
            action_taken=result.recommended_action,
            helix=result.helix,
            raw_heat=result.shot.heat_field,
            seed=s,
        )

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
