"""Full digital twin orchestrator — HELIX + Venturi + Physics + Muon."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from deepiri_fuselk.control.policy_runner import HybridPolicyRunner
from deepiri_fuselk.control.venturi_controller import VenturiState
from deepiri_fuselk.data.imas_loader import IMASShot, synthetic_imas_shot
from deepiri_fuselk.helix.helix_engine import HelixEngine, HelixResult
from deepiri_fuselk.models.disruption_detector import DisruptionDetector
from deepiri_fuselk.models.elm_predictor import ELMPredictor
from deepiri_fuselk.sim.fuel_cycle_context import FuelCycleContext, build_fuel_cycle_context
from deepiri_fuselk.sim.shot_pipeline import ShotPipeline


@dataclass
class TwinState:
    heat_variance: float
    elm_probability: float
    tritium_outflux: float
    watchdog_triggered: bool
    helix_snr: float = 0.0
    venturi_reward: float = 0.0
    muon_breakeven: bool = False
    fracture_vector: tuple[float, float] = (0.0, 0.0)


@dataclass
class TwinHistory:
    elm_probs: list[float] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    snr_gains: list[float] = field(default_factory=list)


class DigitalTwin:
    """
    End-to-end fusion digital twin (VISION full stack).

    Uses the same ShotPipeline as ReactorCell for consistent HELIX → Venturi flow.
    """

    def __init__(self, grid_size: int = 32, device: str = "synthetic") -> None:
        self.grid_size = grid_size
        self.device = device
        self.helix = HelixEngine()
        self.elm_predictor = ELMPredictor()
        self.detector = DisruptionDetector(self.elm_predictor)
        self.hybrid = HybridPolicyRunner()
        self._pipeline = ShotPipeline(self.helix, self.detector, self.hybrid)
        self._fuel_cycle: FuelCycleContext = build_fuel_cycle_context(grid_size)
        self.imas_shot: IMASShot = synthetic_imas_shot(size=grid_size)
        self._step = 0
        self._heat = np.zeros((grid_size, grid_size))
        self._helix_result: HelixResult | None = None
        self._venturi_state: VenturiState | None = None
        self.history = TwinHistory()

    def reset(self, seed: int = 42) -> TwinState:
        self._step = 0
        self.imas_shot = synthetic_imas_shot(size=self.grid_size, seed=seed)
        self.history = TwinHistory()
        self.hybrid.venturi.reset()
        result = self._pipeline.process(
            grid_size=self.grid_size,
            seed=seed,
            q_profile_values=np.array(self.imas_shot.q_profile.values, dtype=np.float64),
            te_profile_values=np.array(self.imas_shot.Te_profile.values, dtype=np.float64),
        )
        self._helix_result = result.helix
        self._venturi_state = result.control.venturi
        self._heat = result.helix.focal_map
        return self._build_state()

    def step(self, action: np.ndarray | None = None) -> TwinState:
        del action  # reserved for external RL override
        self._step += 1
        result = self._pipeline.process(
            grid_size=self.grid_size,
            seed=42 + self._step,
            q_profile_values=np.array(self.imas_shot.q_profile.values, dtype=np.float64),
            te_profile_values=np.array(self.imas_shot.Te_profile.values, dtype=np.float64),
        )
        self._helix_result = result.helix
        self._venturi_state = result.control.venturi
        self._heat = 0.5 * result.shot.heat_field + 0.5 * result.helix.focal_map

        state = self._build_state()
        self.history.elm_probs.append(state.elm_probability)
        self.history.rewards.append(state.venturi_reward)
        self.history.snr_gains.append(state.helix_snr)
        return state

    def _build_state(self) -> TwinState:
        assert self._helix_result is not None
        assert self._venturi_state is not None
        return TwinState(
            heat_variance=self._venturi_state.traffic.variance,
            elm_probability=self._helix_result.elm_probability,
            tritium_outflux=float(self._fuel_cycle.pde.state.n_T[-1]),
            watchdog_triggered=self._venturi_state.action.overridden,
            helix_snr=self._helix_result.phase_locked_snr,
            venturi_reward=self._venturi_state.reward,
            muon_breakeven=self._fuel_cycle.muon_trifecta.breakeven,
            fracture_vector=self._helix_result.fracture_vector,
        )

    def summary(self) -> dict:
        return {
            "device": self.device,
            "steps": self._step,
            "mean_elm_prob": float(np.mean(self.history.elm_probs))
            if self.history.elm_probs
            else 0,
            "mean_reward": float(np.mean(self.history.rewards)) if self.history.rewards else 0,
            "mean_snr": float(np.mean(self.history.snr_gains)) if self.history.snr_gains else 0,
            "muon_breakeven": self._fuel_cycle.muon_trifecta.breakeven,
        }
