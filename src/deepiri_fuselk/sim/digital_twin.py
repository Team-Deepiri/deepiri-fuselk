"""Full digital twin orchestrator — HELIX + Venturi + Physics + Muon."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from deepiri_fuselk.control.venturi_controller import VenturiController, VenturiState
from deepiri_fuselk.data.imas_loader import IMASShot, synthetic_imas_shot
from deepiri_fuselk.helix.helix_engine import HelixEngine, HelixResult
from deepiri_fuselk.models.elm_predictor import ELMPredictor
from deepiri_fuselk.muon.rate_network import RateNetworkParams, run_rate_network
from deepiri_fuselk.physics.pde_solver import solve_oil_water_steady
from deepiri_fuselk.sim.synthetic_data_gen import generate_ece_shot


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
    End-to-end fusion digital twin.

    Integrates:
      - IMAS shot data (synthetic or real)
      - HELIX focal diagnostics
      - ELM predictor
      - Venturi hierarchical controller
      - Oil-water PDE physics
      - Muon rate network
    """

    def __init__(self, grid_size: int = 32, device: str = "synthetic") -> None:
        self.grid_size = grid_size
        self.device = device
        self.helix = HelixEngine()
        self.venturi = VenturiController()
        self.elm_predictor = ELMPredictor()
        self.imas_shot: IMASShot = synthetic_imas_shot(size=grid_size)
        self._pde = solve_oil_water_steady(n_grid=grid_size, max_iter=100)
        self._muon = run_rate_network(
            t_span=(0.0, 1e-5),
            params=RateNetworkParams(R_photon=0.3, R_proton=0.2),
        )
        self._step = 0
        self._heat = np.zeros((grid_size, grid_size))
        self._helix_result: HelixResult | None = None
        self._venturi_state: VenturiState | None = None
        self.history = TwinHistory()

    def reset(self, seed: int = 42) -> TwinState:
        self._step = 0
        self.imas_shot = synthetic_imas_shot(size=self.grid_size, seed=seed)
        self.history = TwinHistory()
        shot = generate_ece_shot(self.grid_size, seed=seed)
        self._helix_result = self.helix.process(shot.heat_field, shot.raw_signal, shot.angles)
        self._venturi_state = self.venturi.step(
            shot.heat_field,
            rotation_khz=self._helix_result.rotation_hz / 1e3,
            elm_probability=self._helix_result.elm_probability,
        )
        self._heat = self._helix_result.focal_map
        return self._build_state()

    def step(self, action: np.ndarray | None = None) -> TwinState:
        self._step += 1
        shot = generate_ece_shot(self.grid_size, seed=42 + self._step)

        # HELIX pipeline
        self._helix_result = self.helix.process(shot.heat_field, shot.raw_signal, shot.angles)
        elm_pred = self.elm_predictor.predict(
            self._helix_result.focal_map,
            rotation_hz=self._helix_result.rotation_hz,
        )

        # Venturi control
        self._venturi_state = self.venturi.step(
            shot.heat_field,
            rotation_khz=self._helix_result.rotation_hz / 1e3,
            elm_probability=elm_pred.probability,
        )
        self._heat = 0.5 * shot.heat_field + 0.5 * self._helix_result.focal_map

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
            tritium_outflux=float(self._pde.state.n_T[-1]),
            watchdog_triggered=self._venturi_state.action.overridden,
            helix_snr=self._helix_result.phase_locked_snr,
            venturi_reward=self._venturi_state.reward,
            muon_breakeven=self._muon.breakeven,
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
            "muon_breakeven": self._muon.breakeven,
        }
