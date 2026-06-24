"""Stateful simulation engine for live dashboard visualization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from deepiri_fuselk.helix.helix_engine import HelixResult
from deepiri_fuselk.models.disruption_detector import DisruptionAssessment
from deepiri_fuselk.models.elm_predictor import ELMPrediction
from deepiri_fuselk.sim.fusion_cell import FusionCell, FusionCellReport
from deepiri_fuselk.sim.reactor_cell import ReactorStep


@dataclass
class SimulationFrame:
    step: int
    seed: int
    raw_heat: np.ndarray
    helix: HelixResult
    elm: ELMPrediction
    disruption: DisruptionAssessment
    controlled_heat: np.ndarray
    action: str
    fusion_score: float
    tbr: float
    muon_fpm: float
    peclet: float
    elm_free_fraction: float
    divertor_uniformity: float


@dataclass
class SimulationState:
    step_count: int = 0
    seed: int = 42
    elm_probs: list[float] = field(default_factory=list)
    fusion_score: float = 0.0
    report: FusionCellReport | None = None


class LiveSimulation:
    """Step FusionCell/ReactorCell for real-time dashboard updates."""

    def __init__(self, grid_size: int = 24) -> None:
        self.grid_size = grid_size
        self.cell = FusionCell(grid_size=grid_size, train_elm=False)
        self.state = SimulationState()
        self._last: SimulationFrame | None = None

    def reset(self, seed: int = 42) -> SimulationFrame:
        self.state = SimulationState(seed=seed)
        self.cell.reactor.reset(seed=seed)
        return self.step()

    def step(self) -> SimulationFrame:
        self.state.step_count += 1
        seed = self.state.seed + self.state.step_count
        rs: ReactorStep = self.cell.step(seed=seed)

        self.state.elm_probs.append(rs.disruption.probability)
        fuel = self.cell._fuel_cycle()
        muon = self.cell._muon_cycle()
        kpis = rs.kpis
        self.state.fusion_score = kpis.score()

        frame = SimulationFrame(
            step=self.state.step_count,
            seed=seed,
            raw_heat=rs.raw_heat,
            helix=rs.helix,
            elm=rs.disruption.elm,
            disruption=rs.disruption,
            controlled_heat=rs.heat_flux,
            action=rs.action_taken,
            fusion_score=self.state.fusion_score,
            tbr=fuel.tritium_breeding_ratio,
            muon_fpm=muon.fusions_per_muon,
            peclet=fuel.peclet_number,
            elm_free_fraction=kpis.elm_free_fraction,
            divertor_uniformity=kpis.divertor_uniformity,
        )
        self._last = frame
        return frame

    @property
    def last_frame(self) -> SimulationFrame | None:
        return self._last
