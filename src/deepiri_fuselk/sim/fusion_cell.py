"""DeepIRI Self-Sustaining Fusion Cell — full architecture orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from deepiri_fuselk.barrier.breeding_blanket import evaluate_breeding_blanket
from deepiri_fuselk.barrier.heat_exhaust import evaluate_brine_coating
from deepiri_fuselk.physics.pde_system import peclet_criterion
from deepiri_fuselk.sim.fuel_cycle_context import FuelCycleContext, build_fuel_cycle_context
from deepiri_fuselk.sim.fusion_kpis import FusionKPIs
from deepiri_fuselk.sim.reactor_cell import ReactorCell, ReactorRun, ReactorStep
from deepiri_fuselk.sim.vision_alignment import VisionAlignmentReport, audit_vision_alignment


@dataclass
class FuelCycleStatus:
    tritium_breeding_ratio: float
    peclet_number: float
    extraction_ok: bool
    outward_flux: float
    brine_viable: bool
    brine_heat_reduction: float


@dataclass
class MuonCycleStatus:
    fusions_per_muon: float
    breakeven: bool
    margin_to_breakeven: float
    strip_rate_total: float
    photon_active: bool
    cyclotron_locked: bool
    literature_aligned: bool


@dataclass
class FusionCellReport:
    fusion_score: float
    reactor: dict
    fuel_cycle: FuelCycleStatus
    muon_cycle: MuonCycleStatus
    elm_free_fraction: float
    disruption_risk: float
    recommended_actions: list[str] = field(default_factory=list)
    vision: VisionAlignmentReport | None = None

    def to_dict(self) -> dict:
        return {
            "fusion_score": self.fusion_score,
            "reactor": self.reactor,
            "fuel_cycle": {
                "tritium_breeding_ratio": self.fuel_cycle.tritium_breeding_ratio,
                "peclet_number": self.fuel_cycle.peclet_number,
                "extraction_ok": self.fuel_cycle.extraction_ok,
                "outward_flux": self.fuel_cycle.outward_flux,
                "brine_viable": self.fuel_cycle.brine_viable,
                "brine_heat_reduction": self.fuel_cycle.brine_heat_reduction,
            },
            "muon_cycle": {
                "fusions_per_muon": self.muon_cycle.fusions_per_muon,
                "breakeven": self.muon_cycle.breakeven,
                "margin_to_breakeven": self.muon_cycle.margin_to_breakeven,
                "strip_rate_total": self.muon_cycle.strip_rate_total,
                "photon_active": self.muon_cycle.photon_active,
                "cyclotron_locked": self.muon_cycle.cyclotron_locked,
                "literature_aligned": self.muon_cycle.literature_aligned,
            },
            "elm_free_fraction": self.elm_free_fraction,
            "disruption_risk": self.disruption_risk,
            "recommended_actions": self.recommended_actions,
            "vision": self.vision.to_dict() if self.vision else None,
        }


class FusionCell:
    """
    Top-level orchestrator for the DeepIRI unified reactor architecture.

    Layers:
      1. ReactorCell — HELIX → disruption → Venturi control loop
      2. Oil-water PDE barrier — tritium breeding + Peclet extraction
      3. Brine coating hypothesis — wall thermal management (sim)
      4. Muon stripping trifecta — photon + proton + cyclotron recycling
    """

    def __init__(
        self,
        grid_size: int = 32,
        policy_path: Path | None = None,
        train_elm: bool = False,
        brine_salinity_ppt: float = 35.0,
        fuel_cycle: FuelCycleContext | None = None,
        audit_vision_on_init: bool = False,
    ) -> None:
        self.grid_size = grid_size
        self.brine_salinity = brine_salinity_ppt
        self._fuel_cycle_ctx = fuel_cycle or build_fuel_cycle_context(grid_size)
        self.reactor = ReactorCell(
            grid_size=grid_size,
            policy_path=policy_path,
            train_elm=train_elm,
            fuel_cycle=self._fuel_cycle_ctx,
        )
        self._vision_cache: VisionAlignmentReport | None = None
        if audit_vision_on_init:
            self.audit_vision()

    def audit_vision(self, *, skip_slow: bool = True) -> VisionAlignmentReport:
        """Run VISION.md alignment audit (lazy — not on every sim step)."""
        self._vision_cache = audit_vision_alignment(
            pde_params=self._fuel_cycle_ctx.pde_params,
            skip_slow=skip_slow,
        )
        return self._vision_cache

    @property
    def vision(self) -> VisionAlignmentReport | None:
        return self._vision_cache

    def _fuel_cycle(self) -> FuelCycleStatus:
        br = evaluate_breeding_blanket(
            self._fuel_cycle_ctx.pde.state, self._fuel_cycle_ctx.pde_params
        )
        pe = peclet_criterion(self._fuel_cycle_ctx.pde_params)
        coating = evaluate_brine_coating(salinity_ppt=self.brine_salinity)
        return FuelCycleStatus(
            tritium_breeding_ratio=br.tritium_breeding_ratio,
            peclet_number=pe,
            extraction_ok=pe > 1.0,
            outward_flux=br.outward_flux,
            brine_viable=coating.viable,
            brine_heat_reduction=coating.effective_heat_reduction,
        )

    def _muon_cycle(self) -> MuonCycleStatus:
        m = self._fuel_cycle_ctx.muon_trifecta
        return MuonCycleStatus(
            fusions_per_muon=m.projected_fpm,
            breakeven=m.breakeven,
            margin_to_breakeven=m.margin_to_breakeven,
            strip_rate_total=m.R_total,
            photon_active=m.photon_viable,
            cyclotron_locked=m.cyclotron_locked,
            literature_aligned=m.literature_aligned,
        )

    def run(
        self,
        n_steps: int = 100,
        seed: int = 42,
        *,
        include_vision: bool = False,
    ) -> tuple[ReactorRun, FusionCellReport]:
        run = self.reactor.run(n_steps=n_steps, seed=seed)
        fuel = self._fuel_cycle()
        muon = self._muon_cycle()

        actions = list({s.action_taken for s in run.steps})
        last_kpis = (
            run.steps[-1].kpis
            if run.steps
            else FusionKPIs(
                tritium_breeding_ratio=0,
                elm_free_fraction=0,
                divertor_uniformity=0,
                disruption_risk=1,
                muon_gain=0,
                helix_snr_mean=0,
                venturi_mean_reward=0,
                q_min=0,
                beta_n=0,
            )
        )

        enriched = FusionKPIs(
            tritium_breeding_ratio=fuel.tritium_breeding_ratio,
            elm_free_fraction=last_kpis.elm_free_fraction,
            divertor_uniformity=last_kpis.divertor_uniformity,
            disruption_risk=last_kpis.disruption_risk,
            muon_gain=muon.fusions_per_muon,
            helix_snr_mean=last_kpis.helix_snr_mean,
            venturi_mean_reward=last_kpis.venturi_mean_reward,
            q_min=last_kpis.q_min,
            beta_n=last_kpis.beta_n,
        )
        run.final_score = enriched.score()

        vision = self.audit_vision() if include_vision else self._vision_cache

        report = FusionCellReport(
            fusion_score=run.final_score,
            reactor=run.to_report(),
            fuel_cycle=fuel,
            muon_cycle=muon,
            elm_free_fraction=last_kpis.elm_free_fraction,
            disruption_risk=last_kpis.disruption_risk,
            recommended_actions=actions,
            vision=vision,
        )
        return run, report

    def step(self, seed: int | None = None) -> ReactorStep:
        return self.reactor.step(seed=seed)
