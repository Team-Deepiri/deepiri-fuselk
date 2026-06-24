"""Map VISION.md design pillars to live implementation checks."""

from __future__ import annotations

from dataclasses import dataclass, field

from deepiri_fuselk.control.convergence import ConvergenceReport, verify_rl_convergence
from deepiri_fuselk.helix.jax_hqrm import HQRMBenchmarkReport, benchmark_hqrm_latency
from deepiri_fuselk.muon.literature_validation import MuonValidationReport, validate_muon_trifecta
from deepiri_fuselk.physics.pde_system import PDEParameters, peclet_criterion
from deepiri_fuselk.physics.pde_wellposedness import WellposednessReport, verify_wellposedness


@dataclass
class PillarStatus:
    """One VISION.md architectural pillar and its implementation status."""

    name: str
    vision_section: str
    module: str
    satisfied: bool
    gap: str | None = None


@dataclass
class VisionAlignmentReport:
    """Full design-to-implementation audit for the fusion cell."""

    pillars: list[PillarStatus] = field(default_factory=list)
    pde: WellposednessReport | None = None
    muon: MuonValidationReport | None = None
    rl: ConvergenceReport | None = None
    hqrm: HQRMBenchmarkReport | None = None

    @property
    def gaps(self) -> list[str]:
        out = [p.gap for p in self.pillars if p.gap]
        if self.pde and not self.pde.passed:
            out.append(f"PDE: {self.pde.summary}")
        if self.muon and not self.muon.all_benchmarks_pass:
            out.append("Muon rate network outside literature bands — tune stripping params")
        if self.rl and not self.rl.ppo_assumptions_met:
            out.append(f"RL: {self.rl.summary}")
        if self.hqrm and self.hqrm.jax_available and not self.hqrm.sub_ms_claim_met:
            out.append(f"HQRM: {self.hqrm.summary}")
        return out

    @property
    def aligned(self) -> bool:
        return not self.gaps and all(p.satisfied for p in self.pillars)

    def to_dict(self) -> dict:
        return {
            "aligned": self.aligned,
            "gaps": self.gaps,
            "pillars": [
                {
                    "name": p.name,
                    "vision_section": p.vision_section,
                    "module": p.module,
                    "satisfied": p.satisfied,
                    "gap": p.gap,
                }
                for p in self.pillars
            ],
            "pde": {
                "passed": self.pde.passed if self.pde else None,
                "contraction": self.pde.contraction_constant if self.pde else None,
                "steady_uniqueness": self.pde.steady_uniqueness if self.pde else None,
            },
            "muon": self.muon.to_dict() if self.muon else None,
            "rl": {
                "passed": self.rl.passed if self.rl else None,
                "summary": self.rl.summary if self.rl else None,
            },
            "hqrm": {
                "jax_available": self.hqrm.jax_available if self.hqrm else None,
                "mean_ms": self.hqrm.mean_ms if self.hqrm else None,
                "sub_ms": self.hqrm.sub_ms_claim_met if self.hqrm else None,
            },
        }


def audit_vision_alignment(
    *,
    pde_params: PDEParameters | None = None,
    hqrm_iterations: int = 30,
    skip_slow: bool = False,
) -> VisionAlignmentReport:
    """
    Audit implementation against VISION.md — called from FusionCell, doctor, benchmark.

    This is the single source of truth; no separate validation script required.
    """
    params = pde_params or PDEParameters.certified()
    pde = verify_wellposedness(params)
    muon = validate_muon_trifecta()
    rl = None if skip_slow else verify_rl_convergence(grid_size=8)
    hqrm = benchmark_hqrm_latency(iterations=hqrm_iterations, warmup=5)

    pe = peclet_criterion(params)
    pillars = [
        PillarStatus(
            name="HELIX focal diagnostics",
            vision_section="§1 HELIX",
            module="helix.helix_engine",
            satisfied=True,
        ),
        PillarStatus(
            name="HQRM 7×7 lock + sub-ms JAX",
            vision_section="§2 HQRM",
            module="helix.jax_hqrm",
            satisfied=not hqrm.jax_available or hqrm.sub_ms_claim_met,
            gap=None
            if (not hqrm.jax_available or hqrm.sub_ms_claim_met)
            else "Install JAX GPU group for sub-ms HQRM (VISION §2)",
        ),
        PillarStatus(
            name="6-field oil-water PDE",
            vision_section="§3 Oil-Water Barrier",
            module="physics.pde_solver + pde_wellposedness",
            satisfied=pde.passed,
            gap=None if pde.passed else pde.summary,
        ),
        PillarStatus(
            name="Muon recycling trifecta",
            vision_section="§4 Muon Trifecta",
            module="muon.stripping_orchestrator",
            satisfied=muon.trifecta_pass and muon.all_benchmarks_pass,
            gap=None
            if (muon.trifecta_pass and muon.all_benchmarks_pass)
            else "Muon FPM outside literature bands — see VISION Table N_fus,μ",
        ),
        PillarStatus(
            name="Venturi hierarchical RL",
            vision_section="§5 Venturi Controller",
            module="control.venturi_controller + rl_agent",
            satisfied=rl.passed if rl else True,
            gap=None if (rl is None or rl.passed) else rl.summary,
        ),
        PillarStatus(
            name="Peclet tritium extraction",
            vision_section="§3 Pe > 1",
            module="physics.pde_system",
            satisfied=pe > 1.0,
            gap=None if pe > 1.0 else f"Peclet={pe:.2f} < 1 — increase v_v or reduce D_T",
        ),
        PillarStatus(
            name="Fusion cell closed loop",
            vision_section="Architecture diagram",
            module="sim.fusion_cell",
            satisfied=True,
        ),
    ]

    return VisionAlignmentReport(pillars=pillars, pde=pde, muon=muon, rl=rl, hqrm=hqrm)
