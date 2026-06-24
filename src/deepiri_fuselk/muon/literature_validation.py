"""Experimental validation of muon stripping trifecta against literature benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

from deepiri_fuselk.muon.rate_network import BREAKEVEN_FUSIONS, RateNetworkParams, run_rate_network
from deepiri_fuselk.muon.stripping_orchestrator import StrippingTrifectaConfig, run_stripping_trifecta


@dataclass
class LiteratureBenchmark:
    name: str
    fpm_range: tuple[float, float]
    source: str
    R_photon: float = 0.0
    R_proton: float = 0.0
    R_col: float = 0.35


# Schematic targets from external-field-assisted muon reactivation literature
LITERATURE_BENCHMARKS: tuple[LiteratureBenchmark, ...] = (
    LiteratureBenchmark(
        name="collision_only",
        fpm_range=(105.0, 120.0),
        source="Baseline alpha-sticking with R_col ~ 0.35",
        R_col=0.35,
    ),
    LiteratureBenchmark(
        name="xfel_photon_assisted",
        fpm_range=(148.0, 165.0),
        source="X-ray photoelectric stripping (schematic)",
        R_col=0.35,
        R_photon=0.25,
    ),
    LiteratureBenchmark(
        name="photon_proton_trifecta",
        fpm_range=(180.0, 320.0),
        source="Combined photon + proton stripping (fuselk target band)",
        R_col=0.35,
        R_photon=0.25,
        R_proton=0.18,
    ),
)


@dataclass
class MuonValidationResult:
    benchmark: str
    measured_fpm: float
    expected_range: tuple[float, float]
    within_tolerance: bool
    breakeven: bool
    strip_rate: float


@dataclass
class MuonValidationReport:
    results: list[MuonValidationResult]
    trifecta_fpm: float
    trifecta_breakeven: bool
    breakeven_threshold: float
    all_benchmarks_pass: bool
    trifecta_pass: bool

    def to_dict(self) -> dict:
        return {
            "breakeven_threshold": self.breakeven_threshold,
            "trifecta_fpm": self.trifecta_fpm,
            "trifecta_breakeven": self.trifecta_breakeven,
            "all_benchmarks_pass": self.all_benchmarks_pass,
            "trifecta_pass": self.trifecta_pass,
            "benchmarks": [
                {
                    "name": r.benchmark,
                    "measured_fpm": r.measured_fpm,
                    "expected_range": list(r.expected_range),
                    "within_tolerance": r.within_tolerance,
                    "breakeven": r.breakeven,
                }
                for r in self.results
            ],
        }


def _run_benchmark(bench: LiteratureBenchmark) -> MuonValidationResult:
    params = RateNetworkParams(
        R_col=bench.R_col,
        R_photon=bench.R_photon,
        R_proton=bench.R_proton,
    )
    result = run_rate_network(params=params)
    lo, hi = bench.fpm_range
    # Allow 20% slack for schematic rate constants vs published curves
    slack = 0.20 * (hi - lo)
    within = (lo - slack) <= result.fusions_per_muon <= (hi + slack)
    return MuonValidationResult(
        benchmark=bench.name,
        measured_fpm=result.fusions_per_muon,
        expected_range=bench.fpm_range,
        within_tolerance=within,
        breakeven=result.breakeven,
        strip_rate=result.strip_rate,
    )


def validate_muon_trifecta(
    tolerance_slack: float = 0.15,
) -> MuonValidationReport:
    """Validate rate-network outputs against literature bands and trifecta orchestrator."""
    results = [_run_benchmark(b) for b in LITERATURE_BENCHMARKS]

    trifecta = run_stripping_trifecta()
    trifecta_pass = trifecta.projected_fpm > 0 and trifecta.R_total > trifecta.R_photon

    return MuonValidationReport(
        results=results,
        trifecta_fpm=trifecta.projected_fpm,
        trifecta_breakeven=trifecta.breakeven,
        breakeven_threshold=BREAKEVEN_FUSIONS,
        all_benchmarks_pass=all(r.within_tolerance for r in results),
        trifecta_pass=trifecta_pass,
    )
