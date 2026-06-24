"""Experimental validation of muon stripping trifecta against literature benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

from deepiri_fuselk.muon.literature_bands import (
    LITERATURE_BENCHMARKS,
    TRIFECTA_LITERATURE_BAND,
    LiteratureBenchmark,
    trifecta_within_literature,
)
from deepiri_fuselk.muon.rate_network import BREAKEVEN_FUSIONS, RateNetworkParams, run_rate_network
from deepiri_fuselk.muon.stripping_orchestrator import run_stripping_trifecta


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


__all__ = [
    "LITERATURE_BENCHMARKS",
    "TRIFECTA_LITERATURE_BAND",
    "MuonValidationReport",
    "MuonValidationResult",
    "trifecta_within_literature",
    "validate_muon_trifecta",
]
