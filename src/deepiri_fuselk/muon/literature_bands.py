"""Literature FPM bands for muon stripping validation (no orchestrator imports)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiteratureBenchmark:
    name: str
    fpm_range: tuple[float, float]
    source: str
    R_photon: float = 0.0
    R_proton: float = 0.0
    R_col: float = 0.35


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

TRIFECTA_LITERATURE_BAND = LITERATURE_BENCHMARKS[-1].fpm_range


def trifecta_within_literature(
    fpm: float, slack_fraction: float = 0.20
) -> tuple[bool, tuple[float, float]]:
    """Check a trifecta FPM value against the photon+proton literature band."""
    lo, hi = TRIFECTA_LITERATURE_BAND
    slack = slack_fraction * (hi - lo)
    within = (lo - slack) <= fpm <= (hi + slack)
    return within, TRIFECTA_LITERATURE_BAND
