"""Unified muon recycling: photon + proton + cyclotron resonance."""

from __future__ import annotations

from dataclasses import dataclass

from deepiri_fuselk.muon.cyclotron_resonance import CyclotronConfig, resonance_match
from deepiri_fuselk.muon.photon_stripper import PhotonStripperConfig, can_strip, stripping_rate as photon_rate
from deepiri_fuselk.muon.proton_stripper import ProtonStripperConfig, stripping_rate as proton_rate
from deepiri_fuselk.muon.rate_network import BREAKEVEN_FUSIONS, RateNetworkParams, RateNetworkResult, run_rate_network


@dataclass
class StrippingTrifectaConfig:
    photon: PhotonStripperConfig
    proton: ProtonStripperConfig
    cyclotron: CyclotronConfig
    collision_fraction: float = 0.35


@dataclass
class StrippingTrifectaResult:
    R_photon: float
    R_proton: float
    R_cyclotron: float
    R_total: float
    cyclotron_locked: bool
    photon_viable: bool
    rate_network: RateNetworkResult
    projected_fpm: float
    breakeven: bool
    margin_to_breakeven: float


def cyclotron_strip_contribution(config: CyclotronConfig) -> float:
    """RF resonance adds stripping when locked; schematic 0–0.15 scale."""
    if not resonance_match(config):
        return 0.0
    return min(0.15, config.amplitude_v / 10000.0)


def normalize_strip_rates(R_photon: float, R_proton: float, R_cyclotron: float, R_col: float) -> tuple[float, float, float]:
    """Map physical rates to rate-network dimensionless stripping fractions."""
    raw = R_photon + R_proton + R_cyclotron
    if raw <= 0:
        return 0.0, 0.0, 0.0
    scale = min(0.65, raw) / raw
    return R_photon * scale, R_proton * scale, R_cyclotron * scale


def run_stripping_trifecta(
    config: StrippingTrifectaConfig | None = None,
    t_span: tuple[float, float] = (0.0, 5e-5),
) -> StrippingTrifectaResult:
    """Run full muon recycling stack and rate network."""
    config = config or StrippingTrifectaConfig(
        photon=PhotonStripperConfig(),
        proton=ProtonStripperConfig(),
        cyclotron=CyclotronConfig(),
    )

    R_ph = photon_rate(config.photon) if can_strip(config.photon) else 0.0
    R_pr = proton_rate(config.proton)
    R_cy = cyclotron_strip_contribution(config.cyclotron)

    R_ph_n, R_pr_n, R_cy_n = normalize_strip_rates(R_ph, R_pr, R_cy, config.collision_fraction)
    R_total = config.collision_fraction + R_ph_n + R_pr_n + R_cy_n

    params = RateNetworkParams(
        R_col=config.collision_fraction,
        R_photon=R_ph_n,
        R_proton=R_pr_n + R_cy_n,
    )
    network = run_rate_network(t_span=t_span, params=params)
    margin = network.fusions_per_muon - BREAKEVEN_FUSIONS

    return StrippingTrifectaResult(
        R_photon=R_ph_n,
        R_proton=R_pr_n,
        R_cyclotron=R_cy_n,
        R_total=R_total,
        cyclotron_locked=resonance_match(config.cyclotron),
        photon_viable=can_strip(config.photon),
        rate_network=network,
        projected_fpm=network.fusions_per_muon,
        breakeven=network.breakeven,
        margin_to_breakeven=margin,
    )
