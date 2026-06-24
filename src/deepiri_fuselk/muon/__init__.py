"""Muon catalytic cycle subpackage."""

from deepiri_fuselk.muon.cyclotron_resonance import (
    CyclotronConfig,
    cyclotron_frequency,
    resonance_match,
)
from deepiri_fuselk.muon.photon_stripper import PhotonStripperConfig
from deepiri_fuselk.muon.photon_stripper import stripping_rate as photon_rate
from deepiri_fuselk.muon.proton_stripper import ProtonStripperConfig
from deepiri_fuselk.muon.proton_stripper import stripping_rate as proton_rate
from deepiri_fuselk.muon.rate_network import (
    BREAKEVEN_FUSIONS,
    RateNetworkParams,
    RateNetworkResult,
    effective_sticking,
    run_rate_network,
)

__all__ = [
    "BREAKEVEN_FUSIONS",
    "CyclotronConfig",
    "PhotonStripperConfig",
    "ProtonStripperConfig",
    "RateNetworkParams",
    "RateNetworkResult",
    "cyclotron_frequency",
    "effective_sticking",
    "photon_rate",
    "proton_rate",
    "resonance_match",
    "run_rate_network",
]
