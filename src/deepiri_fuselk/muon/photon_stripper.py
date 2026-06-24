"""X-ray / laser photon stripping of stuck muons."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhotonStripperConfig:
    energy_kev: float = 2.0
    flux: float = 1e18
    cross_section: float = 1e-22
    recycle_efficiency: float = 0.8
    topological_charge: int = 3


def binding_energy_kev() -> float:
    return 1.5


def photoelectric_cross_section(config: PhotonStripperConfig, Z: int = 2) -> float:
    """sigma_photo ~ Z^5 / E^3.5 (schematic scaling)."""
    return config.cross_section * (Z**5) / (config.energy_kev**3.5)


def stripping_rate(config: PhotonStripperConfig) -> float:
    """R_photon = sigma * flux * eta."""
    sigma = photoelectric_cross_section(config)
    base = sigma * config.flux * config.recycle_efficiency
    # vortex OAM boost
    oam_boost = 1.0 + 0.1 * config.topological_charge
    return base * oam_boost


def can_strip(config: PhotonStripperConfig) -> bool:
    return config.energy_kev > binding_energy_kev()
