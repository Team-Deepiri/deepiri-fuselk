"""Proton collision stripping of (alpha-mu)+ ions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProtonStripperConfig:
    energy_kev: float = 100.0
    flux: float = 1e15
    cross_section: float = 1e-20
    recycle_efficiency: float = 0.7
    relative_velocity: float = 1e5


def collision_cross_section(config: ProtonStripperConfig) -> float:
    """sigma ~ 1 / v_rel^2 (schematic)."""
    return config.cross_section / max(config.relative_velocity**2, 1.0)


def stripping_rate(config: ProtonStripperConfig) -> float:
    return collision_cross_section(config) * config.flux * config.recycle_efficiency
