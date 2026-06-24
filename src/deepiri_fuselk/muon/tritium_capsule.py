"""Tritium breeding capsule engineering model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TritiumCapsule:
    """State-of-the-art tritium breeding capsule parameters."""

    lithium_fraction: float = 0.9
    beryllium_multiplier: float = 1.8
    helium_purge_rate: float = 0.5
    permeation_barrier_mm: float = 0.1
    operating_temp_c: float = 500.0


@dataclass
class CapsulePerformance:
    breeding_ratio: float
    tritium_inventory_g: float
    extraction_efficiency: float
    self_sustaining: bool


def evaluate_capsule(capsule: TritiumCapsule, neutron_flux: float = 1e14) -> CapsulePerformance:
    """
    Evaluate tritium capsule breeding performance.

    Models: n + 6Li -> 4He + 3H with Be neutron multiplication.
    """
    raw_tbr = capsule.lithium_fraction * capsule.beryllium_multiplier * 0.8
    temp_penalty = max(0.5, 1 - (capsule.operating_temp_c - 400) / 1000)
    tbr = raw_tbr * temp_penalty
    inventory = neutron_flux * capsule.lithium_fraction * 1e-20
    extraction = min(1.0, capsule.helium_purge_rate * (1 + capsule.permeation_barrier_mm * 2))
    return CapsulePerformance(
        breeding_ratio=tbr,
        tritium_inventory_g=inventory,
        extraction_efficiency=extraction,
        self_sustaining=tbr > 1.05 and extraction > 0.8,
    )


def thermodynamic_extraction_velocity(
    D_T: float, delta: float, safety_factor: float = 1.1
) -> float:
    """Required vapor velocity: v_v > D_T / delta (Peclet > 1)."""
    return safety_factor * D_T / delta
