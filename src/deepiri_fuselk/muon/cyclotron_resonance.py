"""Cyclotron resonance muon maintenance."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CyclotronConfig:
    magnetic_field_t: float = 5.0
    rf_frequency_hz: float = 1e8
    amplitude_v: float = 1000.0


def cyclotron_frequency(
    config: CyclotronConfig, charge: float = 1.6e-19, mass: float = 1.88e-28
) -> float:
    """f = qB / (2 pi m) for muonic ion."""
    import math

    return charge * config.magnetic_field_t / (2 * math.pi * mass)


def resonance_match(config: CyclotronConfig) -> bool:
    """True if RF matches cyclotron frequency within 10%."""
    f_c = cyclotron_frequency(config)
    return abs(config.rf_frequency_hz - f_c) / f_c < 0.1
