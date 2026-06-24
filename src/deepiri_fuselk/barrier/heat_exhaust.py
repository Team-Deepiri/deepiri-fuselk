"""Heat exhaust and wall protection experiment models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BrineCoatingResult:
    """Simulation result for salinity barrier coating hypothesis."""

    thermal_resistance: float
    corrosion_risk: float
    effective_heat_reduction: float
    viable: bool


@dataclass
class DetachmentState:
    """Plasma detachment regime metrics."""

    upstream_temp_ev: float
    target_heat_flux_mw_m2: float
    detached: bool
    radiation_fraction: float


def evaluate_brine_coating(
    salinity_ppt: float = 35.0,
    coating_thickness_mm: float = 0.5,
    wall_temp_c: float = 600.0,
) -> BrineCoatingResult:
    """
    Simulation-only evaluation of brine/salinity wall coating hypothesis.

    NOT validated for real reactor use — research exploration only.
    """
    thermal_resistance = coating_thickness_mm * 0.01 * (1 + salinity_ppt / 100)
    corrosion_risk = min(1.0, wall_temp_c / 1000 + salinity_ppt / 200)
    heat_reduction = min(0.5, thermal_resistance * 0.3)
    viable = corrosion_risk < 0.7 and heat_reduction > 0.05
    return BrineCoatingResult(
        thermal_resistance=thermal_resistance,
        corrosion_risk=corrosion_risk,
        effective_heat_reduction=heat_reduction,
        viable=viable,
    )


def evaluate_detachment(
    n_sep: float = 1e19,
    T_sep_ev: float = 50.0,
    gas_puff_rate: float = 0.0,
) -> DetachmentState:
    """Evaluate scrape-off layer detachment state."""
    radiation = min(0.95, 0.3 + 0.1 * gas_puff_rate + 0.005 * T_sep_ev)
    q_target = max(0.1, 10.0 * (1 - radiation) * T_sep_ev / 100)
    detached = radiation > 0.6 and q_target < 5.0
    return DetachmentState(
        upstream_temp_ev=T_sep_ev,
        target_heat_flux_mw_m2=q_target,
        detached=detached,
        radiation_fraction=radiation,
    )


def strike_point_sweep_path(
    n_points: int = 100,
    amplitude: float = 0.1,
    frequency_hz: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Lissajous strike point sweep for vent circularization."""
    t = np.linspace(0, 1, n_points)
    R = amplitude * np.sin(2 * np.pi * frequency_hz * t)
    Z = amplitude * np.cos(2 * np.pi * frequency_hz * t * 1.3)
    return R, Z
