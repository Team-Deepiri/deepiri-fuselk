"""Phase 6 — radiance conservation vs SimulationFrame.p_rad_mw.

Gauges and port-view glow must be the *same* continuum model. Absolute
bremsstrahlung accuracy is engineering-grade (see radiance.py docstring);
identity across ITER/JET/DIII-D × presets is the hard requirement.
"""

from __future__ import annotations

import numpy as np
from deepiri_fuselk.devices.registry import DeviceRegistry
from deepiri_fuselk.viz.radiance import (
    CONSERVATION_TOL,
    bremsstrahlung_power_mw,
    integrated_core_radiance,
    plasma_volume_m3,
    radiance_from_frame,
)
from deepiri_fuselk.viz.simulation_engine import LiveSimulation

_DEVICES = ("ITER", "JET", "DIII-D")
_PRESETS = ("H-mode", "L-mode", "Density Limit")


def test_conservation_across_devices_and_presets():
    """Integrated radiance tracks frame.p_rad_mw within CONSERVATION_TOL."""
    for device in _DEVICES:
        for preset in _PRESETS:
            sim = LiveSimulation(grid_size=16, device=device, preset=preset)
            frame = sim.step()
            field = radiance_from_frame(frame)
            integrated = integrated_core_radiance(field)
            # Field self-consistency
            assert abs(field.p_rad_mw - integrated) <= CONSERVATION_TOL * max(1.0, field.p_rad_mw)
            # Gauge vs render identity (same bremsstrahlung model)
            denom = max(abs(frame.p_rad_mw), 1e-12)
            rel = abs(field.p_rad_mw - frame.p_rad_mw) / denom
            assert rel <= 1e-6, (
                f"{device}/{preset}: radiance {field.p_rad_mw} vs p_rad {frame.p_rad_mw} "
                f"(rel={rel})"
            )


def test_larger_device_radiates_more_at_same_ne_te():
    """ITER volume >> DIII-D → higher P_rad at fixed n, Te (sanity)."""
    reg = DeviceRegistry()
    te, ne = 8.0, 5.0
    powers = {}
    for name in ("ITER", "DIII-D"):
        d = reg.get(name)
        v = plasma_volume_m3(d.major_radius_m, d.minor_radius_m, d.elongation)
        powers[name] = bremsstrahlung_power_mw(ne, te, v)
    assert powers["ITER"] > powers["DIII-D"]


def test_p_rad_positive_on_live_frames():
    for device in _DEVICES:
        frame = LiveSimulation(grid_size=16, device=device, preset="H-mode").step()
        assert frame.p_rad_mw > 0.0
        assert np.isfinite(frame.p_rad_mw)
