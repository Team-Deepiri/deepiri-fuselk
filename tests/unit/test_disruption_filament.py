"""Field-aligned disruption filament geometry tests."""

from __future__ import annotations

import numpy as np
from deepiri_fuselk.devices.registry import DeviceRegistry
from deepiri_fuselk.helix.disruption_filament import disruption_filament, midplane_pitch_rad


def test_iter_and_diiid_filaments_differ():
    reg = DeviceRegistry()
    iter_f = disruption_filament(reg.get("ITER"), q95=3.2)
    diiid_f = disruption_filament(reg.get("DIII-D"), q95=3.2)
    assert iter_f.device == "ITER"
    assert diiid_f.device == "DIII-D"
    # Length scales with minor radius → ITER filament longer
    assert iter_f.length > diiid_f.length
    # Directions are unit vectors
    for f in (iter_f, diiid_f):
        n = np.linalg.norm(f.direction)
        assert abs(n - 1.0) < 1e-6


def test_pitch_changes_with_q95():
    reg = DeviceRegistry()
    d = reg.get("JET")
    low_q = midplane_pitch_rad(d, q95=2.0)
    high_q = midplane_pitch_rad(d, q95=5.0)
    # Higher q → shallower pitch (more toroidal)
    assert high_q < low_q


def test_filament_dict_roundtrip_keys():
    reg = DeviceRegistry()
    payload = disruption_filament(reg.get("ITER")).to_dict()
    assert "direction" in payload and len(payload["direction"]) == 3
    assert "pitch_rad" in payload
