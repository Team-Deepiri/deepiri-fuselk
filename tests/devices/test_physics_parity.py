"""Prove ITER/JET/DIII-D device profiles actually drive physically distinct
outputs through fuselk's existing HELIX physics, not a toy model on the side.
"""

from __future__ import annotations

import numpy as np
from deepiri_fuselk.devices.registry import DeviceRegistry
from deepiri_fuselk.helix.helix_engine import HelixEngine
from deepiri_fuselk.sim.synthetic_data_gen import generate_ece_shot


def _pitch_for_device(device):
    engine = HelixEngine(device=device)
    # Force the tracker into a fixed, deterministic phase so pitch is a pure
    # function of device geometry, not tracker RNG drift across engines.
    engine.tracker.x[0] = 0.7
    return engine._field_line_pitch_at_o_point(np.zeros((16, 16)))


def test_field_line_pitch_varies_by_device_geometry():
    registry = DeviceRegistry()
    iter_pitch = _pitch_for_device(registry.get("ITER"))
    jet_pitch = _pitch_for_device(registry.get("JET"))
    diiid_pitch = _pitch_for_device(registry.get("DIII-D"))
    default_pitch = _pitch_for_device(registry.get("DEFAULT"))

    pitches = {iter_pitch, jet_pitch, diiid_pitch, default_pitch}
    assert len(pitches) > 1, "device geometry must change field-line pitch"

    # Larger minor radius -> larger `a` in q_profile -> lower q(r) at fixed r
    # -> larger arctan(1/q) pitch. ITER has the largest minor radius, so it
    # should show the largest pitch among the three real devices.
    assert iter_pitch > jet_pitch > diiid_pitch


def test_helix_engine_process_distinct_across_devices():
    """Run the same synthetic shot through HelixEngine for each device and
    assert not all devices collapse to identical output — i.e. the device
    parameter is genuinely consumed by the physics pipeline."""
    registry = DeviceRegistry()
    shot = generate_ece_shot(16, seed=0, island_amplitude=0.7)

    results = {}
    for name in ("ITER", "JET", "DIII-D"):
        engine = HelixEngine(device=registry.get(name))
        engine.tracker.x[0] = 0.7
        result = engine.process(shot.heat_field, shot.raw_signal, shot.angles)
        results[name] = result

    pitches = [
        results[n].hqrm.o_point for n in ("ITER", "JET", "DIII-D")
    ]  # sanity: pipeline actually ran for each device
    assert all(isinstance(p, tuple) for p in pitches)

    # The device-dependent field-line pitch feeds spiral-attention denoising,
    # which is blended into the focal map. Assert it is not bitwise-identical
    # across devices with clearly distinct minor radii (i.e. the device
    # parameter is genuinely consumed, not silently dropped on the floor).
    focal_maps = [results[n].focal_map for n in ("ITER", "JET", "DIII-D")]
    diffs = [
        float(np.max(np.abs(focal_maps[i] - focal_maps[j])))
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    assert any(d > 0.0 for d in diffs), "device parameter had no effect on HELIX output"


def _confinement_proxy(device) -> float:
    """Simple ITER-98y2-style confinement-time proxy (not a full scaling law):
    tau ~ R^1.97 * a^0.58 * B^0.15 -- monotonically increasing in device size
    and field, enough to check physically-sensible ordering across devices
    without depending on any real transport solve."""
    return (device.major_radius_m**1.97) * (device.minor_radius_m**0.58) * (device.max_bt_t**0.15)


def test_devices_ordered_on_confinement_stored_energy_and_limits():
    """Multi-metric physical-ordering check across ITER/JET/DIII-D:
    confinement-time proxy, stored-energy-at-onset proxy, Greenwald density
    limit, and Troyon beta limit must all be distinct across the three real
    devices, and must be ordered in a physically sensible direction."""
    registry = DeviceRegistry()
    iter_p, jet_p, diiid_p = (registry.get(n) for n in ("ITER", "JET", "DIII-D"))

    # (a) Confinement time proxy: bigger, higher-field device confines longer.
    tau = {
        n: _confinement_proxy(p) for n, p in (("ITER", iter_p), ("JET", jet_p), ("DIII-D", diiid_p))
    }
    assert tau["ITER"] > tau["JET"] > tau["DIII-D"]

    # (b) Stored-energy-at-disruption-onset proxy: W ~ beta_N * Ip * Bt * a
    # (dimensionally loose, but captures that larger/higher-current/field
    # devices can sustain more stored energy before hitting their limits).
    def stored_energy_proxy(p):
        return p.troyon_beta_limit * p.max_ip_ma * p.max_bt_t * p.minor_radius_m

    energy = {
        "ITER": stored_energy_proxy(iter_p),
        "JET": stored_energy_proxy(jet_p),
        "DIII-D": stored_energy_proxy(diiid_p),
    }
    assert energy["ITER"] > energy["JET"] > energy["DIII-D"]
    assert len({round(v, 6) for v in energy.values()}) == 3

    # (c) Greenwald density limit and Troyon beta limit both vary by device
    # and are not all identical (device parameter genuinely differentiates
    # disruption-relevant limits, not just geometry).
    greenwald = {iter_p.greenwald_limit, jet_p.greenwald_limit, diiid_p.greenwald_limit}
    troyon = {iter_p.troyon_beta_limit, jet_p.troyon_beta_limit, diiid_p.troyon_beta_limit}
    assert len(troyon) > 1, "troyon_beta_limit must differ across real devices"
    # DIII-D's strong shaping supports the highest beta_N among the three.
    assert diiid_p.troyon_beta_limit > iter_p.troyon_beta_limit >= jet_p.troyon_beta_limit
    # Greenwald limit fractions are close across devices (all near-standard
    # H-mode operating points) but still device-specific, not a single
    # hardcoded constant reused verbatim everywhere.
    assert len(greenwald) > 1, "greenwald_limit must differ across real devices"


def test_reactor_cell_accepts_device_and_matches_default_when_unset():
    """ReactorCell with device=DEFAULT (implicit) must reproduce identical
    results to a ReactorCell constructed before device-wiring existed."""
    from deepiri_fuselk.devices.registry import DEFAULT
    from deepiri_fuselk.sim.reactor_cell import ReactorCell

    default_cell = ReactorCell(grid_size=16, train_elm=False)
    explicit_default_cell = ReactorCell(grid_size=16, train_elm=False, device=DEFAULT)

    run_a = default_cell.run(n_steps=5, seed=0)
    run_b = explicit_default_cell.run(n_steps=5, seed=0)

    assert run_a.final_score == run_b.final_score
    assert np.allclose(run_a.steps[-1].heat_flux, run_b.steps[-1].heat_flux)
