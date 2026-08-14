"""Tests for the device profile registry."""

from deepiri_fuselk.devices.registry import DEFAULT, DeviceRegistry


def test_list_devices_contains_all_expected():
    registry = DeviceRegistry()
    devices = registry.list_devices()
    assert "DEFAULT" in devices
    assert "ITER" in devices
    assert "JET" in devices
    assert "DIII-D" in devices
    assert len(devices) == len(set(devices))


def test_get_returns_matching_profile():
    registry = DeviceRegistry()
    for name in ("DEFAULT", "ITER", "JET", "DIII-D"):
        profile = registry.get(name)
        assert profile.name == name


def test_get_unknown_device_raises():
    registry = DeviceRegistry()
    try:
        registry.get("NOT-A-DEVICE")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")


def test_get_presets_nonempty_for_all_devices():
    registry = DeviceRegistry()
    for name in registry.list_devices():
        presets = registry.get_presets(name)
        assert len(presets) >= 1
        for preset in presets:
            assert preset.duration_s > 0


def test_default_profile_matches_hardcoded_current_behavior():
    """
    DEFAULT must exactly encode fuselk's current implicit geometry so that
    omitting `device=` anywhere reproduces existing behavior.

    - minor_radius_m=1.0 matches `q_profile`'s default minor radius `a=1.0`
      (src/deepiri_fuselk/helix/coordinate_mapper.py:8), which is the value
      reached via `field_line_pitch` (coordinate_mapper.py:28) from
      `HelixEngine._field_line_pitch_at_o_point`
      (src/deepiri_fuselk/helix/helix_engine.py:72), the only live
      geometry-shaped constant in the wired physics modules.
    - troyon_beta_limit=3.0 matches `mhd_stability_margin`'s default
      `beta_limit=3.0` (src/deepiri_fuselk/sim/fusion_kpis.py:54), the one
      genuine MHD-stability-limit constant hardcoded in the current
      codebase.
    - elongation=1.0 / triangularity=0.0: no shaping term exists in the
      current physics code, so these are the no-shaping identity values.
    """
    assert DEFAULT.name == "DEFAULT"
    assert DEFAULT.minor_radius_m == 1.0
    assert DEFAULT.major_radius_m == 1.0
    assert DEFAULT.elongation == 1.0
    assert DEFAULT.triangularity == 0.0
    assert DEFAULT.troyon_beta_limit == 3.0


def test_device_profiles_are_physically_distinct():
    registry = DeviceRegistry()
    iter_p = registry.get("ITER")
    jet_p = registry.get("JET")
    diiid_p = registry.get("DIII-D")

    # Real devices strictly ordered by size: ITER > JET > DIII-D.
    assert iter_p.major_radius_m > jet_p.major_radius_m > diiid_p.major_radius_m
    assert iter_p.minor_radius_m > jet_p.minor_radius_m > diiid_p.minor_radius_m
    assert iter_p.max_ip_ma > jet_p.max_ip_ma > diiid_p.max_ip_ma
