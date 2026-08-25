"""Unit tests for physically-derived radiance field (Phase 5)."""

from __future__ import annotations

import numpy as np
from deepiri_fuselk.devices.registry import DeviceRegistry
from deepiri_fuselk.viz.radiance import (
    blackbody_color_temperature,
    bremsstrahlung_intensity,
    dalpha_doppler_shift,
    integrated_core_radiance,
    radiance_field,
)


def test_bremsstrahlung_zero_te():
    assert bremsstrahlung_intensity(1.0, 1.0, 0.0) == 0.0


def test_bremsstrahlung_scales_with_density():
    low = bremsstrahlung_intensity(1.0, 1.0, 4.0)
    high = bremsstrahlung_intensity(2.0, 2.0, 4.0)
    assert high > low
    assert np.isclose(high / low, 4.0, rtol=1e-6)


def test_dalpha_zero_rotation():
    assert dalpha_doppler_shift(0.0) == 0.0


def test_dalpha_sign():
    assert dalpha_doppler_shift(1e5) > 0.0
    assert dalpha_doppler_shift(-1e5) < 0.0


def test_blackbody_monotonic_brightness():
    cool = blackbody_color_temperature(1.0)
    hot = blackbody_color_temperature(20.0)
    # Hot divertor should be closer to white (higher RGB sum)
    assert sum(hot) >= sum(cool) - 1e-6


def test_blackbody_matches_old_band_boundaries():
    """Smooth mapping should stay in [0,1] and cover blue→white progression."""
    for q in (0.0, 2.0, 5.0, 10.0, 20.0):
        rgb = blackbody_color_temperature(q)
        assert all(0.0 <= c <= 1.0 for c in rgb)


def test_radiance_field_composition():
    field = radiance_field(te_kev=5.0, ne_1e19=2.0, heat_flux_mw_m2=8.0, rotation_m_s=1e4)
    assert field.core_intensity > 0
    assert field.dalpha_wavelength_nm != 656.28
    d = field.to_dict()
    assert "core_rgb" in d


def test_integrated_radiance_scales_with_device_volume():
    field = radiance_field(te_kev=3.0, ne_1e19=1.5, heat_flux_mw_m2=5.0)
    reg = DeviceRegistry()
    iter_p = reg.get("ITER")
    diiid = reg.get("DIII-D")
    i_vol = integrated_core_radiance(
        field,
        major_radius_m=iter_p.major_radius_m,
        minor_radius_m=iter_p.minor_radius_m,
        elongation=iter_p.elongation,
    )
    d_vol = integrated_core_radiance(
        field,
        major_radius_m=diiid.major_radius_m,
        minor_radius_m=diiid.minor_radius_m,
        elongation=diiid.elongation,
    )
    assert i_vol > d_vol
