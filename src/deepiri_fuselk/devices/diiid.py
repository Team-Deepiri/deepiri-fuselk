"""DIII-D device profile and discharge presets.

Values approximate commonly-cited published DIII-D parameters (General
Atomics / DIII-D National Fusion Facility technical figures):
  - R0 ~= 1.67 m, a ~= 0.67 m (aspect ratio ~2.5, highly flexible shaping)
  - elongation kappa up to ~1.8-2.0, triangularity delta up to ~0.7-0.8 in
    high-shaping scenarios; representative "standard" shaped H-mode values
    used here: kappa ~= 1.8, delta ~= 0.6
  - Ip ~= 2.0-3.0 MA
  - Bt ~= 2.2 T (max)
  - auxiliary heating ~= 20-23 MW (NBI + ECH combined)
  - Greenwald limit fraction target ~= 0.8 (typical H-mode operating point)
  - Troyon normalized beta limit beta_N ~= 3.5-4.0 (DIII-D's strong shaping
    supports higher beta_N than lower-elongation devices; literature-cited
    advanced-tokamak scenarios reach beta_N ~4)
"""

from __future__ import annotations

from deepiri_fuselk.devices.presets import DischargePreset
from deepiri_fuselk.devices.profile import DeviceProfile

DIII_D = DeviceProfile(
    name="DIII-D",
    major_radius_m=1.67,
    minor_radius_m=0.67,
    elongation=1.8,
    triangularity=0.6,
    max_ip_ma=2.0,
    max_bt_t=2.2,
    max_heating_mw=20.0,
    greenwald_limit=0.8,
    troyon_beta_limit=3.5,
)

PRESETS: list[DischargePreset] = [
    DischargePreset(
        name="H-mode",
        duration_s=5.0,
        ip_waveform_ma=[(0.0, 0.0), (0.5, 2.0), (4.0, 2.0), (5.0, 0.0)],
        heating_waveform_mw=[(0.0, 0.0), (0.7, 20.0), (4.0, 20.0), (5.0, 0.0)],
        density_waveform_frac_greenwald=[(0.0, 0.2), (1.2, 0.8), (5.0, 0.8)],
        elongation_waveform=[(0.0, 1.2), (0.5, 1.8), (5.0, 1.8)],
    ),
    DischargePreset(
        name="L-mode",
        duration_s=5.0,
        ip_waveform_ma=[(0.0, 0.0), (0.5, 1.2), (4.0, 1.2), (5.0, 0.0)],
        heating_waveform_mw=[(0.0, 0.0), (0.7, 5.0), (4.0, 5.0), (5.0, 0.0)],
        density_waveform_frac_greenwald=[(0.0, 0.2), (1.2, 0.35), (5.0, 0.35)],
        elongation_waveform=[(0.0, 1.2), (0.5, 1.8), (5.0, 1.8)],
    ),
    DischargePreset(
        name="Density Limit",
        duration_s=5.0,
        ip_waveform_ma=[(0.0, 0.0), (0.5, 2.0), (4.0, 2.0), (5.0, 0.0)],
        heating_waveform_mw=[(0.0, 0.0), (0.7, 20.0), (4.0, 20.0), (5.0, 0.0)],
        density_waveform_frac_greenwald=[(0.0, 0.2), (2.0, 0.9), (3.5, 1.1), (5.0, 1.1)],
        elongation_waveform=[(0.0, 1.2), (0.5, 1.8), (5.0, 1.8)],
    ),
]
