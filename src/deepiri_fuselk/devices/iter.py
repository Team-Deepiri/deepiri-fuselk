"""ITER device profile and discharge presets.

Values approximate commonly-cited published ITER parameters (ITER
Physics Basis / ITER Organization technical figures):
  - R0 ~= 6.2 m, a ~= 2.0 m (aspect ratio ~3.1)
  - elongation kappa_95 ~= 1.7, triangularity delta_95 ~= 0.33
  - Ip ~= 15 MA (nominal inductive scenario)
  - Bt ~= 5.3 T (at R0)
  - auxiliary heating ~= 73 MW installed (NBI + ICRH + ECRH, baseline)
  - Greenwald limit fraction target ~= 0.85 (typical H-mode operating point)
  - Troyon normalized beta limit beta_N ~= 3.0 (conventional ELMy H-mode
    ballooning/kink limit used across tokamak literature)
"""

from __future__ import annotations

from deepiri_fuselk.devices.presets import DischargePreset
from deepiri_fuselk.devices.profile import DeviceProfile

ITER = DeviceProfile(
    name="ITER",
    major_radius_m=6.2,
    minor_radius_m=2.0,
    elongation=1.7,
    triangularity=0.33,
    max_ip_ma=15.0,
    max_bt_t=5.3,
    max_heating_mw=73.0,
    greenwald_limit=0.85,
    troyon_beta_limit=3.0,
)

PRESETS: list[DischargePreset] = [
    DischargePreset(
        name="H-mode",
        duration_s=400.0,
        ip_waveform_ma=[(0.0, 0.0), (80.0, 15.0), (350.0, 15.0), (400.0, 0.0)],
        heating_waveform_mw=[(0.0, 0.0), (100.0, 73.0), (350.0, 73.0), (400.0, 0.0)],
        density_waveform_frac_greenwald=[(0.0, 0.2), (120.0, 0.85), (400.0, 0.85)],
        elongation_waveform=[(0.0, 1.2), (80.0, 1.7), (400.0, 1.7)],
    ),
    DischargePreset(
        name="L-mode",
        duration_s=400.0,
        ip_waveform_ma=[(0.0, 0.0), (80.0, 10.0), (350.0, 10.0), (400.0, 0.0)],
        heating_waveform_mw=[(0.0, 0.0), (100.0, 20.0), (350.0, 20.0), (400.0, 0.0)],
        density_waveform_frac_greenwald=[(0.0, 0.2), (120.0, 0.4), (400.0, 0.4)],
        elongation_waveform=[(0.0, 1.2), (80.0, 1.7), (400.0, 1.7)],
    ),
    DischargePreset(
        name="Density Limit",
        duration_s=400.0,
        ip_waveform_ma=[(0.0, 0.0), (80.0, 15.0), (350.0, 15.0), (400.0, 0.0)],
        heating_waveform_mw=[(0.0, 0.0), (100.0, 73.0), (350.0, 73.0), (400.0, 0.0)],
        density_waveform_frac_greenwald=[(0.0, 0.2), (150.0, 0.95), (300.0, 1.05), (400.0, 1.05)],
        elongation_waveform=[(0.0, 1.2), (80.0, 1.7), (400.0, 1.7)],
    ),
]
