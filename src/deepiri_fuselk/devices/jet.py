"""JET (Joint European Torus) device profile and discharge presets.

Values approximate commonly-cited published JET parameters (JET/EUROfusion
technical figures, post-upgrade ITER-like wall era):
  - R0 ~= 2.96 m, a ~= 0.96 m (aspect ratio ~3.1, similar to ITER's)
  - elongation kappa ~= 1.7, triangularity delta ~= 0.4 (shaped divertor
    configurations)
  - Ip ~= 4.8-5.0 MA (upgraded PF/divertor coil capability)
  - Bt ~= 3.4-3.8 T
  - auxiliary heating ~= 40 MW (NBI + ICRH combined, enhanced performance
    campaigns; JET's original design heating power was lower, but upgraded
    NBI/ICRH systems in later campaigns reached ~40 MW combined)
  - Greenwald limit fraction target ~= 0.85 (typical H-mode operating point)
  - Troyon normalized beta limit beta_N ~= 2.8-3.0 (conventional literature
    value for shaped ELMy H-mode)
"""

from __future__ import annotations

from deepiri_fuselk.devices.presets import DischargePreset
from deepiri_fuselk.devices.profile import DeviceProfile

JET = DeviceProfile(
    name="JET",
    major_radius_m=2.96,
    minor_radius_m=0.96,
    elongation=1.7,
    triangularity=0.4,
    max_ip_ma=4.8,
    max_bt_t=3.6,
    max_heating_mw=40.0,
    greenwald_limit=0.85,
    troyon_beta_limit=2.8,
)

PRESETS: list[DischargePreset] = [
    DischargePreset(
        name="H-mode",
        duration_s=20.0,
        ip_waveform_ma=[(0.0, 0.0), (2.0, 4.8), (16.0, 4.8), (20.0, 0.0)],
        heating_waveform_mw=[(0.0, 0.0), (3.0, 40.0), (16.0, 40.0), (20.0, 0.0)],
        density_waveform_frac_greenwald=[(0.0, 0.2), (5.0, 0.85), (20.0, 0.85)],
        elongation_waveform=[(0.0, 1.2), (2.0, 1.7), (20.0, 1.7)],
    ),
    DischargePreset(
        name="L-mode",
        duration_s=20.0,
        ip_waveform_ma=[(0.0, 0.0), (2.0, 3.0), (16.0, 3.0), (20.0, 0.0)],
        heating_waveform_mw=[(0.0, 0.0), (3.0, 8.0), (16.0, 8.0), (20.0, 0.0)],
        density_waveform_frac_greenwald=[(0.0, 0.2), (5.0, 0.4), (20.0, 0.4)],
        elongation_waveform=[(0.0, 1.2), (2.0, 1.7), (20.0, 1.7)],
    ),
    DischargePreset(
        name="Density Limit",
        duration_s=20.0,
        ip_waveform_ma=[(0.0, 0.0), (2.0, 4.8), (16.0, 4.8), (20.0, 0.0)],
        heating_waveform_mw=[(0.0, 0.0), (3.0, 40.0), (16.0, 40.0), (20.0, 0.0)],
        density_waveform_frac_greenwald=[(0.0, 0.2), (8.0, 0.95), (14.0, 1.1), (20.0, 1.1)],
        elongation_waveform=[(0.0, 1.2), (2.0, 1.7), (20.0, 1.7)],
    ),
]
