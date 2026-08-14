"""Device profile data model.

A `DeviceProfile` captures the physically-distinct geometric and operational
limit parameters of a tokamak device (major/minor radius, shaping, field and
current limits, and disruption-relevant limits). It is deliberately a flat,
frozen dataclass so it can be passed around fuselk's physics modules as a
cheap, hashable, immutable value object.

IMPORTANT: the field set below is a cross-team contract — other concurrent
work may depend on this exact shape. Do not add or rename fields here without
coordinating; add new device-specific data via new files/registries instead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceProfile:
    """Physical + operational-limit description of a tokamak device.

    Attributes:
        name: Human-readable device identifier (e.g. "ITER", "JET", "DIII-D").
        major_radius_m: Plasma major radius R0, in meters.
        minor_radius_m: Plasma minor radius a, in meters.
        elongation: Vertical elongation kappa of the plasma cross-section
            (1.0 = circular).
        triangularity: Plasma shape triangularity delta (0.0 = no triangularity).
        max_ip_ma: Maximum/nominal plasma current, in mega-amperes.
        max_bt_t: Maximum/nominal toroidal magnetic field at R0, in tesla.
        max_heating_mw: Maximum installed auxiliary heating power, in megawatts.
        greenwald_limit: Greenwald density limit fraction reference
            (n/n_GW = 1.0 is the nominal Greenwald density; used as a
            normalized reference point for density-limit disruption checks).
        troyon_beta_limit: Troyon normalized beta limit (beta_N), the
            MHD-stability-derived ceiling on normalized plasma pressure.
    """

    name: str
    major_radius_m: float
    minor_radius_m: float
    elongation: float
    triangularity: float
    max_ip_ma: float
    max_bt_t: float
    max_heating_mw: float
    greenwald_limit: float
    troyon_beta_limit: float

    @property
    def aspect_ratio(self) -> float:
        """R0 / a — convenience derived quantity, not stored."""
        return self.major_radius_m / self.minor_radius_m
