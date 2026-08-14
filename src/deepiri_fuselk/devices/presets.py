"""Discharge (shot) presets: target waveforms for a simulated pulse.

A `DischargePreset` describes the *intended* time evolution of a discharge —
not the physics outcome — as a set of simple piecewise-linear waveforms.
Each waveform is a list of ``(time_s, value)`` breakpoints; the value at any
time between breakpoints is linearly interpolated, the value before the
first breakpoint is the first breakpoint's value, and the value after the
last breakpoint is the last breakpoint's value ("hold" extrapolation).

This keeps the preset representation trivial to construct, serialize, and
plot, while still being expressive enough for typical H-mode / L-mode /
density-limit target scenarios (ramp-up, flat-top, ramp-down).
"""

from __future__ import annotations

from dataclasses import dataclass, field

Breakpoint = tuple[float, float]


def interpolate(waveform: list[Breakpoint], t: float) -> float:
    """Linearly interpolate a waveform at time ``t`` (seconds).

    Values before the first breakpoint hold at the first value; values after
    the last breakpoint hold at the last value.
    """
    if not waveform:
        return 0.0
    if t <= waveform[0][0]:
        return waveform[0][1]
    if t >= waveform[-1][0]:
        return waveform[-1][1]
    for (t0, v0), (t1, v1) in zip(waveform, waveform[1:], strict=False):
        if t0 <= t <= t1:
            if t1 == t0:
                return v1
            frac = (t - t0) / (t1 - t0)
            return v0 + frac * (v1 - v0)
    return waveform[-1][1]


@dataclass(frozen=True)
class DischargePreset:
    """Target waveforms for a single simulated discharge/pulse.

    Attributes:
        name: One of "H-mode", "L-mode", "Density Limit".
        duration_s: Total pulse duration in seconds.
        ip_waveform_ma: Plasma current target waveform, (time_s, MA) points.
        heating_waveform_mw: Auxiliary heating power target waveform,
            (time_s, MW) points.
        density_waveform_frac_greenwald: Line-averaged density target,
            expressed as a fraction of the device's Greenwald limit,
            (time_s, n/n_GW) points.
        elongation_waveform: Plasma elongation target waveform,
            (time_s, kappa) points.
    """

    name: str
    duration_s: float
    ip_waveform_ma: list[Breakpoint] = field(default_factory=list)
    heating_waveform_mw: list[Breakpoint] = field(default_factory=list)
    density_waveform_frac_greenwald: list[Breakpoint] = field(default_factory=list)
    elongation_waveform: list[Breakpoint] = field(default_factory=list)

    def ip_at(self, t: float) -> float:
        return interpolate(self.ip_waveform_ma, t)

    def heating_at(self, t: float) -> float:
        return interpolate(self.heating_waveform_mw, t)

    def density_frac_at(self, t: float) -> float:
        return interpolate(self.density_waveform_frac_greenwald, t)

    def elongation_at(self, t: float) -> float:
        return interpolate(self.elongation_waveform, t)
