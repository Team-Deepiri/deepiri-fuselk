"""Full tokamak discharge pulse — breakdown through flat-top to ramp-down.

This is the immersive reactor clock: device geometry + DischargePreset
waveforms drive a physically ordered power-balance so the control room
feels like an operating machine (ITER / JET / DIII-D), not a random KPI
ticker.

Phases
------
breakdown → ramp_up → flat_top → ramp_down → ended
                                 ↘ disrupted (density-limit / MHD trip)

Power balance (0D, device-scaled)
---------------------------------
P_aux from preset heating waveform; P_ohm ~ Ip² R; P_α from DT proxy;
P_rad from bremsstrahlung volume; W_th / τ_E from Troyon + IPB98-ish
scaling; Q = P_fusion / P_aux. Divertor peak heat from SOL power share.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from deepiri_fuselk.devices.presets import DischargePreset
from deepiri_fuselk.devices.profile import DeviceProfile
from deepiri_fuselk.devices.registry import DeviceRegistry
from deepiri_fuselk.viz.radiance import bremsstrahlung_power_mw

_MU0 = 4.0e-7 * np.pi
_E_DT_MEV = 17.6
_ALPHA_FRAC = 0.2  # fraction of DT power carried by alphas (3.5/17.6)


class PulsePhase(str, Enum):
    BREAKDOWN = "breakdown"
    RAMP_UP = "ramp_up"
    FLAT_TOP = "flat_top"
    RAMP_DOWN = "ramp_down"
    DISRUPTED = "disrupted"
    ENDED = "ended"


@dataclass
class PulseState:
    """One instant on the discharge timeline — the immersive reactor snapshot."""

    device: str
    preset: str
    phase: str
    t_s: float
    duration_s: float
    progress: float  # 0..1 of programmed pulse
    ip_ma: float
    bt_t: float
    heating_mw: float
    ne_bar_1e19: float
    greenwald_fraction: float
    q95: float
    beta_n: float
    elongation: float
    te0_kev: float
    w_th_mj: float
    tau_e_s: float
    p_ohm_mw: float
    p_aux_mw: float
    p_alpha_mw: float
    p_fusion_mw: float
    p_rad_mw: float
    p_loss_mw: float
    p_div_mw: float
    divertor_peak_mw_m2: float
    q_factor: float
    neutron_rate_1e18: float
    disruption_risk: float
    alive: bool
    narrative: str

    def to_dict(self) -> dict:
        return asdict(self)


class ReactorPulseEngine:
    """Advance a device-faithful discharge pulse with power-balance physics."""

    def __init__(
        self,
        device: str | DeviceProfile = "ITER",
        preset: str = "H-mode",
        *,
        dt_s: float = 0.5,
        seed: int = 42,
    ) -> None:
        registry = DeviceRegistry()
        if isinstance(device, DeviceProfile):
            self.device = device
        else:
            self.device = registry.get(device)
        presets = {p.name: p for p in registry.get_presets(self.device.name)}
        if not presets:
            presets = {p.name: p for p in registry.get_presets("DEFAULT")}
        self.preset: DischargePreset = presets.get(preset) or next(iter(presets.values()))
        self.dt_s = max(dt_s, 1e-3)
        self.rng = np.random.default_rng(seed)
        self.t_s = 0.0
        self._disrupted = False
        self._ended = False
        self._state = self._evaluate()

    @property
    def state(self) -> PulseState:
        return self._state

    def reset(self, *, seed: int | None = None) -> PulseState:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t_s = 0.0
        self._disrupted = False
        self._ended = False
        self._state = self._evaluate()
        return self._state

    def step(self, n: int = 1) -> PulseState:
        for _ in range(max(n, 1)):
            if self._ended or self._disrupted:
                break
            self.t_s += self.dt_s
            if self.t_s >= self.preset.duration_s:
                self.t_s = self.preset.duration_s
                self._ended = True
            self._state = self._evaluate()
            if self._state.disruption_risk > 0.92 and self._state.phase in (
                PulsePhase.FLAT_TOP.value,
                PulsePhase.RAMP_UP.value,
            ):
                # Stochastic trip near the hard limit — density-limit / MHD.
                if self.rng.random() < 0.08 * self._state.disruption_risk:
                    self._disrupted = True
                    self._state = self._evaluate()
                    break
        return self._state

    def run(self, *, max_steps: int = 5000) -> list[PulseState]:
        frames = [self.reset()]
        for _ in range(max_steps):
            st = self.step()
            frames.append(st)
            if st.phase in (PulsePhase.ENDED.value, PulsePhase.DISRUPTED.value):
                break
        return frames

    def _phase(self, t: float, ip: float, heat: float) -> PulsePhase:
        if self._disrupted:
            return PulsePhase.DISRUPTED
        if self._ended or t >= self.preset.duration_s:
            return PulsePhase.ENDED
        dur = max(self.preset.duration_s, 1e-6)
        if t < 0.05 * dur or ip < 0.05 * self.device.max_ip_ma:
            return PulsePhase.BREAKDOWN
        if t > 0.85 * dur and ip < 0.7 * self.device.max_ip_ma:
            return PulsePhase.RAMP_DOWN
        ip_tgt = max(self.preset.ip_at(t), 1e-6)
        heat_tgt = max(self.preset.heating_at(t), 1e-6)
        if abs(ip - ip_tgt) / ip_tgt < 0.08 and heat > 0.5 * heat_tgt and t > 0.2 * dur:
            return PulsePhase.FLAT_TOP
        if t < 0.35 * dur:
            return PulsePhase.RAMP_UP
        if heat < 0.25 * self.device.max_heating_mw and t > 0.7 * dur:
            return PulsePhase.RAMP_DOWN
        return PulsePhase.FLAT_TOP if heat > 0.4 * heat_tgt else PulsePhase.RAMP_UP

    def _evaluate(self) -> PulseState:
        d = self.device
        p = self.preset
        t = self.t_s
        a = max(d.minor_radius_m, 1e-3)
        r0 = max(d.major_radius_m, 1e-3)
        bt = d.max_bt_t
        kappa = max(p.elongation_at(t), 1.0)

        ip = max(p.ip_at(t), 0.0)
        heat = max(p.heating_at(t), 0.0)
        dens_frac = max(p.density_frac_at(t), 0.0)

        n_gw_1e19 = 10.0 * max(ip, 0.05) / (np.pi * a**2)
        ne = n_gw_1e19 * dens_frac
        gwf = dens_frac

        q_cyl = 5.0 * a**2 * bt / (r0 * max(ip, 0.05))
        q95 = q_cyl * (1.0 + kappa**2) / 2.0

        # IPB98(y,2)-flavoured confinement proxy
        tau_e = (
            0.0562
            * (max(ip, 0.1) ** 0.93)
            * (bt**0.15)
            * (max(ne, 0.1) ** 0.41)
            * (r0**1.97)
            * (a**0.58)
            * (kappa**0.78)
            / max(heat + 1.0, 1.0) ** 0.69
        )
        tau_e = float(np.clip(tau_e, 0.01, 8.0))

        volume = 2.0 * np.pi**2 * r0 * a**2 * kappa
        p_ohm = max(0.2, 0.15 * ip**2 * (r0 / a) / max(bt, 0.5))
        p_aux = heat

        te0 = float(np.clip((p_aux + p_ohm) * tau_e / max(ne * volume * 0.02, 1e-3), 0.5, 40.0))

        # DT fusion power proxy — Lawson-ish n²T²V, scaled for immersive
        # device-ordered flat-tops (DIII-D tens of MW, ITER hundreds).
        n_norm = max(ne / 3.0, 0.05)
        t_norm = max(te0 / 3.0, 0.05)
        v_norm = max(volume / 8.0, 0.05)
        device_gain = 0.5 + 1.5 * (d.max_ip_ma / 15.0)
        heating_gate = float(np.clip(heat / max(0.25 * d.max_heating_mw, 1.0), 0.05, 1.0))
        p_fusion = float(
            np.clip(
                40.0 * (n_norm**2) * (t_norm**2) * v_norm * device_gain * heating_gate, 0.0, 900.0
            )
        )
        if self.preset.name == "L-mode":
            p_fusion *= 0.35
        if self.preset.name == "Density Limit":
            p_fusion *= 1.15
        p_alpha = _ALPHA_FRAC * p_fusion

        p_rad = bremsstrahlung_power_mw(ne, te0, volume)
        p_in = p_ohm + p_aux + p_alpha
        p_loss = max(0.1, p_in - 0.5 * p_rad)
        w_th = p_loss * tau_e
        beta_t = (2.0 * _MU0 * (w_th * 1e6 / 1.5) / max(volume, 1e-6)) / max(bt**2, 1e-6)
        beta_n = float(
            np.clip(100.0 * beta_t * a * bt / max(ip, 0.05), 0.0, d.troyon_beta_limit * 1.2)
        )

        p_div = 0.5 * p_loss
        lambda_q = 0.003 * (1.0 + 0.5 * (d.name == "ITER"))
        area = max(2.0 * np.pi * r0 * lambda_q * 2.0, 0.05)
        divertor_peak = p_div / area

        q_factor = p_fusion / max(p_aux, 0.5) if p_aux > 0.5 else 0.0
        neutrons = p_fusion / _E_DT_MEV * 0.1

        risk = 0.0
        risk += float(np.clip((gwf - d.greenwald_limit) / 0.2, 0.0, 1.0)) * 0.55
        risk += float(np.clip((2.0 - q95) / 2.0, 0.0, 1.0)) * 0.25
        risk += (
            float(
                np.clip(
                    (beta_n - d.troyon_beta_limit * 0.9) / (0.2 * d.troyon_beta_limit),
                    0.0,
                    1.0,
                )
            )
            * 0.2
        )
        if self._disrupted:
            risk = 1.0
        risk = float(np.clip(risk, 0.0, 1.0))

        phase = self._phase(t, ip, heat)
        alive = phase not in (PulsePhase.DISRUPTED, PulsePhase.ENDED) and not self._disrupted
        if self._disrupted:
            p_fusion *= 0.05
            w_th *= 0.1
            te0 *= 0.2
            divertor_peak *= 3.0
            q_factor = 0.0
            neutrons *= 0.05
            ip *= 0.2

        narrative = _narrative(phase, d.name, p.name, gwf, q_factor, divertor_peak)

        return PulseState(
            device=d.name,
            preset=p.name,
            phase=phase.value,
            t_s=float(t),
            duration_s=float(p.duration_s),
            progress=float(np.clip(t / max(p.duration_s, 1e-6), 0.0, 1.0)),
            ip_ma=float(ip),
            bt_t=float(bt),
            heating_mw=float(heat),
            ne_bar_1e19=float(ne),
            greenwald_fraction=float(gwf),
            q95=float(q95),
            beta_n=float(beta_n),
            elongation=float(kappa),
            te0_kev=float(te0),
            w_th_mj=float(w_th),
            tau_e_s=float(tau_e),
            p_ohm_mw=float(p_ohm),
            p_aux_mw=float(p_aux),
            p_alpha_mw=float(p_alpha),
            p_fusion_mw=float(p_fusion),
            p_rad_mw=float(p_rad),
            p_loss_mw=float(p_loss),
            p_div_mw=float(p_div),
            divertor_peak_mw_m2=float(divertor_peak),
            q_factor=float(q_factor),
            neutron_rate_1e18=float(neutrons),
            disruption_risk=risk,
            alive=alive,
            narrative=narrative,
        )


def _narrative(
    phase: PulsePhase, device: str, preset: str, gwf: float, q: float, qdiv: float
) -> str:
    if phase is PulsePhase.BREAKDOWN:
        return f"{device} {preset}: breakdown / current initiation."
    if phase is PulsePhase.RAMP_UP:
        return f"{device}: ramping Ip and heating toward flat-top."
    if phase is PulsePhase.FLAT_TOP:
        return f"{device} flat-top · n/n_GW={gwf:.2f} · Q~{q:.2f} · divertor ~{qdiv:.1f} MW/m2."
    if phase is PulsePhase.RAMP_DOWN:
        return f"{device}: controlled ramp-down — shedding stored energy."
    if phase is PulsePhase.DISRUPTED:
        return f"{device} DISRUPTED — plasma current collapse / thermal quench proxy."
    return f"{device} pulse ended."
