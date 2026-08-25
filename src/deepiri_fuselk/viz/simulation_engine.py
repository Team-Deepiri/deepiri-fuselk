"""Stateful simulation engine for live dashboard visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from deepiri_fuselk.devices.profile import DeviceProfile
from deepiri_fuselk.devices.registry import DEFAULT as DEFAULT_DEVICE
from deepiri_fuselk.devices.registry import DeviceRegistry
from deepiri_fuselk.helix.helix_engine import HelixResult
from deepiri_fuselk.models.disruption_detector import DisruptionAssessment
from deepiri_fuselk.models.elm_predictor import ELMPrediction
from deepiri_fuselk.sim.fusion_cell import FusionCell, FusionCellReport
from deepiri_fuselk.sim.reactor_cell import ReactorStep

if TYPE_CHECKING:
    from deepiri_fuselk.sim.reactor_pulse import PulseState, ReactorPulseEngine
    from deepiri_fuselk.sim.shot_replay import ShotReplayResult

_REGISTRY = DeviceRegistry()


def get_device(name: str) -> DeviceProfile:
    """Look up a device profile by name, falling back to DEFAULT if unknown."""
    try:
        return _REGISTRY.get(name)
    except KeyError:
        return DEFAULT_DEVICE


def list_device_names() -> list[str]:
    """All registered real device names (excludes the DEFAULT placeholder)."""
    return [n for n in _REGISTRY.list_devices() if n != "DEFAULT"]


def get_preset_names(device_name: str) -> list[str]:
    try:
        return [p.name for p in _REGISTRY.get_presets(device_name)]
    except KeyError:
        return [p.name for p in _REGISTRY.get_presets("DEFAULT")]


@dataclass
class SimulationFrame:
    step: int
    seed: int
    raw_heat: np.ndarray
    helix: HelixResult
    elm: ELMPrediction
    disruption: DisruptionAssessment
    controlled_heat: np.ndarray
    action: str
    fusion_score: float
    tbr: float
    muon_fpm: float
    peclet: float
    elm_free_fraction: float
    divertor_uniformity: float
    active_device: DeviceProfile = field(default_factory=lambda: DEFAULT_DEVICE)
    active_preset: str = "H-mode"
    # Derived tokamak-style diagnostics (lightweight proxy model built from the
    # existing FusionCell/HELIX KPIs, scaled by the active device's limits).
    ip_ma: float = 0.0
    beta_n: float = 0.0
    li: float = 0.0
    d_alpha: float = 0.0
    q95: float = 0.0
    greenwald_fraction: float = 0.0
    te0_kev: float = 0.0
    ne_bar_1e19: float = 0.0
    w_th_mj: float = 0.0
    tau_e_s: float = 0.0
    p_oh_mw: float = 0.0
    p_nbi_mw: float = 0.0
    p_ech_mw: float = 0.0
    p_rad_mw: float = 0.0
    p_loss_mw: float = 0.0
    neutron_rate_1e18: float = 0.0
    q_plasma: float = 0.0
    target_ip_ma: float = 0.0
    target_beta_n: float = 0.0
    target_li: float = 0.0
    target_d_alpha: float = 0.0
    # Shot-scrub identity (shared clock with Shot Workbench / ODL labels).
    mode: str = "live"  # "live" | "scrub" | "pulse"
    shot_id: str | None = None
    scrub_index: int | None = None
    scrub_n: int | None = None
    time_s: float | None = None
    odl_label: int | None = None
    density: float | None = None
    # Immersive reactor pulse theatre
    pulse_phase: str | None = None
    pulse_progress: float | None = None
    pulse_duration_s: float | None = None
    p_fusion_mw: float = 0.0
    p_alpha_mw: float = 0.0
    q_factor: float = 0.0
    divertor_peak_mw_m2: float = 0.0
    pulse_narrative: str | None = None
    pulse_alive: bool | None = None


@dataclass
class SimulationState:
    step_count: int = 0
    seed: int = 42
    elm_probs: list[float] = field(default_factory=list)
    fusion_score: float = 0.0
    report: FusionCellReport | None = None


# Preset-shape targets used to synthesize plausible actual/target traces from
# the underlying FusionCell KPIs (fractions of device limits). Keyed by the
# discharge-preset name (H-mode / L-mode / Density Limit). ``dens_frac`` is
# the target line-averaged density as a fraction of the (physically derived,
# see step()) Greenwald limit for the active device.
_PRESET_TARGETS: dict[str, dict[str, float]] = {
    "H-mode": {"ip_frac": 0.85, "beta_n_frac": 0.75, "li": 0.85, "d_alpha": 0.3, "dens_frac": 0.75},
    "L-mode": {"ip_frac": 0.55, "beta_n_frac": 0.4, "li": 1.05, "d_alpha": 0.6, "dens_frac": 0.5},
    "Density Limit": {
        "ip_frac": 0.7,
        "beta_n_frac": 0.55,
        "li": 0.95,
        "d_alpha": 0.9,
        "dens_frac": 0.95,
    },
}

_MU0 = 4.0e-7 * np.pi  # vacuum permeability, T*m/A


class LiveSimulation:
    """Step FusionCell/ReactorCell for real-time dashboard updates.

    When a shot is attached, ``step`` / ``seek`` scrub the archived discharge
    through the same HELIX → disruption → Venturi stack as Shot Workbench —
    so the port view, gauges, and ODL labels share one clock.
    """

    def __init__(
        self,
        grid_size: int = 24,
        device: str = "DEFAULT",
        preset: str = "H-mode",
    ) -> None:
        self.grid_size = grid_size
        self.cell = FusionCell(grid_size=grid_size, train_elm=False)
        self.state = SimulationState()
        self._last: SimulationFrame | None = None
        self.device: DeviceProfile = get_device(device)
        self.preset = preset if preset in _PRESET_TARGETS else "H-mode"
        self._scrub_result: ShotReplayResult | None = None
        self._scrub_index = 0
        self._pulse: ReactorPulseEngine | None = None

    @property
    def scrub_active(self) -> bool:
        return self._scrub_result is not None and len(self._scrub_result.frames) > 0

    @property
    def pulse_active(self) -> bool:
        return self._pulse is not None

    def set_device(self, name: str) -> None:
        """Switch the active device profile (drives gauge limits/equilibrium shape)."""
        self.device = get_device(name)

    def set_preset(self, name: str) -> None:
        """Switch the active discharge preset (drives target-vs-actual traces)."""
        if name in _PRESET_TARGETS:
            self.preset = name

    def clear_scrub(self) -> None:
        """Drop attached scrub without stepping (used by API reset)."""
        self._scrub_result = None
        self._scrub_index = 0
        self._pulse = None

    def attach_shot(
        self,
        shot: str | Path,
        *,
        n_steps: int = 24,
        seed: int = 42,
        data_root: Path | None = None,
        ensure_data: bool = False,
    ) -> SimulationFrame:
        """Precompute a ShotReplayer scrub and seek to the first frame."""
        from deepiri_fuselk.sim.shot_replay import ShotReplayer
        from deepiri_fuselk.sim.shot_workbench import resolve_shot_path

        self._pulse = None
        path = resolve_shot_path(shot, data_root=data_root, ensure_data=ensure_data)
        result = ShotReplayer(grid_size=self.grid_size).scrub(path=path, n_steps=n_steps, seed=seed)
        self._scrub_result = result
        self._scrub_index = 0
        if result.device and result.device in _REGISTRY.list_devices():
            self.device = get_device(result.device)
        self.state = SimulationState(seed=seed)
        return self.seek(0)

    def detach_shot(self) -> SimulationFrame:
        """Leave scrub mode and resume synthetic live stepping."""
        self._scrub_result = None
        self._scrub_index = 0
        return self.reset(seed=self.state.seed)

    def start_pulse(
        self,
        device: str | None = None,
        preset: str | None = None,
        *,
        dt_s: float = 1.0,
        seed: int = 42,
    ) -> SimulationFrame:
        """Enter immersive reactor-pulse mode (full discharge lifecycle)."""
        from deepiri_fuselk.sim.reactor_pulse import ReactorPulseEngine

        self.clear_scrub()
        if device:
            self.set_device(device)
        if preset:
            self.set_preset(preset)
        self._pulse = ReactorPulseEngine(
            device=self.device,
            preset=self.preset,
            dt_s=dt_s,
            seed=seed,
        )
        self.state = SimulationState(seed=seed)
        return self._frame_from_pulse(self._pulse.state, seed=seed)

    def stop_pulse(self) -> SimulationFrame:
        """Leave pulse theatre and resume synthetic live stepping."""
        self._pulse = None
        return self.reset(seed=self.state.seed)

    def seek(self, index: int) -> SimulationFrame:
        """Jump to a scrub index (requires an attached shot)."""
        if not self.scrub_active:
            raise RuntimeError("no shot attached — call attach_shot first")
        assert self._scrub_result is not None
        n = len(self._scrub_result.frames)
        self._scrub_index = int(np.clip(index, 0, n - 1))
        fr = self._scrub_result.frames[self._scrub_index]
        self.state.step_count = fr.index + 1
        self.state.elm_probs.append(fr.step.disruption.probability)
        self.state.fusion_score = fr.step.kpis.score()
        return self._frame_from_step(
            fr.step,
            seed=fr.step.seed,
            mode="scrub",
            shot_id=self._scrub_result.shot_id,
            scrub_index=fr.index,
            scrub_n=n,
            time_s=fr.time_s,
            odl_label=fr.odl_label,
            density=fr.density,
        )

    def reset(self, seed: int = 42) -> SimulationFrame:
        if self.pulse_active and self._pulse is not None:
            self._pulse.reset(seed=seed)
            self.state = SimulationState(seed=seed)
            return self._frame_from_pulse(self._pulse.state, seed=seed)
        if self.scrub_active:
            self.state = SimulationState(seed=seed)
            return self.seek(0)
        self.state = SimulationState(seed=seed)
        self.cell.reactor.reset(seed=seed)
        return self.step()

    def step(self) -> SimulationFrame:
        if self.pulse_active and self._pulse is not None:
            st = self._pulse.step()
            self.state.step_count += 1
            seed = self.state.seed + self.state.step_count
            rs = self.cell.step(seed=seed)
            self.state.elm_probs.append(max(rs.disruption.probability, st.disruption_risk))
            self.state.fusion_score = float(
                np.clip(0.5 * rs.kpis.score() + 0.5 * min(1.0, st.q_factor / 10.0), 0, 1)
            )
            return self._frame_from_pulse(st, seed=seed, reactor_step=rs)

        if self.scrub_active:
            assert self._scrub_result is not None
            nxt = self._scrub_index + 1
            if nxt >= len(self._scrub_result.frames):
                nxt = 0
            return self.seek(nxt)

        self.state.step_count += 1
        seed = self.state.seed + self.state.step_count
        live_step: ReactorStep = self.cell.step(seed=seed)

        self.state.elm_probs.append(live_step.disruption.probability)
        self.state.fusion_score = live_step.kpis.score()
        return self._frame_from_step(live_step, seed=seed, mode="live")

    def scrub_state(self) -> dict[str, Any]:
        if not self.scrub_active or self._scrub_result is None:
            return {"mode": "live", "shot_id": None, "index": None, "n_frames": 0}
        return {
            "mode": "scrub",
            "shot_id": self._scrub_result.shot_id,
            "device": self._scrub_result.device,
            "index": self._scrub_index,
            "n_frames": len(self._scrub_result.frames),
            "odl_label_rate": self._scrub_result.odl_label_rate,
            "mean_disruption": self._scrub_result.mean_disruption,
        }

    def _frame_from_step(
        self,
        rs: ReactorStep,
        *,
        seed: int,
        mode: str = "live",
        shot_id: str | None = None,
        scrub_index: int | None = None,
        scrub_n: int | None = None,
        time_s: float | None = None,
        odl_label: int | None = None,
        density: float | None = None,
    ) -> SimulationFrame:
        fuel = self.cell._fuel_cycle()
        muon = self.cell._muon_cycle()
        kpis = rs.kpis

        device = self.device
        targets = _PRESET_TARGETS.get(self.preset, _PRESET_TARGETS["H-mode"])
        fusion = self.state.fusion_score
        risk = rs.disruption.probability

        a = max(device.minor_radius_m, 1e-3)
        r0 = max(device.major_radius_m, 1e-3)
        kappa = device.elongation
        bt = device.max_bt_t

        ip_ma = device.max_ip_ma * targets["ip_frac"] * (0.85 + 0.15 * fusion)
        beta_n = device.troyon_beta_limit * targets["beta_n_frac"] * (0.8 + 0.2 * fusion)
        li = targets["li"] * (0.95 + 0.1 * (1 - risk))
        d_alpha = targets["d_alpha"] * (0.6 + 0.8 * risk)

        # q95 (edge safety factor): cylindrical approximation with an
        # elongation correction, q_cyl = 5 a^2 Bt / (R0 Ip[MA]),
        # q95 ~= q_cyl * (1 + kappa^2) / 2 (standard tokamak engineering
        # formula — see e.g. Wesson, "Tokamaks", ch. 3). Real geometry/field
        # dependence instead of an arbitrary constant.
        q_cyl = 5.0 * a**2 * bt / (r0 * max(ip_ma, 0.05))
        q95 = q_cyl * (1.0 + kappa**2) / 2.0

        # Greenwald density limit (Greenwald 1988): n_GW[1e20 m^-3] =
        # Ip[MA] / (pi a[m]^2). Line-averaged density is a preset-driven
        # fraction of this *physically derived* limit, so greenwald_fraction
        # is a genuine ratio rather than a scaled placeholder constant.
        n_gw_1e19 = 10.0 * max(ip_ma, 0.05) / (np.pi * a**2)
        if density is not None and density > 0:
            # ODL archives store density in m^-3; normalize to 1e19.
            dens = density / 1e19 if density > 1e18 else density
            ne_bar_1e19 = dens
            greenwald_fraction = ne_bar_1e19 / max(n_gw_1e19, 1e-9)
        else:
            ne_bar_1e19 = n_gw_1e19 * targets["dens_frac"] * (0.9 + 0.2 * fusion)
            greenwald_fraction = ne_bar_1e19 / max(n_gw_1e19, 1e-9)

        te0_kev = 10.0 + 15.0 * fusion
        p_nbi_mw = device.max_heating_mw * 0.5 * targets["beta_n_frac"]
        p_ech_mw = device.max_heating_mw * 0.15 * (0.5 + 0.5 * fusion)
        p_oh_mw = max(0.5, device.max_heating_mw * 0.05)
        p_input = p_oh_mw + p_nbi_mw + p_ech_mw

        # Plasma volume for an elongated torus V = 2 π² R₀ a² κ (also used for W_th).
        volume_m3 = 2.0 * np.pi**2 * r0 * a**2 * kappa

        # Radiated power from the same bremsstrahlung model the port view uses
        # (viz.radiance) — gauges and glow are one quantity, not two tuned worlds.
        from deepiri_fuselk.viz.radiance import bremsstrahlung_power_mw

        p_rad_mw = bremsstrahlung_power_mw(ne_bar_1e19, te0_kev, volume_m3)
        p_loss_mw = max(0.1, p_input - p_rad_mw * 0.5)

        # Stored thermal energy from the Troyon beta definition instead of an
        # arbitrary scaled constant: beta_N = beta_t[%] * a * Bt / Ip[MA]
        # (Troyon 1984) => beta_t = beta_N * Ip / (a * Bt) / 100; volume-avg
        # pressure p = beta_t * Bt^2 / (2 mu0); W_th = 1.5 * p * V.
        beta_t_pct = beta_n * ip_ma / max(a * bt, 1e-6)
        p_avg_pa = (beta_t_pct / 100.0) * bt**2 / (2.0 * _MU0)
        w_th_mj = 1.5 * p_avg_pa * volume_m3 / 1.0e6

        # tau_E = W_th / P_loss is the actual energy-confinement-time
        # definition (not a proxy).
        tau_e_s = w_th_mj / max(p_loss_mw, 1e-6)
        neutron_rate_1e18 = 0.5 + 4.5 * fusion**2
        q_plasma = (fusion**3) * 8.0 * (1 - risk)

        frame = SimulationFrame(
            step=self.state.step_count,
            seed=seed,
            raw_heat=rs.raw_heat,
            helix=rs.helix,
            elm=rs.disruption.elm,
            disruption=rs.disruption,
            controlled_heat=rs.heat_flux,
            action=rs.action_taken,
            fusion_score=self.state.fusion_score,
            tbr=fuel.tritium_breeding_ratio,
            muon_fpm=muon.fusions_per_muon,
            peclet=fuel.peclet_number,
            elm_free_fraction=kpis.elm_free_fraction,
            divertor_uniformity=kpis.divertor_uniformity,
            active_device=device,
            active_preset=self.preset,
            ip_ma=ip_ma,
            beta_n=beta_n,
            li=li,
            d_alpha=d_alpha,
            q95=q95,
            greenwald_fraction=greenwald_fraction,
            te0_kev=te0_kev,
            ne_bar_1e19=ne_bar_1e19,
            w_th_mj=w_th_mj,
            tau_e_s=tau_e_s,
            p_oh_mw=p_oh_mw,
            p_nbi_mw=p_nbi_mw,
            p_ech_mw=p_ech_mw,
            p_rad_mw=p_rad_mw,
            p_loss_mw=p_loss_mw,
            neutron_rate_1e18=neutron_rate_1e18,
            q_plasma=q_plasma,
            target_ip_ma=device.max_ip_ma * targets["ip_frac"],
            target_beta_n=device.troyon_beta_limit * targets["beta_n_frac"],
            target_li=targets["li"],
            target_d_alpha=targets["d_alpha"],
            mode=mode,
            shot_id=shot_id,
            scrub_index=scrub_index,
            scrub_n=scrub_n,
            time_s=time_s,
            odl_label=odl_label,
            density=density,
        )
        self._last = frame
        return frame

    def _frame_from_pulse(
        self,
        st: PulseState,
        *,
        seed: int,
        reactor_step: ReactorStep | None = None,
    ) -> SimulationFrame:
        """Map ReactorPulseEngine state onto a SimulationFrame for the control room."""
        if reactor_step is None:
            reactor_step = self.cell.step(seed=seed)
            self.state.step_count = max(self.state.step_count, 1)
            self.state.fusion_score = float(np.clip(min(1.0, st.q_factor / 10.0), 0, 1))

        fuel = self.cell._fuel_cycle()
        muon = self.cell._muon_cycle()
        # Scale heat maps by divertor / fusion intensity for immersive glow.
        scale = float(np.clip(0.3 + st.divertor_peak_mw_m2 / 20.0 + st.q_factor / 20.0, 0.2, 4.0))
        if st.phase == "disrupted":
            scale *= 2.5
        raw = reactor_step.raw_heat * scale
        controlled = reactor_step.heat_flux * scale

        # Override disruption probability with pulse risk when higher.
        disruption = reactor_step.disruption
        if st.disruption_risk > disruption.probability:
            from deepiri_fuselk.models.disruption_detector import DisruptionAssessment

            disruption = DisruptionAssessment(
                probability=st.disruption_risk,
                elm=disruption.elm,
                mhd_risk=max(disruption.mhd_risk, st.disruption_risk),
                helix_snr=disruption.helix_snr,
                recommended_action=(
                    "pellet_inject" if st.disruption_risk > 0.85 else disruption.recommended_action
                ),
                time_to_disruption_ms=max(5.0, (1.0 - st.disruption_risk) * 80.0),
            )

        frame = SimulationFrame(
            step=self.state.step_count,
            seed=seed,
            raw_heat=raw,
            helix=reactor_step.helix,
            elm=disruption.elm,
            disruption=disruption,
            controlled_heat=controlled,
            action=reactor_step.action_taken if st.alive else "pulse_hold",
            fusion_score=self.state.fusion_score,
            tbr=fuel.tritium_breeding_ratio,
            muon_fpm=muon.fusions_per_muon,
            peclet=fuel.peclet_number,
            elm_free_fraction=reactor_step.kpis.elm_free_fraction,
            divertor_uniformity=reactor_step.kpis.divertor_uniformity,
            active_device=self.device,
            active_preset=self.preset,
            ip_ma=st.ip_ma,
            beta_n=st.beta_n,
            li=0.9,
            d_alpha=0.4 + 0.5 * st.disruption_risk,
            q95=st.q95,
            greenwald_fraction=st.greenwald_fraction,
            te0_kev=st.te0_kev,
            ne_bar_1e19=st.ne_bar_1e19,
            w_th_mj=st.w_th_mj,
            tau_e_s=st.tau_e_s,
            p_oh_mw=st.p_ohm_mw,
            p_nbi_mw=st.p_aux_mw * 0.7,
            p_ech_mw=st.p_aux_mw * 0.3,
            p_rad_mw=st.p_rad_mw,
            p_loss_mw=st.p_loss_mw,
            neutron_rate_1e18=st.neutron_rate_1e18,
            q_plasma=st.q_factor,
            target_ip_ma=self.device.max_ip_ma * 0.85,
            target_beta_n=self.device.troyon_beta_limit * 0.75,
            target_li=0.85,
            target_d_alpha=0.3,
            mode="pulse",
            time_s=st.t_s,
            pulse_phase=st.phase,
            pulse_progress=st.progress,
            pulse_duration_s=st.duration_s,
            p_fusion_mw=st.p_fusion_mw,
            p_alpha_mw=st.p_alpha_mw,
            q_factor=st.q_factor,
            divertor_peak_mw_m2=st.divertor_peak_mw_m2,
            pulse_narrative=st.narrative,
            pulse_alive=st.alive,
        )
        self._last = frame
        return frame

    @property
    def last_frame(self) -> SimulationFrame | None:
        return self._last
