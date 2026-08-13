"""Stateful simulation engine for live dashboard visualization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from deepiri_fuselk.devices.profile import DeviceProfile
from deepiri_fuselk.devices.registry import DEFAULT as DEFAULT_DEVICE
from deepiri_fuselk.devices.registry import DeviceRegistry
from deepiri_fuselk.helix.helix_engine import HelixResult
from deepiri_fuselk.models.disruption_detector import DisruptionAssessment
from deepiri_fuselk.models.elm_predictor import ELMPrediction
from deepiri_fuselk.sim.fusion_cell import FusionCell, FusionCellReport
from deepiri_fuselk.sim.reactor_cell import ReactorStep

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


@dataclass
class SimulationState:
    step_count: int = 0
    seed: int = 42
    elm_probs: list[float] = field(default_factory=list)
    fusion_score: float = 0.0
    report: FusionCellReport | None = None


# Preset-shape targets used to synthesize plausible actual/target traces from
# the underlying FusionCell KPIs (fractions of device limits). Keyed by the
# discharge-preset name (H-mode / L-mode / Density Limit).
_PRESET_TARGETS: dict[str, dict[str, float]] = {
    "H-mode": {"ip_frac": 0.85, "beta_n_frac": 0.75, "li": 0.85, "d_alpha": 0.3},
    "L-mode": {"ip_frac": 0.55, "beta_n_frac": 0.4, "li": 1.05, "d_alpha": 0.6},
    "Density Limit": {"ip_frac": 0.7, "beta_n_frac": 0.55, "li": 0.95, "d_alpha": 0.9},
}


class LiveSimulation:
    """Step FusionCell/ReactorCell for real-time dashboard updates."""

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

    def set_device(self, name: str) -> None:
        """Switch the active device profile (drives gauge limits/equilibrium shape)."""
        self.device = get_device(name)

    def set_preset(self, name: str) -> None:
        """Switch the active discharge preset (drives target-vs-actual traces)."""
        if name in _PRESET_TARGETS:
            self.preset = name

    def reset(self, seed: int = 42) -> SimulationFrame:
        self.state = SimulationState(seed=seed)
        self.cell.reactor.reset(seed=seed)
        return self.step()

    def step(self) -> SimulationFrame:
        self.state.step_count += 1
        seed = self.state.seed + self.state.step_count
        rs: ReactorStep = self.cell.step(seed=seed)

        self.state.elm_probs.append(rs.disruption.probability)
        fuel = self.cell._fuel_cycle()
        muon = self.cell._muon_cycle()
        kpis = rs.kpis
        self.state.fusion_score = kpis.score()

        device = self.device
        targets = _PRESET_TARGETS.get(self.preset, _PRESET_TARGETS["H-mode"])
        fusion = self.state.fusion_score
        risk = rs.disruption.probability

        ip_ma = device.max_ip_ma * targets["ip_frac"] * (0.85 + 0.15 * fusion)
        beta_n = device.troyon_beta_limit * targets["beta_n_frac"] * (0.8 + 0.2 * fusion)
        li = targets["li"] * (0.95 + 0.1 * (1 - risk))
        d_alpha = targets["d_alpha"] * (0.6 + 0.8 * risk)
        q95 = 3.0 + 2.5 * (1 - targets["ip_frac"])
        greenwald_fraction = device.greenwald_limit * (0.5 + 0.4 * targets["beta_n_frac"])
        te0_kev = 10.0 + 15.0 * fusion
        ne_bar_1e19 = 4.0 + 6.0 * greenwald_fraction
        p_nbi_mw = device.max_heating_mw * 0.5 * targets["beta_n_frac"]
        p_ech_mw = device.max_heating_mw * 0.15 * (0.5 + 0.5 * fusion)
        p_oh_mw = max(0.5, device.max_heating_mw * 0.05)
        p_input = p_oh_mw + p_nbi_mw + p_ech_mw
        p_rad_mw = p_input * (0.2 + 0.3 * risk)
        p_loss_mw = max(0.1, p_input - p_rad_mw * 0.5)
        w_th_mj = 0.5 * beta_n * device.minor_radius_m * device.max_bt_t * 10.0
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
        )
        self._last = frame
        return frame

    @property
    def last_frame(self) -> SimulationFrame | None:
        return self._last
