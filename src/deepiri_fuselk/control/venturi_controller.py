"""Venturi hierarchical hybrid controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepiri_fuselk.control.plasma_traffic_router import PlasmaTrafficRouter, TrafficState
from deepiri_fuselk.control.watchdog import SafetyWatchdog, WatchdogAction
from deepiri_fuselk.models.bayesian_prior import BayesianRotationalPrior, PriorState


@dataclass
class VenturiAction:
    sweep_velocity: float
    rmp_phase: float
    gas_puff: float
    pellet_ready: bool
    overridden: bool


@dataclass
class VenturiState:
    traffic: TrafficState
    prior: PriorState
    action: VenturiAction
    reward: float


class VenturiController:
    """
    Hierarchical Venturi controller.

    Top level (slow): Bayesian rotational prior sets action bounds.
    Bottom level (fast): Traffic-aware policy circularizes divertor exhaust.
    Watchdog: Safety PID override on engineering limit breach.
    """

    def __init__(self, engineering_limit: float = 10.0) -> None:
        self.prior_model = BayesianRotationalPrior()
        self.router = PlasmaTrafficRouter(engineering_limit=engineering_limit)
        self.watchdog = SafetyWatchdog(engineering_limit=engineering_limit)
        self._phase = 0.0

    def slow_loop(
        self,
        ne_pedestal: float = 0.8,
        beta_n: float = 2.5,
        rotation_khz: float = 5.0,
        q95: float = 3.5,
    ) -> PriorState:
        return self.prior_model.update(ne_pedestal, beta_n, rotation_khz, q95)

    def fast_loop(
        self,
        heat_flux: np.ndarray,
        prior: PriorState,
        elm_probability: float = 0.0,
    ) -> VenturiAction:
        traffic = self.router.route(heat_flux)
        congestion = self.router.congestion_ratio(traffic)

        # Circularize: increase sweep when variance is high
        sweep = min(prior.max_sweep_velocity, 0.3 + 0.5 * traffic.variance / 5.0)
        rmp = min(prior.max_rmp_phase, 0.2 + 0.4 * congestion)

        # ELM preempt: gas puff when precursor detected
        gas_puff = 0.5 if elm_probability > 0.7 else 0.0
        pellet_ready = elm_probability > 0.85

        proposed = np.array([sweep, rmp])
        wd: WatchdogAction = self.watchdog.check(traffic.peak_flux, proposed)

        self._phase += wd.safe_sweep_velocity * 0.1
        return VenturiAction(
            sweep_velocity=wd.safe_sweep_velocity,
            rmp_phase=wd.safe_rmp_phase,
            gas_puff=gas_puff,
            pellet_ready=pellet_ready,
            overridden=wd.override,
        )

    def step(
        self,
        heat_flux: np.ndarray,
        ne_pedestal: float = 0.8,
        beta_n: float = 2.5,
        rotation_khz: float = 5.0,
        q95: float = 3.5,
        elm_probability: float = 0.0,
    ) -> VenturiState:
        prior = self.slow_loop(ne_pedestal, beta_n, rotation_khz, q95)
        action = self.fast_loop(heat_flux, prior, elm_probability)
        traffic = self.router.route(heat_flux)
        reward = -traffic.variance - 0.1 * traffic.peak_flux
        if traffic.variance < 0.5:
            reward += 5.0
        if action.overridden:
            reward -= 10.0
        return VenturiState(traffic=traffic, prior=prior, action=action, reward=reward)
