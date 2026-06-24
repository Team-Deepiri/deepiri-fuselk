"""Tests for control layer."""

import numpy as np
from deepiri_fuselk.control.plasma_traffic_router import PlasmaTrafficRouter
from deepiri_fuselk.control.realtime_bus import RealtimeBus
from deepiri_fuselk.control.rl_agent import train_random_baseline
from deepiri_fuselk.control.vent_circularizer_env import VentCircularizerEnv
from deepiri_fuselk.control.watchdog import SafetyWatchdog


def test_traffic_router_variance():
    router = PlasmaTrafficRouter()
    heat = np.ones((8, 8))
    heat[4, 4] = 10.0
    state = router.route(heat)
    assert state.peak_flux == 10.0
    assert state.variance > 0


def test_vent_env_reset_step():
    env = VentCircularizerEnv(grid_size=8, max_steps=10)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (8, 8)
    obs, reward, done, _, info = env.step(np.array([0.5, 0.5]))
    assert "variance" in info
    assert isinstance(reward, float)


def test_watchdog_triggers_on_high_flux():
    wd = SafetyWatchdog(limit_fraction=0.5, engineering_limit=10.0)
    action = wd.check(peak_flux=9.0, proposed_action=np.array([0.1, 0.1]))
    assert action.override is True


def test_realtime_bus_publish():
    bus = RealtimeBus()
    frames = bus.subscribe("diagnostics")
    bus.publish("diagnostics/ece", np.array([1.0, 2.0]))
    assert len(frames) == 1


def test_rl_baseline_runs():
    mean_reward = train_random_baseline(episodes=2, steps=10)
    assert isinstance(mean_reward, float)
