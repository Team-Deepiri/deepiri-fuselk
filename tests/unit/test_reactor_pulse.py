"""Immersive reactor pulse theatre — full discharge lifecycle."""

from __future__ import annotations

from deepiri_fuselk.sim.reactor_pulse import PulsePhase, ReactorPulseEngine
from deepiri_fuselk.viz.api import create_api
from deepiri_fuselk.viz.simulation_engine import LiveSimulation
from fastapi.testclient import TestClient


def test_reactor_pulse_lifecycle_diiid():
    engine = ReactorPulseEngine(device="DIII-D", preset="H-mode", dt_s=0.25, seed=1)
    frames = engine.run(max_steps=80)
    phases = {f.phase for f in frames}
    assert PulsePhase.BREAKDOWN.value in phases or PulsePhase.RAMP_UP.value in phases
    assert any(f.phase in (PulsePhase.FLAT_TOP.value, PulsePhase.RAMP_DOWN.value) for f in frames)
    assert frames[-1].phase in (PulsePhase.ENDED.value, PulsePhase.DISRUPTED.value)
    # Device-faithful: DIII-D Ip stays in physical ballpark during flat-top
    flat = [f for f in frames if f.phase == PulsePhase.FLAT_TOP.value]
    if flat:
        assert max(f.ip_ma for f in flat) <= 2.2
        assert max(f.q_factor for f in flat) >= 0.0


def test_iter_vs_diiid_scale_differently():
    iter_engine = ReactorPulseEngine(device="ITER", preset="H-mode", dt_s=20.0, seed=0)
    diiid_engine = ReactorPulseEngine(device="DIII-D", preset="H-mode", dt_s=0.5, seed=0)
    # Step into mid-pulse
    for _ in range(8):
        iter_engine.step()
        diiid_engine.step()
    assert iter_engine.state.ip_ma > diiid_engine.state.ip_ma
    assert iter_engine.state.duration_s > diiid_engine.state.duration_s


def test_live_simulation_pulse_mode():
    sim = LiveSimulation(grid_size=16, device="DIII-D", preset="H-mode")
    f0 = sim.start_pulse(device="DIII-D", preset="H-mode", dt_s=0.5, seed=2)
    assert f0.mode == "pulse"
    assert f0.pulse_phase is not None
    assert f0.pulse_narrative
    f1 = sim.step()
    assert f1.mode == "pulse"
    assert f1.time_s is not None and f1.time_s > 0
    assert f1.p_fusion_mw >= 0.0
    back = sim.stop_pulse()
    assert back.mode == "live"


def test_density_limit_can_disrupt():
    engine = ReactorPulseEngine(device="DIII-D", preset="Density Limit", dt_s=0.2, seed=7)
    frames = engine.run(max_steps=100)
    # Either disrupted or at least elevated risk during the pulse
    assert any(f.disruption_risk > 0.5 for f in frames) or any(
        f.phase == PulsePhase.DISRUPTED.value for f in frames
    )


def test_api_pulse_start_step():
    client = TestClient(create_api())
    r = client.post(
        "/api/sim/pulse/start",
        json={"device": "DIII-D", "preset": "H-mode", "dt_s": 0.5, "seed": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "pulse"
    assert body["pulse_phase"]
    assert "p_fusion_mw" in body
    r2 = client.post("/api/sim/step")
    assert r2.json()["mode"] == "pulse"
    assert r2.json()["time_s"] >= body.get("time_s", 0)
    theatre = client.get("/api/static/reactor_theatre.html")
    assert theatre.status_code == 200
    assert b"Reactor Theatre" in theatre.content
