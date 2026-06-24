"""Tests for live simulation dashboard and fusion cell."""

from deepiri_fuselk.experiments.runner import run_experiment
from deepiri_fuselk.sim.fusion_cell import FusionCell
from deepiri_fuselk.viz.dashboard.app import create_app
from deepiri_fuselk.viz.dashboard.figures import build_control_room_figure, build_kpi_strip
from deepiri_fuselk.viz.simulation_engine import LiveSimulation


def test_live_simulation_steps():
    sim = LiveSimulation(grid_size=16)
    f1 = sim.reset(seed=0)
    f2 = sim.step()
    assert f1.step == 1
    assert f2.step == 2
    assert f2.raw_heat.shape == (16, 16)
    assert 0 <= f2.fusion_score <= 1


def test_dashboard_figures():
    frame = LiveSimulation(grid_size=16).reset()
    fig = build_control_room_figure(frame)
    kpi = build_kpi_strip(frame)
    assert len(fig.data) >= 5
    assert len(kpi.data) == 5


def test_dashboard_app_creates():
    app = create_app()
    assert app.layout is not None


def test_fusion_cell_report():
    _, report = FusionCell(grid_size=16, train_elm=False).run(n_steps=5, seed=0)
    d = report.to_dict()
    assert "fusion_score" in d
    assert "fuel_cycle" in d
    assert "muon_cycle" in d
    assert d["vision"] is None
    assert d["muon_cycle"]["literature_aligned"] is not None


def test_experiment_runner_traffic():
    r = run_experiment("plasma_traffic_routing")
    assert r["id"] == "plasma_traffic_routing"
    assert "arrow_count" in r


def test_experiment_runner_muon():
    r = run_experiment("muon_photon_stripping")
    assert "fusions_per_muon" in r
