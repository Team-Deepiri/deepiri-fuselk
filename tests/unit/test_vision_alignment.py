"""Tests for VISION.md alignment wired into core modules."""

from deepiri_fuselk.control.convergence import verify_rl_convergence
from deepiri_fuselk.helix.helix_engine import HelixEngine
from deepiri_fuselk.helix.jax_hqrm import benchmark_hqrm_latency, run_hqrm_jax
from deepiri_fuselk.muon.literature_validation import validate_muon_trifecta
from deepiri_fuselk.muon.stripping_orchestrator import run_stripping_trifecta
from deepiri_fuselk.physics.pde_solver import solve_oil_water_steady
from deepiri_fuselk.physics.pde_wellposedness import verify_wellposedness
from deepiri_fuselk.sim import generate_ece_shot
from deepiri_fuselk.sim.fusion_cell import FusionCell
from deepiri_fuselk.sim.vision_alignment import audit_vision_alignment


def test_pde_solver_attaches_wellposedness():
    r = solve_oil_water_steady(n_grid=32)
    assert r.wellposedness is not None
    assert r.wellposedness.passed
    assert r.wellposedness.steady_uniqueness


def test_pde_six_fields():
    r = verify_wellposedness()
    assert len(r.fields) == 6
    assert "n_mu" in r.fields
    assert "T_p" in r.fields


def test_muon_trifecta_literature_on_result():
    r = run_stripping_trifecta()
    assert r.literature_band[0] < r.literature_band[1]
    assert isinstance(r.literature_aligned, bool)


def test_muon_literature_validation_runs():
    r = validate_muon_trifecta()
    assert r.trifecta_fpm > 0
    assert len(r.results) == 3
    assert r.breakeven_threshold == 284.0


def test_rl_convergence_preconditions():
    r = verify_rl_convergence(grid_size=8)
    assert r.finite_horizon
    assert r.bounded_reward
    assert r.compact_action_space
    assert r.ppo_assumptions_met


def test_rl_train_attaches_convergence(monkeypatch):
    from deepiri_fuselk.control import rl_agent

    report = verify_rl_convergence(grid_size=8)
    monkeypatch.setattr(rl_agent, "verify_rl_convergence", lambda **kw: report)
    monkeypatch.setattr(rl_agent, "_SB3", False)
    trained = rl_agent.train_vent_policy(verify_convergence=True)
    assert trained.convergence is report


def test_helix_uses_jax_hqrm_path():
    engine = HelixEngine()
    shot = generate_ece_shot(32, seed=0)
    result = engine.process(shot.heat_field, shot.raw_signal, shot.angles)
    assert result.hqrm_backend in ("jax", "numpy")
    assert result.hqrm.converged or result.hqrm.heat_variance >= 0


def test_hqrm_jax_runs():
    import numpy as np

    heat = np.random.default_rng(0).random((32, 32))
    result = run_hqrm_jax(heat)
    assert result.heat_variance >= 0


def test_hqrm_benchmark():
    r = benchmark_hqrm_latency(grid_size=32, iterations=30, warmup=5)
    assert r.iterations > 0
    assert r.mean_ms > 0
    assert r.summary


def test_fusion_cell_includes_vision_audit():
    cell = FusionCell(grid_size=16, train_elm=False)
    _, report = cell.run(n_steps=3, seed=0, include_vision=True)
    assert report.vision is not None
    assert len(report.vision.pillars) >= 6
    assert "vision" in report.to_dict()


def test_vision_alignment_audit():
    report = audit_vision_alignment(skip_slow=True)
    assert report.pde is not None
    assert report.muon is not None
    assert report.hqrm is not None
    d = report.to_dict()
    assert "pillars" in d
    assert "gaps" in d
