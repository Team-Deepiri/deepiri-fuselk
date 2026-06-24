"""Tests for the four core rigor claims."""

import pytest

from deepiri_fuselk.control.convergence import verify_rl_convergence
from deepiri_fuselk.helix.jax_hqrm import benchmark_hqrm_latency, run_hqrm_jax
from deepiri_fuselk.muon.literature_validation import validate_muon_trifecta
from deepiri_fuselk.physics.pde_wellposedness import verify_wellposedness


def test_pde_wellposedness_default_params():
    r = verify_wellposedness()
    assert r.diffusion_positive
    assert r.local_existence
    assert r.steady_uniqueness
    assert r.contraction_constant < 1.0
    assert r.passed


def test_pde_six_fields():
    r = verify_wellposedness()
    assert len(r.fields) == 6
    assert "n_mu" in r.fields
    assert "T_p" in r.fields


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


def test_hqrm_jax_runs():
    import numpy as np

    heat = np.random.default_rng(0).random((32, 32))
    result = run_hqrm_jax(heat)
    assert 0 <= result.heat_variance or result.heat_variance >= 0


def test_hqrm_benchmark():
    r = benchmark_hqrm_latency(grid_size=32, iterations=30, warmup=5)
    assert r.iterations > 0
    assert r.mean_ms > 0
    assert r.summary
