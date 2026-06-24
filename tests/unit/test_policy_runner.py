"""Tests for hybrid policy runner."""

from pathlib import Path

import numpy as np

from deepiri_fuselk.control.policy_runner import HybridPolicyRunner


def test_hybrid_runner_venturi_only():
    runner = HybridPolicyRunner(policy_path=None)
    heat = np.random.default_rng(0).random((16, 16)).astype(np.float64)
    result = runner.step(heat, elm_probability=0.2)
    assert result.policy_active is False
    assert result.venturi.reward <= 0 or result.venturi.reward > -100


def test_hybrid_runner_train_and_attach(tmp_path):
    runner = HybridPolicyRunner()
    path = runner.train_and_attach(timesteps=512, save_path=tmp_path / "vent_ppo")
    assert runner.has_policy
    assert path.with_suffix(".zip").exists() or Path(str(path) + ".zip").exists()
