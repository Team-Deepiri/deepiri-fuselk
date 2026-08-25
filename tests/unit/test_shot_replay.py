"""Tests for shot-faithful IMAS replay through ShotPipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from deepiri_fuselk.control.policy_runner import HybridPolicyRunner
from deepiri_fuselk.data.fetchers import run_fetch
from deepiri_fuselk.data.imas_loader import load_imas_hdf5, synthetic_imas_shot
from deepiri_fuselk.helix.helix_engine import HelixEngine
from deepiri_fuselk.models.disruption_detector import DisruptionDetector
from deepiri_fuselk.models.elm_predictor import ELMPredictor
from deepiri_fuselk.sim.reactor_cell import ReactorCell
from deepiri_fuselk.sim.shot_pipeline import ShotPipeline
from deepiri_fuselk.sim.shot_replay import ShotReplayer


def test_shot_pipeline_uses_imas_heat():
    imas = synthetic_imas_shot("SYN_TEST", size=16, seed=7)
    pipeline = ShotPipeline(
        HelixEngine(),
        DisruptionDetector(ELMPredictor()),
        HybridPolicyRunner(),
    )
    result = pipeline.process(
        grid_size=16,
        seed=0,
        q_profile_values=np.array([1.0, 2.0]),
        te_profile_values=np.array([2000.0]),
        imas=imas,
    )
    assert result.source == "imas"
    assert np.allclose(result.shot.heat_field, imas.heat_field, atol=1e-6)


def test_reactor_attach_shot_preserves_heat(tmp_path: Path):
    run_fetch(tmp_path, ["synthetic"], n_shots=4, grid_size=16)
    shot_path = next((tmp_path / "shots").glob("SYN*.h5"))
    shot = load_imas_hdf5(shot_path)

    cell = ReactorCell(grid_size=16, train_elm=False)
    cell.attach_shot(shot)
    step = cell.step(seed=1)
    assert step.raw_heat.shape == shot.heat_field.shape
    assert np.allclose(step.raw_heat, shot.heat_field, atol=1e-5)


def test_shot_replayer_scrub(tmp_path: Path):
    run_fetch(tmp_path, ["synthetic", "odl"], n_shots=4, grid_size=16, max_odl_discharges=2)
    cmod = sorted((tmp_path / "shots").glob("CMOD_*.h5"))[0]
    result = ShotReplayer(grid_size=16).scrub(path=cmod, n_steps=5)
    assert result.shot_id.startswith("CMOD_")
    assert len(result.frames) == 5
    assert 0.0 <= result.mean_disruption <= 1.0
    report = result.to_report()
    assert report["n_frames"] == 5
