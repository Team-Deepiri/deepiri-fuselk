"""Tests for unified shot pipeline and running KPI accumulator."""

import numpy as np
from deepiri_fuselk.control.policy_runner import HybridPolicyRunner
from deepiri_fuselk.helix.helix_engine import HelixEngine
from deepiri_fuselk.models.disruption_detector import DisruptionDetector
from deepiri_fuselk.models.elm_predictor import ELMPredictor
from deepiri_fuselk.sim.fusion_kpis import RunningKPIAccumulator
from deepiri_fuselk.sim.shot_pipeline import ShotPipeline


def test_running_kpi_accumulator():
    acc = RunningKPIAccumulator()
    acc.update(snr=2.0, reward=-1.0, elm_probability=0.2)
    acc.update(snr=4.0, reward=1.0, elm_probability=0.8)
    assert acc.helix_snr_mean == 3.0
    assert acc.venturi_mean_reward == 0.0
    assert acc.elm_free_fraction == 0.5


def test_shot_pipeline_single_pass():
    pipeline = ShotPipeline(
        HelixEngine(),
        DisruptionDetector(ELMPredictor()),
        HybridPolicyRunner(),
    )
    result = pipeline.process(
        grid_size=16,
        seed=0,
        q_profile_values=np.array([1.0, 2.0, 3.0]),
        te_profile_values=np.array([2000.0, 2200.0]),
    )
    assert result.shot.heat_field.shape == (16, 16)
    assert result.helix.focal_map.shape == (16, 16)
    assert 0.0 <= result.disruption.probability <= 1.0
    assert result.control.final_heat.shape == (16, 16)
