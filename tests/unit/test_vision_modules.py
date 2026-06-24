"""Tests for HELIX engine, Venturi controller, models, and data layer."""

import numpy as np
from deepiri_fuselk.control.venturi_controller import VenturiController
from deepiri_fuselk.data.imas_loader import synthetic_imas_shot
from deepiri_fuselk.helix.helix_engine import HelixEngine
from deepiri_fuselk.models.bayesian_prior import BayesianRotationalPrior
from deepiri_fuselk.models.elm_predictor import ELMPredictor
from deepiri_fuselk.muon.tritium_capsule import TritiumCapsule, evaluate_capsule
from deepiri_fuselk.sim.synthetic_data_gen import generate_ece_shot


def test_helix_engine_pipeline():
    shot = generate_ece_shot(32, seed=0)
    engine = HelixEngine()
    result = engine.process(shot.heat_field, shot.raw_signal, shot.angles)
    assert result.focal_map.shape == (32, 32)
    assert 0 <= result.elm_probability <= 1
    assert result.phase_locked_snr > 0


def test_venturi_controller_step():
    ctrl = VenturiController()
    heat = np.random.default_rng(0).random((16, 16)) * 5
    state = ctrl.step(heat, elm_probability=0.3)
    assert state.action.sweep_velocity >= 0
    assert isinstance(state.reward, float)


def test_bayesian_prior_regimes():
    prior = BayesianRotationalPrior()
    low = prior.update(0.3, 1.0, 2.0, 2.0)
    high = prior.update(1.5, 4.0, 15.0, 5.0)
    assert low.max_sweep_velocity >= high.max_sweep_velocity


def test_elm_predictor():
    pred = ELMPredictor()
    focal = np.random.default_rng(1).random((16, 16))
    result = pred.predict(focal)
    assert 0 <= result.probability <= 1
    assert result.confidence_interval[0] <= result.confidence_interval[1]


def test_imas_synthetic_shot():
    shot = synthetic_imas_shot("TEST001", size=16)
    assert shot.shot_id == "TEST001"
    assert shot.heat_field.shape == (16, 16)
    assert len(shot.q_profile.radius) == 16


def test_tritium_capsule():
    cap = TritiumCapsule()
    perf = evaluate_capsule(cap)
    assert perf.breeding_ratio > 0
    assert isinstance(perf.self_sustaining, bool)
