"""Tests for reactor cell, fusion KPIs, ELM training, disruption detection."""

import numpy as np
import pytest
from deepiri_fuselk.models.disruption_detector import DisruptionDetector
from deepiri_fuselk.models.elm_predictor import ELMPredictor
from deepiri_fuselk.sim.fusion_kpis import FusionKPIs, divertor_uniformity, elm_free_fraction
from deepiri_fuselk.sim.reactor_cell import ReactorCell
from deepiri_fuselk.sim.shot_corpus import generate_corpus


def test_fusion_kpi_score():
    kpis = FusionKPIs(
        tritium_breeding_ratio=1.05,
        elm_free_fraction=0.9,
        divertor_uniformity=0.8,
        disruption_risk=0.1,
        muon_gain=300.0,
        helix_snr_mean=4.0,
        venturi_mean_reward=1.0,
        q_min=2.5,
        beta_n=2.0,
    )
    assert 0.5 < kpis.score() <= 1.0


def test_divertor_uniformity():
    uniform = np.ones((8, 8))
    hotspot = np.zeros((8, 8))
    hotspot[0, 0] = 10.0
    assert divertor_uniformity(uniform) == 1.0
    assert divertor_uniformity(hotspot) < 0.2


def test_elm_free_fraction():
    assert elm_free_fraction([0.1, 0.2, 0.8]) == 2 / 3


@pytest.mark.slow
def test_elm_train_from_corpus():
    corpus = generate_corpus(n_shots=60, grid_size=16, seed=1)
    model = ELMPredictor()
    acc = model.train_from_corpus(corpus)
    assert acc > 0.7
    assert model.train_accuracy == acc


def test_disruption_detector():
    from deepiri_fuselk.helix.helix_engine import HelixEngine
    from deepiri_fuselk.sim.synthetic_data_gen import generate_ece_shot

    shot = generate_ece_shot(16, seed=0, island_amplitude=0.9)
    helix = HelixEngine().process(shot.heat_field, shot.raw_signal, shot.angles)
    det = DisruptionDetector()
    assessment = det.assess(helix, q_min=1.8, beta_n=3.5)
    assert 0 <= assessment.probability <= 1
    assert assessment.recommended_action in {
        "hold",
        "strike_point_sweep",
        "gas_puff_radiate",
        "rmp_phase_shift",
        "pellet_inject",
    }


def test_reactor_cell_run():
    cell = ReactorCell(grid_size=16, train_elm=False)
    run = cell.run(n_steps=10, seed=0)
    assert len(run.steps) == 10
    assert 0 <= run.final_score <= 1
    report = run.to_report()
    assert report["steps"] == 10
    assert "tritium_breeding_ratio" in report
