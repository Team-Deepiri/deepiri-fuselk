"""Tests for heat exhaust and phase separation experiments."""

from deepiri_fuselk.barrier.heat_exhaust import evaluate_brine_coating, evaluate_detachment
from deepiri_fuselk.barrier.phase_separator import oil_water_separated, simulate_phase_evolution
from deepiri_fuselk.physics.pde_system import PDEParameters


def test_brine_coating_evaluation():
    result = evaluate_brine_coating(salinity_ppt=35.0)
    assert result.thermal_resistance > 0
    assert 0 <= result.corrosion_risk <= 1


def test_detachment_regime():
    state = evaluate_detachment(gas_puff_rate=2.0)
    assert state.radiation_fraction > 0


def test_phase_separation():
    t, y = simulate_phase_evolution()
    assert len(t) > 0
    assert y.shape[0] == 2
    assert oil_water_separated(PDEParameters()) is True
