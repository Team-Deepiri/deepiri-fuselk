"""Tests for breeding blanket."""

from deepiri_fuselk.barrier.breeding_blanket import evaluate_breeding_blanket
from deepiri_fuselk.barrier.vapor_dynamics import barrier_stability, find_interface
from deepiri_fuselk.physics.pde_solver import solve_oil_water_steady
from deepiri_fuselk.physics.pde_system import PDEParameters


def test_find_interface_midplane():
    result = solve_oil_water_steady(n_grid=32, max_iter=200)
    x_i = find_interface(result.state)
    assert 0.0 <= x_i <= 1.0


def test_breeding_evaluation():
    result = solve_oil_water_steady(n_grid=32, max_iter=200)
    br = evaluate_breeding_blanket(result.state, PDEParameters())
    assert br.tritium_breeding_ratio > 0
    assert 0 <= br.extraction_efficiency <= 1.0


def test_barrier_stability():
    assert barrier_stability(PDEParameters()) is True
