"""Tests for shared fuel-cycle context."""

from deepiri_fuselk.sim.fuel_cycle_context import build_fuel_cycle_context
from deepiri_fuselk.sim.fusion_cell import FusionCell
from deepiri_fuselk.sim.reactor_cell import ReactorCell


def test_fusion_cell_shares_fuel_cycle_with_reactor():
    ctx = build_fuel_cycle_context(16)
    cell = FusionCell(grid_size=16, train_elm=False, fuel_cycle=ctx)
    assert cell.reactor._fuel_cycle.pde is ctx.pde
    assert cell._fuel_cycle_ctx.muon_trifecta is ctx.muon_trifecta


def test_reactor_builds_fuel_cycle_when_not_injected():
    cell = ReactorCell(grid_size=16, train_elm=False)
    assert cell._fuel_cycle.pde.converged
    assert cell._fuel_cycle.muon_fpm > 0
