"""Tests for Shot Workbench counterfactual scrub + export."""

from __future__ import annotations

from pathlib import Path

from deepiri_fuselk.data.fetchers import run_fetch
from deepiri_fuselk.sim.shot_workbench import ShotWorkbench, resolve_shot_path


def test_resolve_and_analyze_counterfactual(tmp_path: Path):
    run_fetch(tmp_path, ["synthetic", "odl"], n_shots=4, grid_size=16, max_odl_discharges=2)
    cmod = sorted((tmp_path / "shots").glob("CMOD_*.h5"))[0]
    discharge = cmod.stem.removeprefix("CMOD_")

    resolved = resolve_shot_path(discharge, data_root=tmp_path)
    assert resolved == cmod.resolve()

    wb = ShotWorkbench(grid_size=16, data_root=tmp_path)
    report = wb.analyze(discharge, n_steps=5)
    assert report.shot_id.startswith("CMOD_")
    assert report.n_frames == 5
    assert len(report.closed_loop_timeline) == 5
    assert len(report.open_loop_timeline) == 5
    assert all(p.action == "open_loop" for p in report.open_loop_timeline)
    assert report.radiance_p_rad_mw > 0.0
    assert "pitch_rad" in report.filament
    assert "counterfactual" in report.to_dict()

    md = report.to_markdown()
    assert "Shot Workbench" in md
    assert report.shot_id in md

    paths = wb.export(report, tmp_path / "out")
    assert paths["json"].is_file()
    assert paths["markdown"].is_file()
    assert paths["pdf"].is_file()
    assert paths["pdf"].read_bytes()[:4] == b"%PDF"


def test_analyze_batch_synthetic_fallback(tmp_path: Path):
    run_fetch(tmp_path, ["synthetic"], n_shots=3, grid_size=16)
    wb = ShotWorkbench(grid_size=16, data_root=tmp_path)
    reports = wb.analyze_batch(data_root=tmp_path, max_shots=2, n_steps=4, ensure_data=False)
    assert len(reports) == 2
    assert all(r.n_frames == 4 for r in reports)
