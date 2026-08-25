"""Auto-generated performance PDF / Markdown reports."""

from __future__ import annotations

from pathlib import Path

from deepiri_fuselk import __version__
from deepiri_fuselk.data.fetchers import run_fetch
from deepiri_fuselk.reports import (
    from_fusion_cell,
    from_odl_benchmark,
    render_performance_pdf,
)
from deepiri_fuselk.sim.fusion_cell import FusionCell
from deepiri_fuselk.sim.odl_benchmark import run_odl_benchmark
from deepiri_fuselk.sim.shot_workbench import ShotWorkbench
from deepiri_fuselk.viz.api import create_api
from fastapi.testclient import TestClient


def test_render_fusion_performance_pdf(tmp_path: Path):
    _, cell_report = FusionCell(grid_size=16, train_elm=False).run(n_steps=4, seed=0)
    perf = from_fusion_cell(cell_report, version=__version__, steps=4)
    pdf = render_performance_pdf(perf, tmp_path / "fusion.pdf")
    assert pdf.is_file()
    assert pdf.stat().st_size > 500
    assert pdf.read_bytes()[:4] == b"%PDF"
    md = perf.to_markdown()
    assert "FusionCell" in md
    assert "Fusion score" in md


def test_workbench_export_includes_pdf(tmp_path: Path):
    run_fetch(tmp_path, ["synthetic"], n_shots=2, grid_size=16)
    shot = sorted((tmp_path / "shots").glob("SYN*.h5"))[0]
    wb = ShotWorkbench(grid_size=16, data_root=tmp_path)
    report = wb.analyze(shot, n_steps=4)
    paths = wb.export(report, tmp_path / "out")
    assert paths["json"].is_file()
    assert paths["markdown"].is_file()
    assert paths["pdf"].is_file()
    assert paths["pdf"].read_bytes()[:4] == b"%PDF"


def test_odl_benchmark_pdf(tmp_path: Path):
    run_fetch(tmp_path, ["synthetic", "odl"], n_shots=3, grid_size=16, max_odl_discharges=2)
    bench = run_odl_benchmark(tmp_path, max_shots=2, steps_per_shot=4, ensure_data=False)
    perf = from_odl_benchmark(bench, version=__version__)
    pdf = render_performance_pdf(perf, tmp_path / "odl.pdf")
    assert pdf.is_file()
    assert "ODL" in perf.title


def test_export_dossier_pack(tmp_path: Path):
    _, cell_report = FusionCell(grid_size=16, train_elm=False).run(n_steps=3, seed=1)
    from deepiri_fuselk.reports import export_dossier, from_fusion_cell

    perf = from_fusion_cell(cell_report, version=__version__, steps=3)
    paths = export_dossier(perf, tmp_path / "pack", stem="demo")
    assert paths["json"].name == "demo_performance.json"
    assert paths["markdown"].name == "demo_performance.md"
    assert paths["pdf"].name == "demo_performance.pdf"
    assert all(p.is_file() for p in paths.values())
    assert paths["pdf"].read_bytes()[:4] == b"%PDF"


def test_api_report_pdf_fusion():
    client = TestClient(create_api())
    r = client.post("/api/report/pdf", json={"kind": "fusion", "n_steps": 4})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/pdf")
    assert r.content[:4] == b"%PDF"
