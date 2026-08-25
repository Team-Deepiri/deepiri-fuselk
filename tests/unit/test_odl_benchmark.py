"""ODL public benchmark tests (uses fixture fetch into tmp_path)."""

from __future__ import annotations

from pathlib import Path

from deepiri_fuselk.data.fetchers import run_fetch
from deepiri_fuselk.sim.odl_benchmark import run_odl_benchmark


def test_odl_benchmark_on_fetched_subset(tmp_path: Path):
    run_fetch(tmp_path, ["synthetic", "odl"], n_shots=4, grid_size=16, max_odl_discharges=3)
    report = run_odl_benchmark(
        tmp_path,
        max_shots=3,
        steps_per_shot=4,
        ensure_data=False,
    )
    assert report.n_shots == 3
    assert 0.0 <= report.mean_auc_proxy <= 1.0
    assert len(report.shots) == 3
    payload = report.to_dict()
    assert payload["n_shots"] == 3
