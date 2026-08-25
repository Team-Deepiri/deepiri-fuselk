"""LiveSimulation shot-scrub mode — port view shares Workbench clock."""

from __future__ import annotations

from pathlib import Path

from deepiri_fuselk.data.fetchers import run_fetch
from deepiri_fuselk.viz.api import create_api
from deepiri_fuselk.viz.simulation_engine import LiveSimulation
from fastapi.testclient import TestClient


def test_live_simulation_attach_seek_detach(tmp_path: Path):
    run_fetch(tmp_path, ["synthetic"], n_shots=3, grid_size=16)
    shot = sorted((tmp_path / "shots").glob("SYN*.h5"))[0]

    sim = LiveSimulation(grid_size=16)
    live = sim.reset(seed=0)
    assert live.mode == "live"

    f0 = sim.attach_shot(shot, n_steps=5, seed=1)
    assert f0.mode == "scrub"
    assert f0.shot_id is not None
    assert f0.scrub_index == 0
    assert f0.scrub_n == 5
    assert sim.scrub_state()["mode"] == "scrub"

    f2 = sim.seek(2)
    assert f2.scrub_index == 2
    assert f2.raw_heat.shape[0] >= 8

    f3 = sim.step()
    assert f3.scrub_index == 3

    # Loop at end
    sim.seek(4)
    wrapped = sim.step()
    assert wrapped.scrub_index == 0

    back = sim.detach_shot()
    assert back.mode == "live"
    assert sim.scrub_state()["mode"] == "live"


def test_api_attach_seek_scrub(tmp_path: Path):
    run_fetch(tmp_path, ["synthetic"], n_shots=2, grid_size=16)
    shot = sorted((tmp_path / "shots").glob("SYN*.h5"))[0]
    client = TestClient(create_api())

    r = client.post(
        "/api/sim/attach-shot",
        json={"shot": str(shot), "n_steps": 4, "data_root": str(tmp_path)},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "scrub"
    assert body["scrub_n"] == 4
    assert "shot_id" in body

    r2 = client.post("/api/sim/seek", json={"index": 2})
    assert r2.status_code == 200
    assert r2.json()["scrub_index"] == 2

    state = client.get("/api/sim/scrub").json()
    assert state["mode"] == "scrub"
    assert state["n_frames"] == 4

    det = client.post("/api/sim/detach-shot")
    assert det.status_code == 200
    assert det.json()["mode"] == "live"
