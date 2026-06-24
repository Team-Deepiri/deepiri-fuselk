"""Tests for the fuselk desktop API backend."""

from deepiri_fuselk.viz.api import create_api, frame_to_dict
from deepiri_fuselk.viz.simulation_engine import LiveSimulation
from fastapi.testclient import TestClient


def test_api_health():
    client = TestClient(create_api())
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_sim_step_and_reset():
    client = TestClient(create_api())
    r = client.post("/api/sim/reset", json={"grid_size": 16, "seed": 0})
    assert r.status_code == 200
    data = r.json()
    assert data["step"] == 1
    r2 = client.post("/api/sim/step")
    assert r2.json()["step"] == 2


def test_api_experiments_list():
    client = TestClient(create_api())
    r = client.get("/api/experiments")
    assert r.status_code == 200
    items = r.json()
    assert len(items) >= 1
    assert "id" in items[0]


def test_api_physics_muon():
    client = TestClient(create_api())
    r = client.get("/api/physics/muon")
    assert r.status_code == 200
    assert "fusions_per_muon" in r.json()


def test_frame_to_dict_shape():
    frame = LiveSimulation(grid_size=16).reset(seed=1)
    d = frame_to_dict(frame)
    assert len(d["raw_heat"]) == 16
    assert "helix" in d
