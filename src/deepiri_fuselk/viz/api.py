"""FastAPI backend for the fuselk desktop GUI."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from deepiri_fuselk import __version__
from deepiri_fuselk.experiments.registry import load_registry
from deepiri_fuselk.experiments.runner import run_experiment
from deepiri_fuselk.viz.simulation_engine import LiveSimulation, SimulationFrame

_STATIC = Path(__file__).resolve().parent / "static"
_sim = LiveSimulation(grid_size=24)


def _ndarray_to_list(arr: np.ndarray) -> list:
    return np.asarray(arr).tolist()


def frame_to_dict(frame: SimulationFrame) -> dict[str, Any]:
    return {
        "step": frame.step,
        "seed": frame.seed,
        "action": frame.action,
        "fusion_score": frame.fusion_score,
        "tbr": frame.tbr,
        "muon_fpm": frame.muon_fpm,
        "peclet": frame.peclet,
        "elm_free_fraction": frame.elm_free_fraction,
        "divertor_uniformity": frame.divertor_uniformity,
        "disruption_probability": frame.disruption.probability,
        "elm_probability": frame.elm.probability,
        "helix": {
            "o_point": list(frame.helix.o_point),
            "phase_locked_snr": frame.helix.phase_locked_snr,
            "elm_probability": frame.helix.elm_probability,
            "fracture_vector": list(frame.helix.fracture_vector),
            "focal_map": _ndarray_to_list(frame.helix.focal_map),
        },
        "raw_heat": _ndarray_to_list(frame.raw_heat),
        "controlled_heat": _ndarray_to_list(frame.controlled_heat),
    }


class SimConfig(BaseModel):
    grid_size: int = Field(default=24, ge=8, le=64)
    seed: int = 0


class FusionRunRequest(BaseModel):
    steps: int = Field(default=50, ge=1, le=500)
    grid: int = Field(default=24, ge=8, le=64)


class OilWaterRequest(BaseModel):
    mode: str = "steady"
    n_grid: int = Field(default=32, ge=16, le=128)


def create_api() -> FastAPI:
    api = FastAPI(title="deepiri-fuselk API", version=__version__)
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @api.get("/api/doctor")
    def doctor() -> dict[str, Any]:
        modules = [
            "numpy",
            "scipy",
            "xarray",
            "pydantic",
            "zmq",
            "pyarrow",
            "gymnasium",
            "stable_baselines3",
            "dash",
            "plotly",
        ]
        results: list[dict[str, str]] = []
        ok = True
        for name in modules:
            try:
                importlib.import_module(name)
                results.append({"module": name, "status": "ok"})
            except ImportError:
                ok = False
                results.append({"module": name, "status": "missing"})

        from deepiri_fuselk.sim.vision_alignment import audit_vision_alignment

        vision = audit_vision_alignment(skip_slow=True).to_dict()
        if vision.get("gaps"):
            ok = False
        return {"ok": ok, "modules": results, "vision": vision}

    @api.get("/api/sim/frame")
    def sim_frame() -> dict[str, Any]:
        frame = _sim.last_frame
        if frame is None:
            frame = _sim.reset(seed=0)
        return frame_to_dict(frame)

    @api.post("/api/sim/step")
    def sim_step() -> dict[str, Any]:
        return frame_to_dict(_sim.step())

    @api.post("/api/sim/reset")
    def sim_reset(config: SimConfig | None = None) -> dict[str, Any]:
        cfg = config or SimConfig()
        global _sim
        if cfg.grid_size != _sim.grid_size:
            _sim = LiveSimulation(grid_size=cfg.grid_size)
        return frame_to_dict(_sim.reset(seed=cfg.seed))

    @api.post("/api/sim/fusion-run")
    def sim_fusion_run(req: FusionRunRequest) -> dict[str, Any]:
        from deepiri_fuselk.sim.fusion_cell import FusionCell

        _, report = FusionCell(grid_size=req.grid, train_elm=False).run(
            n_steps=req.steps, seed=42
        )
        return report.to_dict()

    @api.get("/api/experiments")
    def experiments_list() -> list[dict[str, str]]:
        return [
            {
                "id": e.id,
                "name": e.name,
                "status": e.status,
                "category": e.category,
                "description": e.description,
            }
            for e in load_registry()
        ]

    @api.post("/api/experiments/{exp_id}/run")
    def experiments_run(exp_id: str) -> dict[str, Any]:
        try:
            return run_experiment(exp_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @api.post("/api/physics/oil-water")
    def physics_oil_water(req: OilWaterRequest) -> dict[str, Any]:
        from deepiri_fuselk.physics.pde_solver import (
            solve_oil_water_steady,
            solve_oil_water_transient,
        )

        out: dict[str, Any] = {"mode": req.mode}
        if req.mode in ("steady", "both"):
            r = solve_oil_water_steady(n_grid=req.n_grid)
            out["steady"] = {
                "converged": r.converged,
                "residual": r.residual,
                "iterations": r.iterations,
            }
        if req.mode in ("transient", "both"):
            hist = solve_oil_water_transient(n_grid=min(req.n_grid, 64), t_end=1.0)
            out["transient"] = {
                "steps": len(hist),
                "final_n_T_wall": float(hist[-1].n_T[-1]),
            }
        return out

    @api.get("/api/physics/muon")
    def physics_muon() -> dict[str, Any]:
        from deepiri_fuselk.muon import RateNetworkParams, run_rate_network

        r = run_rate_network(params=RateNetworkParams(R_photon=0.5, R_proton=0.3))
        return {
            "fusions_per_muon": r.fusions_per_muon,
            "effective_sticking": r.effective_sticking,
            "breakeven": r.breakeven,
        }

    @api.get("/api/static/{filename}")
    def static_file(filename: str) -> FileResponse:
        static_root = _STATIC.resolve()
        path = (static_root / filename).resolve()
        try:
            path.relative_to(static_root)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="not found") from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path)

    return api
