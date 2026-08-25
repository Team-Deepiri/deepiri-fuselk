"""FastAPI backend for the fuselk desktop GUI."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from deepiri_fuselk import __version__
from deepiri_fuselk.experiments.registry import load_registry
from deepiri_fuselk.experiments.runner import run_experiment
from deepiri_fuselk.helix.disruption_filament import disruption_filament
from deepiri_fuselk.viz.radiance import radiance_from_frame
from deepiri_fuselk.viz.simulation_engine import (
    LiveSimulation,
    SimulationFrame,
    get_preset_names,
    list_device_names,
)

_STATIC = Path(__file__).resolve().parent / "static"
_STATIC_ROOT = _STATIC.resolve()
_ALLOWED_STATIC: dict[str, Path] = {}
for _rel in ("tokamak_viewer.html", "reactor_theatre.html", "branding/deepiri_favicon.svg"):
    _candidate = (_STATIC / _rel).resolve()
    if _candidate.is_relative_to(_STATIC_ROOT):
        _ALLOWED_STATIC[_rel] = _candidate
_sim = LiveSimulation(grid_size=24)


def _ndarray_to_list(arr: np.ndarray) -> list:
    return np.asarray(arr).tolist()


def frame_to_dict(frame: SimulationFrame) -> dict[str, Any]:
    fx, fy = frame.helix.fracture_vector
    filament = disruption_filament(
        frame.active_device,
        q95=frame.q95,
        fracture_vector=(float(fx), float(fy)),
    )
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
        "active_device": frame.active_device.name,
        "active_preset": frame.active_preset,
        "ip_ma": frame.ip_ma,
        "beta_n": frame.beta_n,
        "li": frame.li,
        "d_alpha": frame.d_alpha,
        "q95": frame.q95,
        "greenwald_fraction": frame.greenwald_fraction,
        "te0_kev": frame.te0_kev,
        "ne_bar_1e19": frame.ne_bar_1e19,
        "w_th_mj": frame.w_th_mj,
        "tau_e_s": frame.tau_e_s,
        "p_oh_mw": frame.p_oh_mw,
        "p_nbi_mw": frame.p_nbi_mw,
        "p_ech_mw": frame.p_ech_mw,
        "p_rad_mw": frame.p_rad_mw,
        "p_loss_mw": frame.p_loss_mw,
        "neutron_rate_1e18": frame.neutron_rate_1e18,
        "q_plasma": frame.q_plasma,
        "target_ip_ma": frame.target_ip_ma,
        "target_beta_n": frame.target_beta_n,
        "target_li": frame.target_li,
        "target_d_alpha": frame.target_d_alpha,
        "mode": frame.mode,
        "shot_id": frame.shot_id,
        "scrub_index": frame.scrub_index,
        "scrub_n": frame.scrub_n,
        "time_s": frame.time_s,
        "odl_label": frame.odl_label,
        "density": frame.density,
        "pulse_phase": frame.pulse_phase,
        "pulse_progress": frame.pulse_progress,
        "pulse_duration_s": frame.pulse_duration_s,
        "p_fusion_mw": frame.p_fusion_mw,
        "p_alpha_mw": frame.p_alpha_mw,
        "q_factor": frame.q_factor,
        "divertor_peak_mw_m2": frame.divertor_peak_mw_m2,
        "pulse_narrative": frame.pulse_narrative,
        "pulse_alive": frame.pulse_alive,
        "helix": {
            "o_point": list(frame.helix.o_point),
            "phase_locked_snr": frame.helix.phase_locked_snr,
            "elm_probability": frame.helix.elm_probability,
            "fracture_vector": list(frame.helix.fracture_vector),
            "focal_map": _ndarray_to_list(frame.helix.focal_map),
        },
        "raw_heat": _ndarray_to_list(frame.raw_heat),
        "controlled_heat": _ndarray_to_list(frame.controlled_heat),
        "radiance": radiance_from_frame(frame).to_dict(),
        "disruption_filament": filament.to_dict(),
        "device_shape": {
            "major_radius_m": frame.active_device.major_radius_m,
            "minor_radius_m": frame.active_device.minor_radius_m,
            "elongation": frame.active_device.elongation,
            "triangularity": frame.active_device.triangularity,
        },
    }


class SimConfig(BaseModel):
    grid_size: int = Field(default=24, ge=8, le=64)
    seed: int = 0
    device: str = "DEFAULT"
    preset: str = "H-mode"


class DeviceSelect(BaseModel):
    device: str


class PresetSelect(BaseModel):
    preset: str


class FusionRunRequest(BaseModel):
    steps: int = Field(default=50, ge=1, le=500)
    grid: int = Field(default=24, ge=8, le=64)


class OilWaterRequest(BaseModel):
    mode: str = "steady"
    n_grid: int = Field(default=32, ge=16, le=128)


class WorkbenchRequest(BaseModel):
    shot: str = "1140226012"
    n_steps: int = Field(default=24, ge=2, le=128)
    data_root: str | None = None
    ensure_data: bool = False


class AttachShotRequest(BaseModel):
    shot: str
    n_steps: int = Field(default=24, ge=2, le=128)
    data_root: str | None = None
    ensure_data: bool = False
    seed: int = 42


class SeekRequest(BaseModel):
    index: int = Field(ge=0)


class PerformancePdfRequest(BaseModel):
    kind: str = Field(default="workbench", description="workbench | odl | fusion")
    shot: str = "1140226012"
    n_steps: int = Field(default=16, ge=2, le=128)
    data_root: str | None = None
    ensure_data: bool = False
    max_shots: int = Field(default=8, ge=1, le=40)


class PulseStartRequest(BaseModel):
    device: str = "ITER"
    preset: str = "H-mode"
    dt_s: float = Field(default=2.0, ge=0.1, le=20.0)
    seed: int = 42


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
            "fpdf",
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
            _sim = LiveSimulation(grid_size=cfg.grid_size, device=cfg.device, preset=cfg.preset)
        else:
            _sim.clear_scrub()
            _sim.set_device(cfg.device)
            _sim.set_preset(cfg.preset)
        return frame_to_dict(_sim.reset(seed=cfg.seed))

    @api.get("/api/devices")
    def devices_list() -> dict[str, Any]:
        return {
            "devices": list_device_names(),
            "presets": get_preset_names("ITER"),
        }

    @api.post("/api/sim/device")
    def sim_set_device(req: DeviceSelect) -> dict[str, Any]:
        _sim.set_device(req.device)
        return frame_to_dict(_sim.step())

    @api.post("/api/sim/preset")
    def sim_set_preset(req: PresetSelect) -> dict[str, Any]:
        _sim.set_preset(req.preset)
        return frame_to_dict(_sim.step())

    @api.post("/api/sim/attach-shot")
    def sim_attach_shot(req: AttachShotRequest) -> dict[str, Any]:
        root = Path(req.data_root) if req.data_root else None
        try:
            frame = _sim.attach_shot(
                req.shot,
                n_steps=req.n_steps,
                seed=req.seed,
                data_root=root,
                ensure_data=req.ensure_data,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return frame_to_dict(frame)

    @api.post("/api/sim/seek")
    def sim_seek(req: SeekRequest) -> dict[str, Any]:
        try:
            return frame_to_dict(_sim.seek(req.index))
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/api/sim/detach-shot")
    def sim_detach_shot() -> dict[str, Any]:
        return frame_to_dict(_sim.detach_shot())

    @api.get("/api/sim/scrub")
    def sim_scrub_state() -> dict[str, Any]:
        return _sim.scrub_state()

    @api.post("/api/sim/pulse/start")
    def sim_pulse_start(req: PulseStartRequest) -> dict[str, Any]:
        return frame_to_dict(_sim.start_pulse(req.device, req.preset, dt_s=req.dt_s, seed=req.seed))

    @api.post("/api/sim/pulse/stop")
    def sim_pulse_stop() -> dict[str, Any]:
        return frame_to_dict(_sim.stop_pulse())

    @api.post("/api/sim/fusion-run")
    def sim_fusion_run(req: FusionRunRequest) -> dict[str, Any]:
        from deepiri_fuselk.sim.fusion_cell import FusionCell

        _, report = FusionCell(grid_size=req.grid, train_elm=False).run(n_steps=req.steps, seed=42)
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
        known = {e.id for e in load_registry()}
        if exp_id not in known:
            raise HTTPException(status_code=404, detail="unknown experiment")
        try:
            return run_experiment(exp_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="unknown experiment") from None

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

    @api.post("/api/workbench/analyze")
    def workbench_analyze(req: WorkbenchRequest) -> dict[str, Any]:
        from pathlib import Path

        from deepiri_fuselk.sim.shot_workbench import ShotWorkbench

        root = Path(req.data_root) if req.data_root else None
        try:
            report = ShotWorkbench(data_root=root).analyze(
                req.shot,
                n_steps=req.n_steps,
                ensure_data=req.ensure_data,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return report.to_dict()

    @api.post("/api/report/pdf")
    def report_pdf(req: PerformancePdfRequest) -> Response:
        """Auto-generate a performance PDF and return the bytes."""
        import tempfile

        from deepiri_fuselk.reports import (
            from_fusion_cell,
            from_odl_benchmark,
            from_workbench,
            render_performance_pdf,
        )

        kind = req.kind.lower().strip()
        root = Path(req.data_root) if req.data_root else None
        try:
            if kind == "workbench":
                from deepiri_fuselk.sim.shot_workbench import ShotWorkbench

                wb_report = ShotWorkbench(data_root=root).analyze(
                    req.shot,
                    n_steps=req.n_steps,
                    ensure_data=req.ensure_data,
                )
                perf = from_workbench(wb_report, version=__version__)
            elif kind == "odl":
                from deepiri_fuselk.sim.odl_benchmark import run_odl_benchmark

                perf = from_odl_benchmark(
                    run_odl_benchmark(
                        root,
                        max_shots=req.max_shots,
                        steps_per_shot=req.n_steps,
                        ensure_data=req.ensure_data,
                    ),
                    version=__version__,
                )
            elif kind == "fusion":
                from deepiri_fuselk.sim.fusion_cell import FusionCell

                _, cell_report = FusionCell(grid_size=16, train_elm=False).run(
                    n_steps=req.n_steps, seed=42
                )
                perf = from_fusion_cell(cell_report, version=__version__, steps=req.n_steps)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="kind must be workbench, odl, or fusion",
                )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        with tempfile.TemporaryDirectory() as tmp:
            path = render_performance_pdf(perf, Path(tmp) / "performance.pdf")
            data = path.read_bytes()
        return Response(
            content=data,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="fuselk_performance.pdf"'},
        )

    @api.get("/api/static/{filename}")
    def static_file(filename: str) -> FileResponse:
        key = filename.replace("\\", "/").lstrip("/")
        path = _ALLOWED_STATIC.get(key)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path)

    return api
