"""Execute experiments from the YAML registry."""

from __future__ import annotations

from typing import Any

from deepiri_fuselk.experiments.registry import ExperimentEntry, get_experiment


def run_experiment(exp_id: str) -> dict[str, Any]:
    """Run a registered experiment by id and return JSON-serializable results."""
    entry = get_experiment(exp_id)
    if entry is None:
        raise ValueError(f"Unknown experiment: {exp_id}")

    runners: dict[str, Any] = {
        "elm_helix_denoising": _run_helix,
        "oil_water_barrier": _run_oil_water,
        "tritium_vapor_extraction": _run_tritium,
        "muon_photon_stripping": _run_muon_stripping,
        "brine_wall_coating": _run_brine,
        "vent_circularization_rl": _run_vent_rl,
        "plasma_traffic_routing": _run_traffic,
        "fusion_cell_full": _run_fusion_cell,
    }
    fn = runners.get(exp_id)
    if fn is None:
        return {"id": exp_id, "status": entry.status, "message": "no runner implemented"}
    return fn(entry)


def _run_helix(_: ExperimentEntry) -> dict:
    from deepiri_fuselk.helix.helix_engine import HelixEngine
    from deepiri_fuselk.sim.synthetic_data_gen import generate_ece_shot

    shot = generate_ece_shot(32, seed=0)
    r = HelixEngine().process(shot.heat_field, shot.raw_signal, shot.angles)
    return {
        "id": "elm_helix_denoising",
        "snr": r.phase_locked_snr,
        "elm_probability": r.elm_probability,
        "o_point": list(r.o_point),
    }


def _run_oil_water(_: ExperimentEntry) -> dict:
    from deepiri_fuselk.physics.pde_solver import solve_oil_water_steady

    r = solve_oil_water_steady(n_grid=32)
    return {"id": "oil_water_barrier", "converged": r.converged, "residual": r.residual}


def _run_tritium(_: ExperimentEntry) -> dict:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    from experiments.reactor.tritium_vapor_loop import run_tritium_extraction_sweep

    results = run_tritium_extraction_sweep(n_steps=5)
    ok = sum(1 for r in results if r.extraction_ok)
    return {"id": "tritium_vapor_extraction", "peclet_ok": ok, "samples": len(results)}


def _run_muon_stripping(_: ExperimentEntry) -> dict:
    from deepiri_fuselk.muon.stripping_orchestrator import run_stripping_trifecta

    r = run_stripping_trifecta()
    return {
        "id": "muon_photon_stripping",
        "fusions_per_muon": r.projected_fpm,
        "breakeven": r.breakeven,
        "R_total": r.R_total,
    }


def _run_brine(_: ExperimentEntry) -> dict:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    from experiments.lab.brine_wall import best_salinity, run_brine_sweep

    results = run_brine_sweep(n_steps=5)
    best = best_salinity(results)
    return {"id": "brine_wall_coating", "best_salinity_ppt": best, "viable_count": sum(1 for r in results if r.viable)}


def _run_vent_rl(_: ExperimentEntry) -> dict:
    from deepiri_fuselk.control.rl_agent import train_random_baseline

    reward = train_random_baseline(episodes=2, steps=20)
    return {"id": "vent_circularization_rl", "random_baseline_reward": reward}


def _run_traffic(_: ExperimentEntry) -> dict:
    import numpy as np

    from deepiri_fuselk.control.plasma_traffic_router import PlasmaTrafficRouter
    from deepiri_fuselk.viz.traffic_viewer import traffic_arrows

    heat = np.random.default_rng(0).random((16, 16)) * 5
    traffic = PlasmaTrafficRouter().route(heat)
    arrows = traffic_arrows(heat)
    return {
        "id": "plasma_traffic_routing",
        "variance": traffic.variance,
        "arrow_count": len(arrows),
    }


def _run_fusion_cell(_: ExperimentEntry) -> dict:
    from deepiri_fuselk.sim.fusion_cell import FusionCell

    _, report = FusionCell(grid_size=16, train_elm=False).run(n_steps=10, seed=0)
    return {"id": "fusion_cell_full", **report.to_dict()}
