#!/usr/bin/env python3
"""End-to-end fuselk benchmark — complements notebooks/ tutorials."""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


def bench_physics() -> dict:
    from deepiri_fuselk.physics import peclet_criterion, solve_oil_water_steady
    from deepiri_fuselk.physics.pde_solver import solve_oil_water_transient
    from deepiri_fuselk.physics.pde_system import PDEParameters

    t0 = time.perf_counter()
    steady = solve_oil_water_steady(n_grid=64, max_iter=500)
    transient = solve_oil_water_transient(n_grid=32, t_end=0.5, dt=0.01)
    elapsed = time.perf_counter() - t0
    return {
        "steady_converged": steady.converged,
        "steady_residual": steady.residual,
        "peclet": peclet_criterion(PDEParameters()),
        "transient_steps": len(transient),
        "elapsed_s": elapsed,
    }


def bench_helix() -> dict:
    from deepiri_fuselk.helix import HelixEngine
    from deepiri_fuselk.sim import generate_ece_shot

    t0 = time.perf_counter()
    engine = HelixEngine()
    snrs = []
    for i in range(20):
        shot = generate_ece_shot(32, seed=i)
        r = engine.process(shot.heat_field, shot.raw_signal, shot.angles)
        snrs.append(r.phase_locked_snr)
    elapsed = time.perf_counter() - t0
    return {"mean_snr": float(np.mean(snrs)), "frames": 20, "elapsed_s": elapsed}


def bench_muon() -> dict:
    from deepiri_fuselk.muon import RateNetworkParams, run_rate_network

    t0 = time.perf_counter()
    base = run_rate_network(params=RateNetworkParams(R_photon=0, R_proton=0))
    boosted = run_rate_network(params=RateNetworkParams(R_photon=0.5, R_proton=0.3))
    elapsed = time.perf_counter() - t0
    return {
        "baseline_fpm": base.fusions_per_muon,
        "boosted_fpm": boosted.fusions_per_muon,
        "breakeven": boosted.breakeven,
        "elapsed_s": elapsed,
    }


def bench_twin(steps: int = 50) -> dict:
    from deepiri_fuselk.sim import DigitalTwin

    t0 = time.perf_counter()
    twin = DigitalTwin(grid_size=16)
    twin.reset()
    for _ in range(steps):
        twin.step()
    summary = twin.summary()
    elapsed = time.perf_counter() - t0
    summary["elapsed_s"] = elapsed
    summary["steps"] = steps
    return summary


def bench_reactor(steps: int = 30) -> dict:
    from deepiri_fuselk.sim.reactor_cell import ReactorCell

    t0 = time.perf_counter()
    cell = ReactorCell(grid_size=16, train_elm=True)
    run = cell.run(n_steps=steps, seed=42)
    elapsed = time.perf_counter() - t0
    report = run.to_report()
    report["elapsed_s"] = elapsed
    report["fusion_score"] = run.final_score
    return report


def bench_elm(n_shots: int = 150) -> dict:
    from deepiri_fuselk.models.elm_predictor import ELMPredictor
    from deepiri_fuselk.sim.shot_corpus import generate_corpus

    t0 = time.perf_counter()
    corpus = generate_corpus(n_shots=n_shots, grid_size=16, seed=7)
    model = ELMPredictor()
    acc = model.train_from_corpus(corpus)
    elapsed = time.perf_counter() - t0
    return {
        "train_accuracy": acc,
        "corpus_size": n_shots,
        "elm_rate": corpus.elm_rate,
        "elapsed_s": elapsed,
    }


def bench_rl(timesteps: int = 5000) -> dict:
    from deepiri_fuselk.control.rl_agent import train_random_baseline, train_vent_policy

    t0 = time.perf_counter()
    random_r = train_random_baseline(episodes=5, steps=50)
    trained = train_vent_policy(timesteps=timesteps)
    elapsed = time.perf_counter() - t0
    return {
        "random_reward": random_r,
        "trained_reward": trained.mean_reward,
        "policy_saved": str(trained.policy_path) if trained.policy_path else None,
        "timesteps": trained.timesteps,
        "elapsed_s": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="fuselk benchmark suite")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--physics", action="store_true")
    parser.add_argument("--helix", action="store_true")
    parser.add_argument("--muon", action="store_true")
    parser.add_argument("--twin", action="store_true")
    parser.add_argument("--reactor", action="store_true")
    parser.add_argument("--elm", action="store_true")
    parser.add_argument("--rl", action="store_true")
    parser.add_argument("--rl-steps", type=int, default=5000)
    parser.add_argument("--twin-steps", type=int, default=50)
    parser.add_argument("--reactor-steps", type=int, default=30)
    args = parser.parse_args()

    run_all = args.all or not any(
        [args.physics, args.helix, args.muon, args.twin, args.reactor, args.elm, args.rl]
    )
    results = {}

    if run_all or args.physics:
        results["physics"] = bench_physics()
    if run_all or args.helix:
        results["helix"] = bench_helix()
    if run_all or args.muon:
        results["muon"] = bench_muon()
    if run_all or args.twin:
        results["twin"] = bench_twin(args.twin_steps)
    if run_all or args.reactor:
        results["reactor"] = bench_reactor(args.reactor_steps)
    if run_all or args.elm:
        results["elm"] = bench_elm()
    if run_all or args.rl:
        results["rl"] = bench_rl(args.rl_steps)

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
