#!/usr/bin/env python3
"""Validate the four core fuselk rigor claims."""

from __future__ import annotations

import argparse
import json
import sys


def _json_safe(obj):
    """Convert numpy scalars to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def validate_pde() -> dict:
    from deepiri_fuselk.physics.pde_wellposedness import verify_wellposedness

    r = verify_wellposedness()
    return {
        "claim": "6-field PDE existence/uniqueness",
        "passed": r.passed,
        "contraction_constant": r.contraction_constant,
        "steady_uniqueness": r.steady_uniqueness,
        "local_existence": r.local_existence,
        "peclet": r.peclet_number,
        "summary": r.summary,
    }


def validate_muon() -> dict:
    from deepiri_fuselk.muon.literature_validation import validate_muon_trifecta

    r = validate_muon_trifecta()
    d = r.to_dict()
    d["claim"] = "muon trifecta literature validation"
    d["passed"] = r.all_benchmarks_pass and r.trifecta_pass
    return d


def validate_rl() -> dict:
    from deepiri_fuselk.control.convergence import verify_rl_convergence

    r = verify_rl_convergence()
    return {
        "claim": "RL convergence guarantees",
        "passed": r.passed,
        "bellman_residual": r.bellman_residual,
        "value_iteration_iterations": r.value_iteration_iterations,
        "optimal_discretized_return": r.optimal_discretized_return,
        "summary": r.summary,
    }


def validate_hqrm(iterations: int = 100) -> dict:
    from deepiri_fuselk.helix.jax_hqrm import benchmark_hqrm_latency

    r = benchmark_hqrm_latency(iterations=iterations)
    return {
        "claim": "JAX HQRM sub-ms GPU latency",
        "passed": r.passed,
        "jax_available": r.jax_available,
        "backend": r.backend,
        "mean_ms": r.mean_ms,
        "p99_ms": r.p99_ms,
        "sub_ms_claim_met": r.sub_ms_claim_met,
        "speedup_vs_numpy": r.speedup_vs_numpy,
        "summary": r.summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate fuselk rigor claims")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pde", action="store_true")
    parser.add_argument("--muon", action="store_true")
    parser.add_argument("--rl", action="store_true")
    parser.add_argument("--hqrm", action="store_true")
    parser.add_argument("--hqrm-iters", type=int, default=100)
    args = parser.parse_args()

    run_all = args.all or not any([args.pde, args.muon, args.rl, args.hqrm])
    results: dict = {}

    if run_all or args.pde:
        results["pde_wellposedness"] = validate_pde()
    if run_all or args.muon:
        results["muon_validation"] = validate_muon()
    if run_all or args.rl:
        results["rl_convergence"] = validate_rl()
    if run_all or args.hqrm:
        results["hqrm_latency"] = validate_hqrm(args.hqrm_iters)

    all_passed = all(v.get("passed", False) for v in results.values())
    results["_all_passed"] = all_passed

    print(json.dumps(_json_safe(results), indent=2))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
