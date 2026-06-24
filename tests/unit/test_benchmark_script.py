"""Smoke test for benchmark script."""

import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_physics_only():
    script = Path(__file__).resolve().parents[2] / "scripts" / "benchmark.py"
    out = subprocess.check_output(
        [sys.executable, str(script), "--physics"],
        text=True,
    )
    data = json.loads(out)
    assert "physics" in data
    assert "steady_converged" in data["physics"]
