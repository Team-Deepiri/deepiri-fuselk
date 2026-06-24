"""fuselk CLI — simulation, training, benchmarks, visualization."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from deepiri_fuselk import __version__

app = typer.Typer(name="fuselk", help="Fusion Unified Simulation, ELM Learning & Kinetics", no_args_is_help=True)
console = Console()

sim_app = typer.Typer(help="Run simulations")
train_app = typer.Typer(help="Train RL policies")
data_app = typer.Typer(help="Data import/export")
viz_app = typer.Typer(help="Visualization")
app.add_typer(sim_app, name="sim")
app.add_typer(train_app, name="train")
app.add_typer(data_app, name="data")
app.add_typer(viz_app, name="viz")


@app.command()
def version() -> None:
    console.print(f"deepiri-fuselk {__version__}")


@app.command()
def doctor() -> None:
    modules = ["numpy", "scipy", "xarray", "pydantic", "zmq", "pyarrow", "gymnasium", "stable_baselines3"]
    table = Table(title="fuselk doctor")
    table.add_column("Module")
    table.add_column("Status")
    ok = True
    for name in modules:
        try:
            importlib.import_module(name)
            table.add_row(name, "[green]ok[/green]")
        except ImportError as exc:
            ok = False
            table.add_row(name, f"[red]missing: {exc}[/red]")
    console.print(table)
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def benchmark(all_modules: bool = typer.Option(True, "--all"), rl_steps: int = 5000) -> None:
    """Run full benchmark suite."""
    script = Path(__file__).resolve().parents[3] / "scripts" / "benchmark.py"
    cmd = [sys.executable, str(script), "--rl-steps", str(rl_steps)]
    if all_modules:
        cmd.append("--all")
    out = subprocess.check_output(cmd, text=True)
    console.print(out)
    console.print("[green]Benchmark complete.[/green]")


@sim_app.command("oil-water")
def sim_oil_water(steady: bool = True, transient: bool = False) -> None:
    from deepiri_fuselk.physics.pde_solver import solve_oil_water_steady, solve_oil_water_transient

    if steady:
        r = solve_oil_water_steady(n_grid=64)
        console.print(f"Steady: converged={r.converged} residual={r.residual:.2e} iter={r.iterations}")
    if transient:
        hist = solve_oil_water_transient(n_grid=32, t_end=1.0)
        console.print(f"Transient: {len(hist)} steps, final n_T wall={hist[-1].n_T[-1]:.4e}")


@sim_app.command("run")
def sim_run(steps: int = 100, grid: int = 32) -> None:
    from deepiri_fuselk.sim.digital_twin import DigitalTwin

    twin = DigitalTwin(grid_size=grid)
    twin.reset()
    for _ in range(steps):
        twin.step()
    console.print_json(json.dumps(twin.summary()))


@sim_app.command("muon")
def sim_muon() -> None:
    from deepiri_fuselk.muon import RateNetworkParams, run_rate_network

    r = run_rate_network(params=RateNetworkParams(R_photon=0.5, R_proton=0.3))
    console.print(
        f"fusions/muon={r.fusions_per_muon:.1f} sticking={r.effective_sticking:.4f} "
        f"breakeven={r.breakeven}"
    )


@train_app.command("vent")
def train_vent(timesteps: int = 20_000, output: Path = Path(".fuselk-data/policies/vent_ppo")) -> None:
    from deepiri_fuselk.control.rl_agent import train_vent_policy

    r = train_vent_policy(timesteps=timesteps, save_path=output)
    console.print(f"Mean reward={r.mean_reward:.2f} saved={r.policy_path}")


@data_app.command("export")
def data_export(shot_id: str = "SYN001", output: Path = Path(".fuselk-data/shots/export.h5")) -> None:
    from deepiri_fuselk.data.imas_loader import export_imas_hdf5, synthetic_imas_shot

    shot = synthetic_imas_shot(shot_id)
    p = export_imas_hdf5(shot, output)
    console.print(f"Exported {shot_id} -> {p}")


@data_app.command("import")
def data_import(path: Path) -> None:
    from deepiri_fuselk.data.imas_loader import load_imas_hdf5

    shot = load_imas_hdf5(path)
    console.print(f"Loaded {shot.shot_id} device={shot.device} heat={shot.heat_field.shape}")


@viz_app.command("serve")
def viz_serve(host: str = "127.0.0.1", port: int = 8050) -> None:
    from deepiri_fuselk.viz.dashboard.app import create_app

    create_app().run_server(host=host, port=port, debug=False)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
