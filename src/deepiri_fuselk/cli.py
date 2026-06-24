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

app = typer.Typer(
    name="fuselk", help="Fusion Unified Simulation, ELM Learning & Kinetics", no_args_is_help=True
)
console = Console()

sim_app = typer.Typer(help="Run simulations")
train_app = typer.Typer(help="Train RL policies")
data_app = typer.Typer(help="Data import/export")
viz_app = typer.Typer(help="Visualization")
app.add_typer(sim_app, name="sim")
app.add_typer(train_app, name="train")
app.add_typer(data_app, name="data")
reactor_app = typer.Typer(help="Full reactor cell simulation")
app.add_typer(reactor_app, name="reactor")
experiments_app = typer.Typer(help="Run catalog experiments")
app.add_typer(experiments_app, name="experiments")
app.add_typer(viz_app, name="viz")


@app.command()
def version() -> None:
    console.print(f"deepiri-fuselk {__version__}")


@app.command()
def doctor(
    vision: bool = typer.Option(True, "--vision/--no-vision", help="Run VISION.md alignment audit"),
    skip_slow: bool = typer.Option(False, "--skip-slow", help="Skip RL value-iteration audit"),
) -> None:
    modules = [
        "numpy",
        "scipy",
        "xarray",
        "pydantic",
        "zmq",
        "pyarrow",
        "gymnasium",
        "stable_baselines3",
    ]
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

    if vision:
        from deepiri_fuselk.sim.vision_alignment import audit_vision_alignment

        report = audit_vision_alignment(skip_slow=skip_slow)
        vtable = Table(title="VISION.md alignment")
        vtable.add_column("Pillar")
        vtable.add_column("Section")
        vtable.add_column("Status")
        for pillar in report.pillars:
            status = "[green]ok[/green]" if pillar.satisfied else f"[yellow]{pillar.gap}[/yellow]"
            vtable.add_row(pillar.name, pillar.vision_section, status)
        console.print(vtable)
        if report.gaps:
            ok = False
            console.print("[yellow]Gaps:[/yellow]")
            for gap in report.gaps:
                console.print(f"  • {gap}")
        else:
            console.print("[green]All VISION pillars aligned with implementation.[/green]")

    if not ok:
        raise typer.Exit(code=1)


@app.command()
def benchmark(rl_steps: int = 5000) -> None:
    """Run full benchmark suite."""
    script = Path(__file__).resolve().parents[3] / "scripts" / "benchmark.py"
    cmd = [sys.executable, str(script), "--rl-steps", str(rl_steps), "--all"]
    out = subprocess.check_output(cmd, text=True)
    console.print(out)
    console.print("[green]Benchmark complete.[/green]")


@sim_app.command("oil-water")
def sim_oil_water(
    mode: str = typer.Option("steady", help="steady, transient, or both"),
) -> None:
    from deepiri_fuselk.physics.pde_solver import solve_oil_water_steady, solve_oil_water_transient

    steady = mode in ("steady", "both")
    transient = mode in ("transient", "both")
    if steady:
        r = solve_oil_water_steady(n_grid=64)
        console.print(
            f"Steady: converged={r.converged} residual={r.residual:.2e} iter={r.iterations}"
        )
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
def train_vent(
    timesteps: int = 20_000, output: Path = Path(".fuselk-data/policies/vent_ppo")
) -> None:
    from deepiri_fuselk.control.rl_agent import train_vent_policy

    r = train_vent_policy(timesteps=timesteps, save_path=output, verify_convergence=True)
    console.print(f"Mean reward={r.mean_reward:.2f} saved={r.policy_path}")


@data_app.command("export")
def data_export(
    shot_id: str = "SYN001", output: Path = Path(".fuselk-data/shots/export.h5")
) -> None:
    from deepiri_fuselk.data.imas_loader import export_imas_hdf5, synthetic_imas_shot

    shot = synthetic_imas_shot(shot_id)
    p = export_imas_hdf5(shot, output)
    console.print(f"Exported {shot_id} -> {p}")


@data_app.command("import")
def data_import(path: Path) -> None:
    from deepiri_fuselk.data.imas_loader import load_imas_hdf5

    shot = load_imas_hdf5(path)
    console.print(f"Loaded {shot.shot_id} device={shot.device} heat={shot.heat_field.shape}")


@data_app.command("fetch")
def data_fetch(
    all_sources: bool = typer.Option(False, "--all", help="Fetch all public sources"),
    source: list[str] = typer.Option(None, "--source", help="Source id (repeatable)"),
    root: Path = typer.Option(Path(".fuselk-data"), "--root"),
    force: bool = typer.Option(False, "--force"),
    shots: int = typer.Option(100, "--shots"),
    grid: int = typer.Option(32, "--grid"),
    max_odl: int = typer.Option(50, "--max-odl"),
) -> None:
    """Download and normalize public fusion datasets into .fuselk-data/."""
    from deepiri_fuselk.data.fetchers import run_fetch

    selected = source if source else (None if not all_sources else None)
    if not all_sources and not source:
        selected = None  # defaults inside run_fetch
    results = run_fetch(
        root,
        selected,
        force=force,
        n_shots=shots,
        grid_size=grid,
        max_odl_discharges=max_odl,
    )
    console.print_json(json.dumps({k: v.__dict__ for k, v in results.items()}, indent=2))


@data_app.command("catalog")
def data_catalog() -> None:
    """List registered data sources and feedback loops."""
    from deepiri_fuselk.data.fetchers import FETCHERS
    from deepiri_fuselk.data.sources import load_catalog

    sources, loops = load_catalog()
    table = Table(title="fuselk data sources")
    table.add_column("ID")
    table.add_column("Tier")
    table.add_column("Device")
    table.add_column("Fetch")
    for s in sources:
        table.add_row(s.id, s.tier, s.device, "yes" if s.id in FETCHERS else "manual")
    console.print(table)
    console.print("\n[bold]Feedback loops[/bold]")
    for fb in loops:
        console.print(f"  {fb.name}: {fb.in_} → {fb.out}")
    console.print("\nDocs: docs/DATA_PIPELINE.md")


@data_app.command("manifest")
def data_manifest(root: Path = typer.Option(Path(".fuselk-data"), "--root")) -> None:
    """Show fetch manifest (provenance, checksums, shot counts)."""
    from deepiri_fuselk.data.fetchers.manifest import load_manifest

    console.print_json(json.dumps(load_manifest(root).to_dict(), indent=2))


@sim_app.command("reactor")
def sim_reactor(
    steps: int = typer.Option(100, "--steps"),
    grid: int = typer.Option(32, "--grid"),
    train_elm: bool = typer.Option(False, "--train-elm", help="Train ELM on synthetic corpus"),
    output: Path | None = typer.Option(None, "--output", help="Save JSON report"),
) -> None:
    """Run closed-loop reactor cell with fusion KPI scoring."""
    from deepiri_fuselk.sim.reactor_cell import ReactorCell

    cell = ReactorCell(grid_size=grid, train_elm=train_elm)
    run = cell.run(n_steps=steps, seed=42)
    report = run.to_report()
    report["fusion_score"] = run.final_score
    if output:
        output.write_text(json.dumps(report, indent=2))
        console.print(f"[green]Report saved to {output}[/green]")
    console.print_json(json.dumps(report))


@train_app.command("elm")
def train_elm(
    shots: int = 300,
    grid: int = 32,
    output: Path = Path(".fuselk-data/models/elm_predictor.json"),
) -> None:
    """Train ELM predictor on labeled synthetic corpus."""
    from deepiri_fuselk.models.elm_predictor import ELMPredictor
    from deepiri_fuselk.sim.shot_corpus import generate_corpus

    corpus = generate_corpus(n_shots=shots, grid_size=grid, seed=42)
    model = ELMPredictor()
    acc = model.train_from_corpus(corpus)
    model.save(output)
    console.print(f"Train accuracy={acc:.3f} corpus={shots} saved={output}")


@reactor_app.command("run")
def reactor_run(
    steps: int = typer.Option(100, "--steps"),
    grid: int = typer.Option(32, "--grid"),
    report: Path | None = typer.Option(None, "--report", help="Save JSON report"),
) -> None:
    sim_reactor(steps=steps, grid=grid, output=report)


@reactor_app.command("score")
def reactor_score(
    steps: int = typer.Option(50, "--steps"),
    grid: int = typer.Option(16, "--grid"),
) -> None:
    """Quick fusion progress score (fast benchmark)."""
    from deepiri_fuselk.sim.reactor_cell import ReactorCell

    run = ReactorCell(grid_size=grid, train_elm=False).run(n_steps=steps, seed=0)
    console.print(f"[bold]Fusion score:[/bold] {run.final_score:.3f}")
    console.print_json(json.dumps(run.to_report()))


@sim_app.command("fusion")
def sim_fusion(
    steps: int = typer.Option(50, "--steps"),
    grid: int = typer.Option(24, "--grid"),
    output: Path | None = typer.Option(None, "--output", help="Save JSON report"),
) -> None:
    """Run full FusionCell (reactor + fuel cycle + muon trifecta)."""
    from deepiri_fuselk.sim.fusion_cell import FusionCell

    _, report = FusionCell(grid_size=grid, train_elm=False).run(n_steps=steps, seed=42)
    data = report.to_dict()
    if output:
        output.write_text(json.dumps(data, indent=2))
        console.print(f"[green]Saved {output}[/green]")
    console.print_json(json.dumps(data))


@experiments_app.command("list")
def experiments_list() -> None:
    from deepiri_fuselk.experiments.registry import load_registry

    table = Table(title="fuselk experiments")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Category")
    table.add_column("Description")
    for e in load_registry():
        table.add_row(e.id, e.status, e.category, e.description[:60])
    console.print(table)


@experiments_app.command("run")
def experiments_run(exp_id: str) -> None:
    from deepiri_fuselk.experiments.runner import run_experiment

    result = run_experiment(exp_id)
    console.print_json(json.dumps(result, indent=2))


@viz_app.command("sim")
def viz_sim(host: str = "127.0.0.1", port: int = 8050) -> None:
    """Launch live FusionCell simulation dashboard."""
    viz_serve(host=host, port=port)


@viz_app.command("serve")
def viz_serve(host: str = "127.0.0.1", port: int = 8050) -> None:
    from deepiri_fuselk.viz.dashboard.app import create_app

    create_app().run_server(host=host, port=port, debug=False)


@app.command("gui")
def gui_launch(
    host: str = typer.Option("127.0.0.1", "--host"),
    dash_port: int = typer.Option(8050, "--dash-port"),
    api_port: int = typer.Option(8765, "--api-port"),
) -> None:
    """Launch the PySide6 desktop control room (Dash + native tools)."""
    from deepiri_fuselk.viz.desktop.app import run_desktop_gui

    run_desktop_gui(dash_port=dash_port, api_port=api_port, host=host)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
