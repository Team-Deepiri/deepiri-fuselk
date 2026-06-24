"""fuselk CLI — entry point for simulation, visualization, and diagnostics."""

from __future__ import annotations

import importlib

import typer
from rich.console import Console
from rich.table import Table

from deepiri_fuselk import __version__

app = typer.Typer(
    name="fuselk",
    help="Fusion Unified Simulation, ELM Learning & Kinetics",
    no_args_is_help=True,
)
console = Console()


@app.command()
def version() -> None:
    """Print package version."""
    console.print(f"deepiri-fuselk {__version__}")


@app.command()
def doctor() -> None:
    """Check that core dependencies import correctly."""
    modules = [
        "numpy",
        "scipy",
        "xarray",
        "pydantic",
        "plotly",
        "dash",
        "gymnasium",
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
    if not ok:
        raise typer.Exit(code=1)
    console.print("[green]All core modules healthy.[/green]")


sim_app = typer.Typer(help="Run simulations")
viz_app = typer.Typer(help="Visualization commands")
app.add_typer(sim_app, name="sim")
app.add_typer(viz_app, name="viz")


@sim_app.command("oil-water")
def sim_oil_water(steps: int = typer.Option(100, help="Solver iterations")) -> None:
    """Run 1D oil-water barrier steady-state solve."""
    from deepiri_fuselk.physics.pde_solver import solve_oil_water_steady

    result = solve_oil_water_steady(n_grid=64, max_iter=steps)
    console.print(
        f"Solved oil-water PDE: converged={result.converged}, "
        f"iterations={result.iterations}, residual={result.residual:.2e}"
    )


@sim_app.command("run")
def sim_run(episodes: int = typer.Option(10, help="Digital twin episodes")) -> None:
    """Run digital twin simulation loop."""
    from deepiri_fuselk.sim.digital_twin import DigitalTwin

    twin = DigitalTwin()
    for ep in range(episodes):
        state = twin.reset()
        for _ in range(50):
            state = twin.step()
        console.print(f"Episode {ep + 1}: final heat variance={state.heat_variance:.4f}")


@viz_app.command("serve")
def viz_serve(host: str = "127.0.0.1", port: int = 8050) -> None:
    """Launch the Dash dashboard."""
    from deepiri_fuselk.viz.dashboard.app import create_app

    dash_app = create_app()
    console.print(f"Starting fuselk dashboard at http://{host}:{port}")
    dash_app.run_server(host=host, port=port, debug=False)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
