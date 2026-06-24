"""Rich multi-panel fuselk control room with live simulation."""

from __future__ import annotations

from pathlib import Path

from dash import Dash, Input, Output, State, dcc, html
from deepiri_fuselk.viz.dashboard.figures import build_control_room_figure, build_kpi_strip
from deepiri_fuselk.viz.simulation_engine import LiveSimulation

_STATIC = Path(__file__).resolve().parent.parent / "static"
_sim = LiveSimulation(grid_size=24)


def create_app() -> Dash:
    """Create the full fuselk control room with live FusionCell simulation."""
    frame = _sim.reset(seed=0)

    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        assets_folder=str(_STATIC),
    )
    app.layout = html.Div(
        style={
            "fontFamily": "system-ui, sans-serif",
            "padding": "16px",
            "backgroundColor": "#0f1117",
            "color": "#e8e8e8",
            "minHeight": "100vh",
        },
        children=[
            html.Header(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "14px",
                    "marginBottom": "4px",
                },
                children=[
                    html.Img(
                        src="/assets/branding/deepiri_logo.png",
                        alt="Deepiri",
                        style={"height": "44px", "width": "44px"},
                    ),
                    html.Div(
                        children=[
                            html.H1(
                                "deepiri-fuselk Control Room",
                                style={"margin": "0 0 4px 0", "color": "#fff"},
                            ),
                            html.P(
                                "Live FusionCell simulation · HELIX · Venturi · ELM/disruption · fuel & muon cycle",
                                style={"margin": 0, "color": "#888"},
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, 1fr)",
                    "gap": "12px",
                    "margin": "16px 0",
                },
                children=[
                    _stat_card("Step", "stat-step", f"{frame.step}"),
                    _stat_card(
                        "O-Point",
                        "stat-opoint",
                        f"({frame.helix.o_point[0]:.2f}, {frame.helix.o_point[1]:.2f})",
                    ),
                    _stat_card("SNR", "stat-snr", f"{frame.helix.phase_locked_snr:.1f}x"),
                    _stat_card("Action", "stat-action", frame.action),
                ],
            ),
            dcc.Graph(
                id="kpi-strip",
                figure=build_kpi_strip(frame),
                config={"displayModeBar": False},
                style={"height": "220px"},
            ),
            dcc.Graph(id="control-room", figure=build_control_room_figure(frame)),
            html.Div(
                style={
                    "display": "flex",
                    "gap": "12px",
                    "alignItems": "center",
                    "marginTop": "12px",
                },
                children=[
                    html.Button(
                        "Reset simulation",
                        id="btn-reset",
                        n_clicks=0,
                        style=_button_style(),
                    ),
                    html.Label("Update interval (ms): ", style={"marginLeft": "8px"}),
                    dcc.Input(
                        id="interval-ms",
                        type="number",
                        value=2000,
                        min=500,
                        max=10000,
                        step=500,
                        style={"width": "80px", "marginLeft": "4px"},
                    ),
                    html.A(
                        "Open 3D Tokamak Viewer",
                        href="/assets/tokamak_viewer.html",
                        target="_blank",
                        style={"marginLeft": "auto", "color": "#66aaff"},
                    ),
                ],
            ),
            dcc.Interval(id="interval", interval=2000, n_intervals=0),
            html.Footer(
                "deepiri-fuselk v0.4 — Fusion Unified Simulation, ELM Learning & Kinetics",
                style={"marginTop": "24px", "color": "#555", "fontSize": "12px"},
            ),
        ],
    )

    @app.callback(
        Output("interval", "interval"),
        Input("interval-ms", "value"),
    )
    def set_interval(ms: int | None) -> int:
        return max(500, int(ms or 2000))

    @app.callback(
        Output("control-room", "figure"),
        Output("kpi-strip", "figure"),
        Output("stat-step", "children"),
        Output("stat-opoint", "children"),
        Output("stat-snr", "children"),
        Output("stat-action", "children"),
        Input("interval", "n_intervals"),
        Input("btn-reset", "n_clicks"),
        State("interval", "n_intervals"),
        prevent_initial_call=False,
    )
    def tick(n_intervals: int, reset_clicks: int, _prev: int):
        from dash import callback_context

        ctx = callback_context
        if ctx.triggered and ctx.triggered[0]["prop_id"].startswith("btn-reset"):
            frame = _sim.reset(seed=int(reset_clicks or 0))
        else:
            frame = _sim.step()
        op = f"({frame.helix.o_point[0]:.2f}, {frame.helix.o_point[1]:.2f})"
        return (
            build_control_room_figure(frame),
            build_kpi_strip(frame),
            str(frame.step),
            op,
            f"{frame.helix.phase_locked_snr:.1f}x",
            frame.action,
        )

    return app


def _stat_card(title: str, elem_id: str, value: str) -> html.Div:
    return html.Div(
        style={
            "background": "#1a1d27",
            "borderRadius": "8px",
            "padding": "12px 16px",
            "border": "1px solid #2a2d37",
        },
        children=[
            html.H4(title, style={"margin": "0 0 4px 0", "fontSize": "12px", "color": "#888"}),
            html.P(value, id=elem_id, style={"margin": 0, "fontSize": "18px", "fontWeight": 600}),
        ],
    )


def _button_style() -> dict:
    return {
        "background": "#4488ff",
        "color": "#fff",
        "border": "none",
        "borderRadius": "6px",
        "padding": "8px 16px",
        "cursor": "pointer",
        "fontWeight": 600,
    }
