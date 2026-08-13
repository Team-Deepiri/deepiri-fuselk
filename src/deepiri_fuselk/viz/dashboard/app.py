"""Rich multi-panel fuselk control room with live simulation."""

from __future__ import annotations

from pathlib import Path

from dash import Dash, Input, Output, State, dcc, html
from deepiri_fuselk.viz.dashboard.figures import (
    TRACE_FIELDS,
    build_control_room_figure,
    build_equilibrium_figure,
    build_kpi_strip,
    build_power_balance_figure,
    build_stability_figure,
    build_status_panel_figure,
    build_trace_figure,
)
from deepiri_fuselk.viz.dashboard.theme import (
    ACCENT,
    BG,
    FAINT,
    FONT_STACK,
    MUTED,
    STAT_CARD_STYLE,
    STAT_LABEL_STYLE,
    STAT_VALUE_STYLE,
    TEXT,
)
from deepiri_fuselk.viz.simulation_engine import LiveSimulation, get_preset_names, list_device_names

_STATIC = Path(__file__).resolve().parent.parent / "static"
_sim = LiveSimulation(grid_size=24)
_HISTORY_MAXLEN = 300
_history: list[dict] = []


def _frame_history_entry(frame) -> dict:
    return {
        "step": frame.step,
        "ip_ma": frame.ip_ma,
        "target_ip_ma": frame.target_ip_ma,
        "beta_n": frame.beta_n,
        "target_beta_n": frame.target_beta_n,
        "li": frame.li,
        "target_li": frame.target_li,
        "d_alpha": frame.d_alpha,
        "target_d_alpha": frame.target_d_alpha,
    }


def _record_history(frame, *, reset: bool = False) -> None:
    if reset:
        _history.clear()
    _history.append(_frame_history_entry(frame))
    del _history[:-_HISTORY_MAXLEN]


def create_app() -> Dash:
    """Create the full fuselk control room with live FusionCell simulation."""
    _sim.set_device("ITER")
    _sim.set_preset("H-mode")
    frame = _sim.reset(seed=0)
    _record_history(frame, reset=True)

    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        assets_folder=str(_STATIC),
    )
    app.layout = html.Div(
        style={
            "fontFamily": FONT_STACK,
            "padding": "16px",
            "backgroundColor": BG,
            "color": TEXT,
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
                                style={"margin": 0, "color": MUTED},
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="stat-row",
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
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
            html.Div(
                style={
                    "display": "flex",
                    "gap": "12px",
                    "alignItems": "center",
                    "flexWrap": "wrap",
                    "margin": "12px 0",
                },
                children=[
                    html.Label("Device:", style={"color": MUTED}),
                    dcc.Dropdown(
                        id="device-select",
                        options=[{"label": d, "value": d} for d in list_device_names()],
                        value="ITER",
                        clearable=False,
                        style={"width": "160px", "color": "#000"},
                    ),
                    html.Label("Preset:", style={"color": MUTED, "marginLeft": "8px"}),
                    dcc.Dropdown(
                        id="preset-select",
                        options=[{"label": p, "value": p} for p in get_preset_names("ITER")],
                        value="H-mode",
                        clearable=False,
                        style={"width": "170px", "color": "#000"},
                    ),
                    html.Button("▶ Play", id="btn-play", n_clicks=0, style=_button_style()),
                    html.Label("Speed:", style={"color": MUTED, "marginLeft": "8px"}),
                    dcc.Dropdown(
                        id="speed-select",
                        options=[{"label": f"{s}x", "value": s} for s in (0.5, 1.0, 1.5, 2.0)],
                        value=1.0,
                        clearable=False,
                        style={"width": "90px", "color": "#000"},
                    ),
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
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "12px",
                    "marginTop": "12px",
                },
                children=[
                    dcc.Graph(id="equilibrium", figure=build_equilibrium_figure(frame)),
                    html.Div(
                        children=[
                            html.Label("Traces:", style={"color": MUTED}),
                            dcc.Dropdown(
                                id="trace-select",
                                options=[{"label": k, "value": k} for k in TRACE_FIELDS],
                                value=list(TRACE_FIELDS.keys()),
                                multi=True,
                                style={"color": "#000", "marginBottom": "8px"},
                            ),
                            dcc.Graph(id="trace-panel", figure=build_trace_figure(_history)),
                        ]
                    ),
                ],
            ),
            html.H3("Status panel", style={"marginTop": "16px", "color": "#fff"}),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                    "gap": "12px",
                },
                children=[
                    _stat_card("Core: Ip / Bt / Te0", "stat-core", "—"),
                    _stat_card("Core: n̄e / Wth / τE", "stat-core2", "—"),
                ],
            ),
            dcc.Graph(id="power-balance", figure=build_power_balance_figure(frame)),
            dcc.Graph(id="stability", figure=build_stability_figure(frame)),
            dcc.Graph(id="fusion-performance", figure=build_status_panel_figure(frame)),
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
                        style={"marginLeft": "auto", "color": ACCENT},
                    ),
                ],
            ),
            dcc.Interval(id="interval", interval=2000, n_intervals=0),
            html.Footer(
                "deepiri-fuselk v0.4 — Fusion Unified Simulation, ELM Learning & Kinetics",
                style={"marginTop": "24px", "color": FAINT, "fontSize": "12px"},
            ),
        ],
    )

    @app.callback(
        Output("interval", "interval"),
        Input("interval-ms", "value"),
        Input("speed-select", "value"),
    )
    def set_interval(ms: int | None, speed: float | None) -> int:
        base = max(500, int(ms or 2000))
        return max(200, int(base / float(speed or 1.0)))

    @app.callback(
        Output("btn-play", "children"),
        Input("btn-play", "n_clicks"),
    )
    def toggle_play_label(n_clicks: int) -> str:
        playing = (n_clicks or 0) % 2 == 1
        return "⏸ Pause" if playing else "▶ Play"

    @app.callback(
        Output("interval", "disabled"),
        Input("btn-play", "n_clicks"),
    )
    def toggle_play(n_clicks: int) -> bool:
        # 0 clicks -> playing (not disabled); each click toggles play/pause.
        playing = (n_clicks or 0) % 2 == 0
        return not playing

    @app.callback(
        Output("preset-select", "options"),
        Input("device-select", "value"),
    )
    def refresh_presets(device: str) -> list[dict]:
        return [{"label": p, "value": p} for p in get_preset_names(device)]

    @app.callback(
        Output("control-room", "figure"),
        Output("kpi-strip", "figure"),
        Output("equilibrium", "figure"),
        Output("trace-panel", "figure"),
        Output("power-balance", "figure"),
        Output("stability", "figure"),
        Output("fusion-performance", "figure"),
        Output("stat-step", "children"),
        Output("stat-opoint", "children"),
        Output("stat-snr", "children"),
        Output("stat-action", "children"),
        Output("stat-core", "children"),
        Output("stat-core2", "children"),
        Input("interval", "n_intervals"),
        Input("btn-reset", "n_clicks"),
        Input("device-select", "value"),
        Input("preset-select", "value"),
        Input("trace-select", "value"),
        Input("trace-panel", "relayoutData"),
        State("interval", "n_intervals"),
        prevent_initial_call=False,
    )
    def tick(
        n_intervals: int,
        reset_clicks: int,
        device: str | None,
        preset: str | None,
        trace_selection: list[str] | None,
        relayout: dict | None,
        _prev: int,
    ):
        from dash import callback_context

        ctx = callback_context
        trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

        if device:
            _sim.set_device(device)
        if preset:
            _sim.set_preset(preset)

        if trigger.startswith("btn-reset"):
            frame = _sim.reset(seed=int(reset_clicks or 0))
            _record_history(frame, reset=True)
        elif trigger.startswith("device-select") or trigger.startswith("preset-select"):
            frame = _sim.step()
            _record_history(frame)
        elif trigger.startswith("trace-select") or trigger.startswith("trace-panel"):
            frame = _sim.last_frame or _sim.reset(seed=0)
        else:
            frame = _sim.step()
            _record_history(frame)

        op = f"({frame.helix.o_point[0]:.2f}, {frame.helix.o_point[1]:.2f})"

        scrub_range = None
        if relayout and "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
            scrub_range = (relayout["xaxis.range[0]"], relayout["xaxis.range[1]"])

        core1 = (
            f"Ip {frame.ip_ma:.2f} MA · Bt {frame.active_device.max_bt_t:.1f} T · "
            f"Te0 {frame.te0_kev:.1f} keV"
        )
        core2 = f"n̄e {frame.ne_bar_1e19:.1f}e19 · Wth {frame.w_th_mj:.1f} MJ · τE {frame.tau_e_s:.2f} s"

        return (
            build_control_room_figure(frame),
            build_kpi_strip(frame),
            build_equilibrium_figure(frame),
            build_trace_figure(_history, trace_selection, scrub_range),
            build_power_balance_figure(frame),
            build_stability_figure(frame),
            build_status_panel_figure(frame),
            str(frame.step),
            op,
            f"{frame.helix.phase_locked_snr:.1f}x",
            frame.action,
            core1,
            core2,
        )

    return app


def _stat_card(title: str, elem_id: str, value: str) -> html.Div:
    return html.Div(
        style=STAT_CARD_STYLE,
        children=[
            html.H4(title, style=STAT_LABEL_STYLE),
            html.P(value, id=elem_id, style=STAT_VALUE_STYLE),
        ],
    )


def _button_style() -> dict:
    return {
        "background": ACCENT,
        "color": "#fff",
        "border": "none",
        "borderRadius": "8px",
        "padding": "8px 16px",
        "cursor": "pointer",
        "fontWeight": 600,
    }
