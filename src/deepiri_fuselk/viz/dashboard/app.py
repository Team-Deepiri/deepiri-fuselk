"""Rich multi-panel fuselk control room with live simulation.

Layout follows the reference tokamak-simulator control room: a dark top
toolbar (device/preset/start/reset/speed/edit + t=.../10.0s readout), an
equilibrium cross-section top-left, a stacked oscilloscope-style trace column
top-right, status/diagnostics bottom-left, and the port/3D view bottom-right.
Older single-column panels (KPI strip, 2x3 control room, batch KPIs) are kept
below in an "Advanced diagnostics" section for parity with the desktop app.
"""

from __future__ import annotations

from pathlib import Path

from dash import Dash, Input, Output, State, dcc, html
from deepiri_fuselk.viz.dashboard.figures import (
    OSCILLOSCOPE_FIELDS,
    build_control_room_figure,
    build_equilibrium_figure,
    build_kpi_strip,
    build_oscilloscope_figure,
    build_power_balance_figure,
    build_stability_figure,
    build_status_panel_figure,
)
from deepiri_fuselk.viz.dashboard.theme import (
    BG,
    BORDER,
    CARD,
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

# Fixed pulse duration (matches the "t=.../10.0s" readout in the reference
# UI): each Interval tick advances the simulated shot clock by _DT_S,
# stepping the underlying FusionCell loop; the shot halts automatically once
# it reaches _DURATION_S so the trace panel has a finite, scrubbable window.
_DT_S = 0.1
_DURATION_S = 10.0
_MAX_STEPS = int(_DURATION_S / _DT_S)
_SPEEDS = (2.0, 1.0, 0.75, 0.5)


def _speed_id(s: float) -> str:
    return f"btn-speed-{s:g}".replace(".", "-")


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
        "w_th_mj": frame.w_th_mj,
    }


def _record_history(frame, *, reset: bool = False) -> None:
    if reset:
        _history.clear()
    _history.append(_frame_history_entry(frame))
    del _history[:-_HISTORY_MAXLEN]


def _pulse_step() -> int:
    """Current step count within the fixed-duration pulse (0.._MAX_STEPS)."""
    frame = _sim.last_frame
    return 0 if frame is None else min(frame.step, _MAX_STEPS)


def _pulse_finished() -> bool:
    return _pulse_step() >= _MAX_STEPS


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
            _build_toolbar(),
            dcc.Store(id="scrub-store", data=None),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gridTemplateRows": "auto auto",
                    "gap": "12px",
                    "marginTop": "8px",
                },
                children=[
                    _panel_card(
                        "Equilibrium",
                        dcc.Graph(
                            id="equilibrium",
                            figure=build_equilibrium_figure(frame),
                            config={"displayModeBar": False},
                        ),
                    ),
                    _panel_card(
                        "Traces",
                        html.Div(
                            children=[
                                dcc.Dropdown(
                                    id="trace-select",
                                    options=[{"label": k, "value": k} for k in OSCILLOSCOPE_FIELDS],
                                    value=list(OSCILLOSCOPE_FIELDS.keys()),
                                    multi=True,
                                    style={"marginBottom": "8px"},
                                ),
                                dcc.Graph(
                                    id="trace-panel",
                                    figure=build_oscilloscope_figure(_history, dt_s=_DT_S),
                                    config={"displayModeBar": False},
                                ),
                            ]
                        ),
                    ),
                    _panel_card(
                        "Status",
                        html.Div(
                            children=[
                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "1fr 1fr",
                                        "gap": "8px",
                                        "marginBottom": "8px",
                                    },
                                    children=[
                                        _stat_card("Core: Ip / Bt / Te0", "stat-core", "—"),
                                        _stat_card("Core: n̄e / Wth / τE", "stat-core2", "—"),
                                    ],
                                ),
                                dcc.Graph(
                                    id="power-balance",
                                    figure=build_power_balance_figure(frame),
                                    config={"displayModeBar": False},
                                ),
                                dcc.Graph(
                                    id="stability",
                                    figure=build_stability_figure(frame),
                                    config={"displayModeBar": False},
                                ),
                                dcc.Graph(
                                    id="fusion-performance",
                                    figure=build_status_panel_figure(frame),
                                    config={"displayModeBar": False},
                                ),
                            ]
                        ),
                    ),
                    _panel_card(
                        "Port view",
                        html.Iframe(
                            src="/assets/tokamak_viewer.html",
                            style={
                                "width": "100%",
                                "height": "460px",
                                "border": "none",
                                "borderRadius": "8px",
                            },
                        ),
                    ),
                ],
            ),
            html.Details(
                style={"marginTop": "16px"},
                children=[
                    html.Summary(
                        "Advanced diagnostics (legacy KPI strip / control room)",
                        style={"cursor": "pointer", "color": MUTED},
                    ),
                    dcc.Graph(
                        id="kpi-strip",
                        figure=build_kpi_strip(frame),
                        config={"displayModeBar": False},
                        style={"height": "220px"},
                    ),
                    dcc.Graph(id="control-room", figure=build_control_room_figure(frame)),
                ],
            ),
            dcc.Interval(id="interval", interval=int(1000 * _DT_S), n_intervals=0),
            html.Footer(
                "deepiri-fuselk v0.4 — Fusion Unified Simulation, ELM Learning & Kinetics",
                style={"marginTop": "24px", "color": FAINT, "fontSize": "12px"},
            ),
        ],
    )

    _register_callbacks(app)
    return app


def _build_toolbar() -> html.Div:
    return html.Div(
        style={
            "display": "flex",
            "gap": "10px",
            "alignItems": "center",
            "flexWrap": "wrap",
            "margin": "12px 0",
            "padding": "10px 14px",
            "background": CARD,
            "border": f"1px solid {BORDER}",
            "borderRadius": "10px",
        },
        children=[
            html.Label("Device:", style={"color": MUTED}),
            dcc.Dropdown(
                id="device-select",
                options=[{"label": d, "value": d} for d in list_device_names()],
                value="ITER",
                clearable=False,
                style={"width": "150px"},
            ),
            html.Label("Preset:", style={"color": MUTED}),
            dcc.Dropdown(
                id="preset-select",
                options=[{"label": p, "value": p} for p in get_preset_names("ITER")],
                value="H-mode",
                clearable=False,
                style={"width": "160px"},
            ),
            html.Button("▶ Start", id="btn-play", n_clicks=0, className="fuselk-toolbar-btn"),
            html.Button("⟲ Reset", id="btn-reset", n_clicks=0, className="fuselk-toolbar-btn"),
            html.Div(
                id="speed-buttons",
                style={"display": "flex", "gap": "4px"},
                children=[
                    html.Button(
                        f"{s:g}x",
                        id=_speed_id(s),
                        n_clicks=0,
                        className="fuselk-toolbar-btn" + (" active" if s == 1.0 else ""),
                    )
                    for s in _SPEEDS
                ],
            ),
            html.Button("✎ Edit", id="btn-edit", n_clicks=0, className="fuselk-toolbar-btn"),
            html.Div(style={"flex": "1 1 auto"}),
            html.Span(
                id="pulse-clock",
                children=f"t = 0.0 / {_DURATION_S:.1f}s",
                style={"fontFamily": "monospace", "fontSize": "16px", "color": TEXT},
            ),
            html.Div(
                id="preset-editor",
                style={
                    "display": "none",
                    "width": "100%",
                    "marginTop": "8px",
                    "color": MUTED,
                    "fontSize": "12px",
                    "fontFamily": "monospace",
                    "whiteSpace": "pre-wrap",
                },
            ),
        ],
    )


def _panel_card(title: str, body) -> html.Div:
    return html.Div(
        style={
            "background": CARD,
            "border": f"1px solid {BORDER}",
            "borderRadius": "10px",
            "padding": "10px",
        },
        children=[
            html.H4(title, style={"margin": "0 0 8px 0", "color": MUTED, "fontSize": "13px"}),
            body,
        ],
    )


def _register_callbacks(app: Dash) -> None:  # noqa: C901 - dashboard wiring, kept together for clarity
    @app.callback(
        Output("preset-select", "options"),
        Input("device-select", "value"),
    )
    def refresh_presets(device: str) -> list[dict]:
        return [{"label": p, "value": p} for p in get_preset_names(device)]

    @app.callback(
        [Output(_speed_id(s), "className") for s in _SPEEDS] + [Output("interval", "interval")],
        [Input(_speed_id(s), "n_clicks") for s in _SPEEDS],
    )
    def highlight_speed(*_clicks):
        from dash import callback_context

        ctx = callback_context
        chosen = 1.0
        if ctx.triggered:
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            for s in _SPEEDS:
                if trigger_id == _speed_id(s):
                    chosen = s
        classes = tuple(
            "fuselk-toolbar-btn active" if s == chosen else "fuselk-toolbar-btn" for s in _SPEEDS
        )
        interval_ms = max(50, int(1000 * _DT_S / chosen))
        return (*classes, interval_ms)

    @app.callback(
        Output("btn-play", "children"),
        Output("interval", "disabled"),
        Input("btn-play", "n_clicks"),
        Input("interval", "n_intervals"),
    )
    def toggle_play(n_clicks: int, _n_intervals: int) -> tuple[str, bool]:
        playing = (n_clicks or 0) % 2 == 1
        finished = _pulse_finished()
        label = "⏸ Pause" if (playing and not finished) else "▶ Start"
        return label, not playing or finished

    @app.callback(
        Output("preset-editor", "style"),
        Output("preset-editor", "children"),
        Input("btn-edit", "n_clicks"),
        State("preset-select", "value"),
    )
    def toggle_editor(n_clicks: int, preset: str | None) -> tuple[dict, str]:
        visible = (n_clicks or 0) % 2 == 1
        style = {
            "display": "block" if visible else "none",
            "width": "100%",
            "marginTop": "8px",
            "color": MUTED,
            "fontSize": "12px",
            "fontFamily": "monospace",
            "whiteSpace": "pre-wrap",
        }
        text = f"Preset target-waveform inspector — {preset or 'H-mode'} (read-only)"
        return style, text

    @app.callback(
        Output("scrub-store", "data"),
        Input("trace-panel", "relayoutData"),
        Input("btn-reset", "n_clicks"),
        Input("device-select", "value"),
        Input("preset-select", "value"),
    )
    def update_scrub(relayout: dict | None, *_reset_triggers) -> list[float] | None:
        from dash import callback_context

        ctx = callback_context
        trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""
        if not trigger.startswith("trace-panel"):
            return None  # any non-scrub trigger (reset/device/preset) clears the scrub window
        if relayout and "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
            return [relayout["xaxis.range[0]"], relayout["xaxis.range[1]"]]
        if relayout and "xaxis.autorange" in relayout:
            return None
        return None

    @app.callback(
        Output("control-room", "figure"),
        Output("kpi-strip", "figure"),
        Output("equilibrium", "figure"),
        Output("trace-panel", "figure"),
        Output("power-balance", "figure"),
        Output("stability", "figure"),
        Output("fusion-performance", "figure"),
        Output("pulse-clock", "children"),
        Output("stat-core", "children"),
        Output("stat-core2", "children"),
        Input("interval", "n_intervals"),
        Input("btn-reset", "n_clicks"),
        Input("device-select", "value"),
        Input("preset-select", "value"),
        Input("trace-select", "value"),
        Input("scrub-store", "data"),
        State("btn-play", "n_clicks"),
        prevent_initial_call=False,
    )
    def tick(
        n_intervals: int,
        reset_clicks: int,
        device: str | None,
        preset: str | None,
        trace_selection: list[str] | None,
        scrub_data: list[float] | None,
        play_clicks: int,
    ):
        from dash import callback_context

        ctx = callback_context
        trigger = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

        if device:
            _sim.set_device(device)
        if preset:
            _sim.set_preset(preset)

        playing = (play_clicks or 0) % 2 == 1
        if trigger.startswith("btn-reset"):
            frame = _sim.reset(seed=int(reset_clicks or 0))
            _record_history(frame, reset=True)
        elif trigger.startswith("device-select") or trigger.startswith("preset-select"):
            frame = _sim.step()
            _record_history(frame)
        elif trigger.startswith("interval") and playing and not _pulse_finished():
            frame = _sim.step()
            _record_history(frame)
        else:
            frame = _sim.last_frame or _sim.reset(seed=0)

        scrub_range = tuple(scrub_data) if scrub_data else None
        t_now = min(frame.step, _MAX_STEPS) * _DT_S
        clock = f"t = {t_now:.1f} / {_DURATION_S:.1f}s"

        core1 = (
            f"Ip {frame.ip_ma:.2f} MA · Bt {frame.active_device.max_bt_t:.1f} T · "
            f"Te0 {frame.te0_kev:.1f} keV"
        )
        core2 = f"n̄e {frame.ne_bar_1e19:.1f}e19 · Wth {frame.w_th_mj:.1f} MJ · τE {frame.tau_e_s:.2f} s"

        return (
            build_control_room_figure(frame),
            build_kpi_strip(frame),
            build_equilibrium_figure(frame),
            build_oscilloscope_figure(_history, trace_selection, scrub_range, dt_s=_DT_S),
            build_power_balance_figure(frame),
            build_stability_figure(frame),
            build_status_panel_figure(frame),
            clock,
            core1,
            core2,
        )


def _stat_card(title: str, elem_id: str, value: str) -> html.Div:
    return html.Div(
        style=STAT_CARD_STYLE,
        children=[
            html.H4(title, style=STAT_LABEL_STYLE),
            html.P(value, id=elem_id, style=STAT_VALUE_STYLE),
        ],
    )


