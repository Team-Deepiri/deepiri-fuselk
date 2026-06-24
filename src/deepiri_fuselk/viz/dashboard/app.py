"""Rich multi-panel fuselk control room dashboard."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Dash, dcc, html
from deepiri_fuselk.helix.helix_engine import HelixEngine
from deepiri_fuselk.models.elm_predictor import ELMPredictor
from deepiri_fuselk.sim.synthetic_data_gen import generate_ece_shot
from deepiri_fuselk.viz.traffic_viewer import traffic_arrows
from plotly.subplots import make_subplots


def _make_control_room_figure(seed: int = 0):
    shot = generate_ece_shot(32, seed=seed)
    engine = HelixEngine()
    result = engine.process(shot.heat_field, shot.raw_signal, shot.angles)
    predictor = ELMPredictor()
    elm = predictor.predict(result.focal_map, result.rotation_hz)
    arrows = traffic_arrows(shot.heat_field)

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Raw ECE (noisy)",
            "HELIX Focal Map",
            "HQRM O-Point Lock",
            "ELM Probability",
            "Divertor Traffic",
            "Fracture Vector",
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "scatter"}],
            [{"type": "indicator"}, {"type": "scatter"}, {"type": "scatter"}],
        ],
    )

    fig.add_trace(go.Heatmap(z=shot.heat_field, colorscale="Hot", showscale=False), row=1, col=1)
    fig.add_trace(
        go.Heatmap(z=result.focal_map, colorscale="Viridis", showscale=False), row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=[result.o_point[0]],
            y=[result.o_point[1]],
            mode="markers",
            marker={"size": 15, "color": "red", "symbol": "x"},
            name="O-point",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=elm.probability * 100,
            title={"text": "ELM %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red" if elm.probability > 0.5 else "green"},
            },
        ),
        row=2,
        col=1,
    )
    if arrows:
        fig.add_trace(
            go.Scatter(
                x=[a["x"] for a in arrows],
                y=[a["y"] for a in arrows],
                mode="markers",
                marker={
                    "size": 8,
                    "color": [a["magnitude"] for a in arrows],
                    "colorscale": "YlOrRd",
                },
                name="traffic",
            ),
            row=2,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=[0, result.fracture_vector[0]],
            y=[0, result.fracture_vector[1]],
            mode="lines+markers",
            line={"color": "orange", "width": 3},
            name="fracture",
        ),
        row=2,
        col=3,
    )
    fig.update_layout(height=700, showlegend=False, title_text="fuselk Control Room")
    return fig, elm, result


def create_app() -> Dash:
    """Create the full fuselk control room dashboard."""
    fig, elm, result = _make_control_room_figure()

    app = Dash(__name__)
    app.layout = html.Div(
        style={"fontFamily": "sans-serif", "padding": "20px"},
        children=[
            html.H1("deepiri-fuselk Control Room"),
            html.P(
                "HELIX focal diagnostics · Venturi control · ELM prediction · Plasma traffic routing",
                style={"color": "#666"},
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr 1fr",
                    "gap": "10px",
                    "marginBottom": "20px",
                },
                children=[
                    html.Div(
                        [
                            html.H4("O-Point"),
                            html.P(f"({result.o_point[0]:.3f}, {result.o_point[1]:.3f})"),
                        ]
                    ),
                    html.Div([html.H4("SNR Gain"), html.P(f"{result.phase_locked_snr:.1f}x")]),
                    html.Div([html.H4("ELM Mode"), html.P(elm.precursor_mode)]),
                ],
            ),
            dcc.Graph(figure=fig, id="control-room"),
            dcc.Interval(id="interval", interval=5000, n_intervals=0),
            html.Footer(
                "deepiri-fuselk — Fusion Unified Simulation, ELM Learning & Kinetics",
                style={"marginTop": "20px", "color": "#999", "fontSize": "12px"},
            ),
        ],
    )
    return app
