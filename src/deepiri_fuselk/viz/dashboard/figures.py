"""Plotly figure builders for the fuselk control room."""

from __future__ import annotations

import plotly.graph_objects as go
from deepiri_fuselk.viz.simulation_engine import SimulationFrame
from deepiri_fuselk.viz.traffic_viewer import traffic_arrows
from plotly.subplots import make_subplots


def build_control_room_figure(frame: SimulationFrame) -> go.Figure:
    """Build the full 2×3 control room figure from a simulation frame."""
    arrows = traffic_arrows(frame.controlled_heat)
    elm_prob = frame.disruption.probability

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Raw ECE (noisy)",
            "HELIX Focal Map",
            "HQRM O-Point Lock",
            "Disruption Risk",
            "Divertor Traffic (post-Venturi)",
            "Fracture Vector",
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "scatter"}],
            [{"type": "indicator"}, {"type": "scatter"}, {"type": "scatter"}],
        ],
    )

    fig.add_trace(
        go.Heatmap(z=frame.raw_heat, colorscale="Hot", showscale=False, name="raw"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(z=frame.helix.focal_map, colorscale="Viridis", showscale=False, name="focal"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[frame.helix.o_point[0]],
            y=[frame.helix.o_point[1]],
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
            value=elm_prob * 100,
            title={"text": "Disruption %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red" if elm_prob > 0.5 else "green"},
                "steps": [
                    {"range": [0, 45], "color": "#1a3d1a"},
                    {"range": [45, 65], "color": "#4d4d1a"},
                    {"range": [65, 100], "color": "#4d1a1a"},
                ],
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
                    "showscale": False,
                },
                name="traffic",
            ),
            row=2,
            col=2,
        )
    fig.add_trace(
        go.Scatter(
            x=[0, frame.helix.fracture_vector[0]],
            y=[0, frame.helix.fracture_vector[1]],
            mode="lines+markers",
            line={"color": "orange", "width": 3},
            name="fracture",
        ),
        row=2,
        col=3,
    )
    fig.update_layout(
        height=720,
        showlegend=False,
        title_text=f"fuselk Live Simulation — step {frame.step} · action: {frame.action}",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1a1d27",
        font={"color": "#e0e0e0"},
    )
    return fig


def build_kpi_strip(frame: SimulationFrame) -> go.Figure:
    """Horizontal KPI indicators: fusion score, TBR, muon, ELM-free, divertor uniformity."""
    # Titles live on each Indicator (not subplot_titles) to avoid bleed in tight columns.
    titles = ("Fusion", "TBR", "μ fus", "ELM-free", "Div U")
    fig = make_subplots(
        rows=1,
        cols=5,
        specs=[[{"type": "indicator"}] * 5],
        horizontal_spacing=0.08,
    )
    metrics = [
        (frame.fusion_score * 100, 100),
        (min(frame.tbr, 1.5) / 1.5 * 100, 100),
        (min(frame.muon_fpm, 350) / 350 * 100, 100),
        (frame.elm_free_fraction * 100, 100),
        (frame.divertor_uniformity * 100, 100),
    ]
    for i, ((val, mx), title) in enumerate(zip(metrics, titles, strict=True), start=1):
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=val,
                title={"text": title, "font": {"size": 11, "color": "#9aa8c4"}},
                number={"suffix": "%", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, mx], "tickwidth": 0},
                    "bar": {"color": "#4488ff", "thickness": 0.35},
                    "bgcolor": "#1a1d27",
                    "borderwidth": 0,
                },
            ),
            row=1,
            col=i,
        )
    fig.update_layout(
        height=220,
        showlegend=False,
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font={"color": "#e0e0e0"},
        margin={"t": 28, "b": 12, "l": 16, "r": 16},
    )
    return fig
