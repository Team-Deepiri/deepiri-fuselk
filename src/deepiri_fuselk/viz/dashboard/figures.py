"""Plotly figure builders for the fuselk control room."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from deepiri_fuselk.viz.dashboard.theme import (
    ACCENT,
    ACCENT_ALT,
    BG,
    CARD,
    DANGER,
    MUTED,
    OK,
    PLOT_THEME,
    TEXT,
    WARN,
)
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
                "bar": {"color": DANGER if elm_prob > 0.5 else OK},
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
        paper_bgcolor=CARD,
        plot_bgcolor=BG,
        **PLOT_THEME,
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
                title={"text": title, "font": {"size": 11, "color": MUTED}},
                number={"suffix": "%", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, mx], "tickwidth": 0},
                    "bar": {"color": ACCENT, "thickness": 0.35},
                    "bgcolor": CARD,
                    "borderwidth": 0,
                },
            ),
            row=1,
            col=i,
        )
    fig.update_layout(
        height=220,
        showlegend=False,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        **PLOT_THEME,
        margin={"t": 28, "b": 12, "l": 16, "r": 16},
    )
    return fig


def _risk_color(value: float, limit: float) -> str:
    frac = value / limit if limit else 0.0
    if frac >= 0.95:
        return DANGER
    if frac >= 0.75:
        return WARN
    return OK


def build_equilibrium_figure(frame: SimulationFrame) -> go.Figure:
    """2D poloidal cross-section: nested flux surfaces, separatrix, X-point, strike points.

    Uses a Miller-style analytic parameterization driven by the active
    device's major/minor radius, elongation, and triangularity so the shape
    changes when the device is switched (ITER vs JET vs DIII-D).
    """
    dev = frame.active_device
    r0, a, kappa, delta = (
        dev.major_radius_m,
        dev.minor_radius_m,
        dev.elongation,
        dev.triangularity,
    )
    theta = np.linspace(0, 2 * np.pi, 200)

    fig = go.Figure()
    n_surfaces = 6
    for i in range(1, n_surfaces + 1):
        frac = i / n_surfaces
        r = r0 + frac * a * np.cos(theta + delta * np.sin(theta))
        z = frac * a * kappa * np.sin(theta)
        is_sep = i == n_surfaces
        fig.add_trace(
            go.Scatter(
                x=r,
                y=z,
                mode="lines",
                line={
                    "color": DANGER if is_sep else ACCENT_ALT,
                    "width": 3 if is_sep else 1,
                },
                name="Separatrix" if is_sep else f"Flux surface {frac:.1f}",
                showlegend=is_sep,
            )
        )

    # X-point: lower-null geometry, placed below the separatrix minimum.
    x_point = (r0 - delta * a, -kappa * a * 1.05)
    fig.add_trace(
        go.Scatter(
            x=[x_point[0]],
            y=[x_point[1]],
            mode="markers",
            marker={"size": 14, "color": TEXT, "symbol": "x"},
            name="X-point",
        )
    )

    # Strike points on the divertor plates either side of the X-point.
    strike_points = [
        (x_point[0] - 0.35 * a, x_point[1] - 0.25 * kappa * a),
        (x_point[0] + 0.35 * a, x_point[1] - 0.25 * kappa * a),
    ]
    fig.add_trace(
        go.Scatter(
            x=[p[0] for p in strike_points],
            y=[p[1] for p in strike_points],
            mode="markers",
            marker={"size": 12, "color": WARN, "symbol": "diamond"},
            name="Strike points",
        )
    )

    # Plasma core color mapped to disruption risk.
    core_r = r0 + 0.15 * a * np.cos(theta)
    core_z = 0.15 * a * kappa * np.sin(theta)
    risk = frame.disruption.probability
    fig.add_trace(
        go.Scatter(
            x=core_r,
            y=core_z,
            mode="lines",
            fill="toself",
            line={"color": _risk_color(risk, 1.0), "width": 0},
            fillcolor=_risk_color(risk, 1.0),
            opacity=0.35,
            name="Core",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"{dev.name} poloidal cross-section",
        xaxis={"title": "R (m)", "scaleanchor": "y", "scaleratio": 1, "gridcolor": "#2a2d37"},
        yaxis={"title": "Z (m)", "gridcolor": "#2a2d37"},
        paper_bgcolor=CARD,
        plot_bgcolor=BG,
        height=420,
        **PLOT_THEME,
    )
    return fig


TRACE_FIELDS: dict[str, dict[str, str]] = {
    "Ip": {"actual": "ip_ma", "target": "target_ip_ma", "unit": "MA"},
    "betaN": {"actual": "beta_n", "target": "target_beta_n", "unit": ""},
    "li": {"actual": "li", "target": "target_li", "unit": ""},
    "Dalpha": {"actual": "d_alpha", "target": "target_d_alpha", "unit": "a.u."},
}


def build_trace_figure(
    history: list[dict],
    selected: list[str] | None = None,
    scrub_range: tuple[float, float] | None = None,
) -> go.Figure:
    """Synchronized multi-trace time panel: dashed target vs solid actual.

    `history` is a list of per-step dicts with at least `step` plus the
    actual/target keys named in TRACE_FIELDS. `scrub_range` optionally
    restricts the x-axis to a click-dragged (start, end) step window once
    the pulse has completed.
    """
    selected = selected or list(TRACE_FIELDS.keys())
    fig = go.Figure()
    steps = [h["step"] for h in history]
    colors = [ACCENT, ACCENT_ALT, OK, WARN]

    for i, key in enumerate(selected):
        spec = TRACE_FIELDS.get(key)
        if spec is None:
            continue
        color = colors[i % len(colors)]
        actual = [h.get(spec["actual"], 0.0) for h in history]
        target = [h.get(spec["target"], 0.0) for h in history]
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=actual,
                mode="lines",
                line={"color": color, "width": 2},
                name=f"{key} actual",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=target,
                mode="lines",
                line={"color": color, "width": 1.5, "dash": "dash"},
                name=f"{key} target",
            )
        )

    if scrub_range is not None:
        fig.update_xaxes(range=list(scrub_range))

    fig.update_layout(
        title="Discharge traces — dashed = target, solid = actual",
        xaxis={"title": "step", "gridcolor": "#2a2d37"},
        yaxis={"title": "value", "gridcolor": "#2a2d37"},
        paper_bgcolor=CARD,
        plot_bgcolor=BG,
        height=360,
        legend={"orientation": "h", "y": -0.2},
        dragmode="zoom",
        **PLOT_THEME,
    )
    return fig


def build_power_balance_figure(frame: SimulationFrame) -> go.Figure:
    """Colored input/output power balance bars (Poh/PNBI/PECH -> Prad/Ploss)."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["Poh", "PNBI", "PECH"],
            y=[frame.p_oh_mw, frame.p_nbi_mw, frame.p_ech_mw],
            marker_color=[ACCENT, ACCENT_ALT, OK],
            name="input",
        )
    )
    fig.add_trace(
        go.Bar(
            x=["Prad", "Ploss"],
            y=[frame.p_rad_mw, frame.p_loss_mw],
            marker_color=[WARN, DANGER],
            name="output",
        )
    )
    fig.update_layout(
        title="Power balance (MW)",
        barmode="group",
        paper_bgcolor=CARD,
        plot_bgcolor=BG,
        height=240,
        showlegend=False,
        **PLOT_THEME,
        margin={"t": 32, "b": 24, "l": 32, "r": 12},
    )
    return fig


def build_stability_figure(frame: SimulationFrame) -> go.Figure:
    """Stability & disruption-risk gauges: q95, betaN, fGW, disruption risk."""
    dev = frame.active_device
    risk = frame.disruption.probability
    specs_titles = ("q95", "betaN", "fGW", "Disruption risk")
    fig = make_subplots(rows=1, cols=4, specs=[[{"type": "indicator"}] * 4])

    gauges = [
        (frame.q95, 6.0, frame.q95 < 3.0),
        (frame.beta_n, dev.troyon_beta_limit, frame.beta_n >= dev.troyon_beta_limit * 0.95),
        (
            frame.greenwald_fraction,
            dev.greenwald_limit,
            frame.greenwald_fraction >= dev.greenwald_limit * 0.95,
        ),
        (risk * 100, 100, risk > 0.65),
    ]
    for i, ((val, mx, danger), title) in enumerate(zip(gauges, specs_titles, strict=True), start=1):
        bar_color = DANGER if danger else (WARN if val / mx > 0.75 else OK)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=val,
                title={"text": title, "font": {"size": 12, "color": MUTED}},
                gauge={
                    "axis": {"range": [0, mx]},
                    "bar": {"color": bar_color},
                    "bgcolor": CARD,
                    "steps": [
                        {"range": [0, mx * 0.75], "color": "#1a3d1a"},
                        {"range": [mx * 0.75, mx * 0.95], "color": "#4d4d1a"},
                        {"range": [mx * 0.95, mx], "color": "#4d1a1a"},
                    ],
                },
            ),
            row=1,
            col=i,
        )
    fig.update_layout(
        height=220,
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        showlegend=False,
        **PLOT_THEME,
        margin={"t": 28, "b": 12, "l": 16, "r": 16},
    )
    return fig


def build_status_panel_figure(frame: SimulationFrame) -> go.Figure:
    """Fusion performance mini-panel: neutron rate + Qplasma."""
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "indicator"}] * 2])
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=frame.neutron_rate_1e18,
            title={"text": "Neutron rate (1e18/s)", "font": {"size": 12, "color": MUTED}},
            number={"font": {"size": 22, "color": TEXT}},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=frame.q_plasma,
            title={"text": "Q_plasma", "font": {"size": 12, "color": MUTED}},
            number={"font": {"size": 22, "color": TEXT}, "suffix": "x"},
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        height=140,
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        showlegend=False,
        **PLOT_THEME,
        margin={"t": 24, "b": 8, "l": 16, "r": 16},
    )
    return fig
