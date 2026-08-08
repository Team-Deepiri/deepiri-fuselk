"""Central design tokens for the fuselk Dash control room.

Keeps layout colors/typography in one place instead of scattering hex values
throughout the callback/layout code, so theming is easy to extend later.
"""

from __future__ import annotations

ACCENT = "#4488ff"
ACCENT_ALT = "#6f9fff"
OK = "#37d67a"
WARN = "#f0c061"
DANGER = "#ff5c6c"
TEXT = "#e8e8e8"
MUTED = "#888888"
FAINT = "#555555"
BG = "#0f1117"
CARD = "#1a1d27"
BORDER = "#2a2d37"

FONT_STACK = "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"

STAT_CARD_STYLE = {
    "background": CARD,
    "borderRadius": "10px",
    "padding": "12px 16px",
    "border": f"1px solid {BORDER}",
}

STAT_LABEL_STYLE = {
    "margin": "0 0 4px 0",
    "fontSize": "12px",
    "letterSpacing": "0.04em",
    "textTransform": "uppercase",
    "color": MUTED,
}

STAT_VALUE_STYLE = {
    "margin": 0,
    "fontSize": "18px",
    "fontWeight": 600,
    "color": TEXT,
}

PLOT_THEME = {
    "paper_bgcolor": CARD,
    "plot_bgcolor": BG,
    "font": {"color": TEXT, "family": FONT_STACK},
    "title_font_color": TEXT,
}