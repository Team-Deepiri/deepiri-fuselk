"""In-app tutorial / onboarding walkthrough for the fuselk control room.

A short, dismissible sequence of pages that orients a new user: what a
tokamak is, how to read each panel (equilibrium, trace, status, port view),
and how to drive a pulse. Launched on demand from Help -> Tutorial; not
tied to any first-run mechanism (none exists in the shell today).
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class TutorialPage:
    title: str
    body: str


TUTORIAL_PAGES: list[TutorialPage] = [
    TutorialPage(
        "Welcome to the control room",
        "A tokamak confines hot plasma inside a doughnut-shaped vessel using "
        "magnetic fields — a strong toroidal field running the long way around, "
        "combined with a weaker poloidal field, together weave helical field "
        "lines that hold the plasma away from the walls.\n\n"
        "A “pulse” is one discharge: you program a waveform for heating "
        "power, plasma current, density, and shape, then watch the plasma "
        "respond in real time. Your job as operator is to keep the discharge "
        "stable, avoid disruptions, and get the best fusion performance you "
        "can out of the device.",
    ),
    TutorialPage(
        "Equilibrium panel",
        "This is the poloidal cross-section — a slice through the vessel. The "
        "nested contour lines are flux surfaces: surfaces of constant magnetic "
        "flux, hottest and densest at the core and cooler toward the edge.\n\n"
        "The red outline is the separatrix, the boundary between confined "
        "plasma and the open field lines that carry heat to the divertor. "
        "Where the poloidal field goes to zero on that boundary is the "
        "X-point. Plasma shape is described by elongation (κ, how tall vs. "
        "wide) and triangularity (δ, how “D”-shaped). Strike points are "
        "where the separatrix meets the divertor plates — the small spots "
        "that take a concentrated heat load.",
    ),
    TutorialPage(
        "Trace panel",
        "The trace panel is an oscilloscope view of key signals over time — "
        "by default Ip (plasma current), βN (normalized pressure), li "
        "(internal inductance), and Dα (edge recycling light). Use the "
        "dropdown to add or remove traces.\n\n"
        "Dashed lines are the programmed target waveform; solid lines are the "
        "plasma's actual response. Once a pulse finishes, click-drag along "
        "the trace to scrub through time and replay the equilibrium and "
        "diagnostics at any point in the run.",
    ),
    TutorialPage(
        "Status panel",
        "Four groups at a glance. Core Parameters: Ip, Bt, Te0, n̄e, Wth, "
        "τE. Power Balance: heating in (Poh / PNBI / PECH) versus losses out "
        "(Prad, Ploss), shown as colored bars.\n\n"
        "Stability & Disruption Risk: q95, βN, fGW (Greenwald fraction), and "
        "a red disruption-risk gauge that fills up as you approach the "
        "limits. Fusion Performance: neutron rate and Qplasma — Q = 1 is "
        "scientific breakeven, Q = 10 is the ITER design target.",
    ),
    TutorialPage(
        "Port view",
        "A 3D view looking through a diagnostic port into the vessel. The "
        "glowing plasma column's color and intensity track temperature and "
        "density; the divertor plates are colored by heat flux from blue "
        "(cool) through cyan and yellow up to red/white (hottest).\n\n"
        "Real tokamaks build this same picture from visible-light and "
        "infrared cameras aimed through actual diagnostic ports — this "
        "panel renders the equivalent from the live simulation.",
    ),
    TutorialPage(
        "Mission briefing",
        "Your goals: don't disrupt, avoid ELMs (edge-localized modes that "
        "erode the tungsten divertor over repeated pulses), get enough "
        "fusion power to hit a good Q, and protect the divertor — typically "
        "via detachment or seeded impurities that spread the heat load.\n\n"
        "Strategy tip: smaller, lower-field devices are lower-consequence to "
        "experiment on. Start on DIII-D or JET, learn how the controls "
        "behave, then work up to ITER-scale discharges.",
    ),
    TutorialPage(
        "Quick controls recap",
        "Start / Pause runs and halts the pulse; the speed buttons change how "
        "fast simulated time advances. The device dropdown switches between "
        "ITER, JET, and DIII-D; preset buttons load H-mode, L-mode, or "
        "Density Limit starting points.\n\n"
        "Edit / shot-planner lets you shape the programmed waveform before "
        "firing. After a pulse completes, click-drag on the trace panel to "
        "scrub back through the discharge. That's the whole loop — you're "
        "ready to run a shot.",
    ),
]


class TutorialDialog(QDialog):
    """Dismissible, paged walkthrough of the control room."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("fuselk — Control Room Tutorial")
        self.setModal(True)
        self.resize(560, 420)
        self._pages = TUTORIAL_PAGES
        self._index = 0
        self._build_ui()
        self._render_page()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 16)
        root.setSpacing(12)

        self._step_label = QLabel()
        self._step_label.setStyleSheet("color: #8a94a6; font-size: 11px;")
        root.addWidget(self._step_label)

        self._title_label = QLabel()
        self._title_label.setStyleSheet("font-size: 18px; font-weight: 700;")
        root.addWidget(self._title_label)

        self._body_label = QLabel()
        self._body_label.setWordWrap(True)
        self._body_label.setTextFormat(Qt.TextFormat.PlainText)
        self._body_label.setStyleSheet("font-size: 13px; line-height: 1.4;")
        self._body_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        root.addWidget(self._body_label, stretch=1)

        btn_row = QHBoxLayout()
        self._skip_btn = QPushButton("Skip")
        self._skip_btn.clicked.connect(self.reject)
        btn_row.addWidget(self._skip_btn)
        btn_row.addStretch()

        self._back_btn = QPushButton("Back")
        self._back_btn.clicked.connect(self._go_back)
        btn_row.addWidget(self._back_btn)

        self._next_btn = QPushButton("Next")
        self._next_btn.setDefault(True)
        self._next_btn.clicked.connect(self._go_next)
        btn_row.addWidget(self._next_btn)

        root.addLayout(btn_row)

    def _render_page(self) -> None:
        page = self._pages[self._index]
        self._step_label.setText(f"Step {self._index + 1} of {len(self._pages)}")
        self._title_label.setText(page.title)
        self._body_label.setText(page.body)
        self._back_btn.setEnabled(self._index > 0)
        is_last = self._index == len(self._pages) - 1
        self._next_btn.setText("Done" if is_last else "Next")

    def _go_next(self) -> None:
        if self._index >= len(self._pages) - 1:
            self.accept()
            return
        self._index += 1
        self._render_page()

    def _go_back(self) -> None:
        if self._index > 0:
            self._index -= 1
            self._render_page()
