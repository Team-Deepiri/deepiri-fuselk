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
        "This is the poloidal cross-section — a slice through the vessel, "
        "updated live as the pulse runs. The nested contour lines are flux "
        "surfaces: surfaces of constant poloidal magnetic flux, one inside "
        "the next like an onion, hottest and densest at the core and "
        "progressively cooler toward the edge.\n\n"
        "The red outline is the separatrix — the last closed flux surface, "
        "and the boundary between confined plasma and the open field lines "
        "that lead straight to the divertor. Somewhere on that boundary the "
        "poloidal field cancels to zero; that null point is the X-point, and "
        "it's what makes the separatrix a separatrix instead of just another "
        "flux surface. Where the open field lines actually land on the "
        "divertor plates are the strike points — narrow, intensely loaded "
        "spots that carry almost all of the exhaust heat leaving the "
        "plasma.\n\n"
        "Plasma shape is read off the same picture: elongation (κ) is how "
        "tall the cross-section is relative to its width, and triangularity "
        "(δ) is how strongly it's pulled into a “D” shape. Both are shaping "
        "knobs — pushing them further from a simple circle generally buys "
        "better confinement and stability margin, at the cost of a harder "
        "shape to control.",
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
        "Four groups, read top to bottom. Core Parameters is the plasma's "
        "vital signs: plasma current Ip, toroidal field Bt, central "
        "electron temperature Te0, line-averaged density n̄e, stored "
        "thermal energy Wth, and energy confinement time τE — how long that "
        "energy would take to leak away with no more heating.\n\n"
        "Power Balance is the energy ledger: heating power in from ohmic "
        "current (Poh), neutral beams (PNBI), and electron-cyclotron "
        "heating (PECH), set against what leaves as radiation (Prad) and "
        "conducted/convected loss (Ploss). The colored bars make imbalance "
        "obvious — losses creeping up on input heating is an early warning "
        "sign.\n\n"
        "Stability & Disruption Risk tracks the safety factor q95, "
        "normalized beta βN, and Greenwald density fraction fGW against "
        "their operational limits, rolled up into a single red "
        "disruption-risk gauge that fills as any of them gets close to the "
        "edge. Fusion Performance closes the loop: neutron rate (a direct "
        "measure of fusion reactions happening right now) and Qplasma, the "
        "ratio of fusion power out to heating power in — Q = 1 is "
        "scientific breakeven, and Q = 10 is the performance ITER is "
        "designed to demonstrate.",
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
        "Four goals, in tension with each other: don't disrupt — a sudden "
        "loss of confinement dumps the plasma's current and energy into "
        "the vessel in milliseconds, and is the single worst thing that can "
        "happen to a pulse. Avoid ELMs — edge-localized modes, repeated "
        "bursts that eject heat and particles from the plasma edge; "
        "individually survivable, but they erode the tungsten divertor "
        "over many pulses if left unchecked. Achieve sufficient fusion "
        "power — push βN and density high enough to get a Q worth reporting, "
        "without wandering into the disruption or Greenwald limits doing "
        "it. And protect the divertor — the heat and particle exhaust has "
        "to go somewhere; detachment (cooling the plasma near the strike "
        "points) and seeded impurities (radiating power away volumetrically "
        "before it concentrates on the plates) are how real devices spread "
        "that load out.\n\n"
        "Strategy tip: these goals get harder together as you scale up. "
        "Smaller, lower-field devices like DIII-D are lower-consequence to "
        "push toward their limits — a disruption there teaches you the same "
        "lesson at a fraction of the stored energy. Learn how the controls "
        "behave on DIII-D or JET, then carry that intuition up to "
        "ITER-scale discharges, where the same mistakes are far more "
        "expensive.",
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
