"""Live simulation lab — step/reset and KPI readout."""

from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from deepiri_fuselk.viz.api import frame_to_dict
from deepiri_fuselk.viz.desktop.widgets import KpiCard
from deepiri_fuselk.viz.simulation_engine import LiveSimulation


class SimLabPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sim = LiveSimulation(grid_size=24)
        self._cards: dict[str, KpiCard] = {}
        self._build_ui()
        self._reset()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        controls = QGroupBox("Simulation controls")
        form = QFormLayout(controls)
        self._grid = QSpinBox()
        self._grid.setRange(8, 64)
        self._grid.setValue(24)
        self._seed = QSpinBox()
        self._seed.setRange(0, 9999)
        form.addRow("Grid size:", self._grid)
        form.addRow("Seed:", self._seed)

        btn_row = QHBoxLayout()
        self._btn_step = QPushButton("▶  Step")
        self._btn_step.setObjectName("accent-green")
        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setObjectName("secondary")
        self._btn_fusion = QPushButton("Run FusionCell batch")
        self._fusion_steps = QSpinBox()
        self._fusion_steps.setRange(5, 500)
        self._fusion_steps.setValue(50)
        btn_row.addWidget(self._btn_step)
        btn_row.addWidget(self._btn_reset)
        btn_row.addStretch()
        btn_row.addWidget(QLabel("Batch steps:"))
        btn_row.addWidget(self._fusion_steps)
        btn_row.addWidget(self._btn_fusion)

        kpi_grid = QGridLayout()
        kpi_grid.setSpacing(10)
        specs = [
            ("step", "Step"),
            ("fusion", "Fusion Score"),
            ("disruption", "Disruption"),
            ("tbr", "TBR"),
            ("muon", "μ Fusions"),
            ("snr", "HELIX SNR"),
            ("action", "Venturi Action"),
            ("peclet", "Peclet #"),
        ]
        for i, (key, title) in enumerate(specs):
            card = KpiCard(title)
            self._cards[key] = card
            kpi_grid.addWidget(card, i // 4, i % 4)

        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setPlaceholderText("FusionCell batch report JSON…")
        self._output.setMinimumHeight(160)

        root.addWidget(controls)
        root.addLayout(btn_row)
        root.addLayout(kpi_grid)
        root.addWidget(QLabel("Batch output"))
        root.addWidget(self._output, stretch=1)

        self._btn_step.clicked.connect(self._step)
        self._btn_reset.clicked.connect(self._reset)
        self._btn_fusion.clicked.connect(self._run_fusion)

    def _show_frame(self, frame_dict: dict) -> None:
        risk = frame_dict["disruption_probability"]
        self._cards["step"].set_value(str(frame_dict["step"]), f"seed {frame_dict['seed']}")
        self._cards["fusion"].set_value(f"{frame_dict['fusion_score']:.1%}", "composite KPI")
        self._cards["disruption"].set_value(f"{risk:.1%}", "ELM + disruption", alert=risk > 0.5)
        self._cards["tbr"].set_value(f"{frame_dict['tbr']:.3f}", "breeding ratio")
        self._cards["muon"].set_value(f"{frame_dict['muon_fpm']:.0f}", "fusions / muon")
        self._cards["snr"].set_value(
            f"{frame_dict['helix']['phase_locked_snr']:.1f}x",
            f"O-point {tuple(frame_dict['helix']['o_point'])}",
        )
        self._cards["action"].set_value(frame_dict["action"], "controller output")
        self._cards["peclet"].set_value(
            f"{frame_dict['peclet']:.2f}",
            f"ELM-free {frame_dict['elm_free_fraction']:.0%}",
        )

    def _reset(self) -> None:
        grid = self._grid.value()
        if grid != self._sim.grid_size:
            self._sim = LiveSimulation(grid_size=grid)
        frame = self._sim.reset(seed=self._seed.value())
        self._show_frame(frame_to_dict(frame))

    def _step(self) -> None:
        frame = self._sim.step()
        self._show_frame(frame_to_dict(frame))

    def _run_fusion(self) -> None:
        from deepiri_fuselk.sim.fusion_cell import FusionCell

        self._btn_fusion.setEnabled(False)
        try:
            _, report = FusionCell(grid_size=self._grid.value(), train_elm=False).run(
                n_steps=self._fusion_steps.value(), seed=self._seed.value()
            )
            self._output.setPlainText(json.dumps(report.to_dict(), indent=2))
        finally:
            self._btn_fusion.setEnabled(True)
