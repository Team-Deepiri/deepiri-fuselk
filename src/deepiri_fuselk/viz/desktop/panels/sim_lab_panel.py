"""Live simulation lab — device/preset selection, play/pause, step/reset, KPI readout."""

from __future__ import annotations

import json

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
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
from deepiri_fuselk.viz.simulation_engine import (
    LiveSimulation,
    get_preset_names,
    list_device_names,
)


class SimLabPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sim = LiveSimulation(grid_size=24, device="ITER", preset="H-mode")
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
        self._device = QComboBox()
        self._device.addItems(list_device_names())
        self._device.setCurrentText("ITER")
        self._preset = QComboBox()
        self._refresh_presets()
        form.addRow("Grid size:", self._grid)
        form.addRow("Seed:", self._seed)
        form.addRow("Device:", self._device)
        form.addRow("Preset:", self._preset)

        btn_row = QHBoxLayout()
        self._btn_step = QPushButton("▶  Step")
        self._btn_step.setObjectName("accent-green")
        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setObjectName("secondary")
        self._btn_play = QPushButton("▶ Play")
        self._speed = QDoubleSpinBox()
        self._speed.setRange(0.5, 2.0)
        self._speed.setSingleStep(0.5)
        self._speed.setValue(1.0)
        self._btn_fusion = QPushButton("Run FusionCell batch")
        self._fusion_steps = QSpinBox()
        self._fusion_steps.setRange(5, 500)
        self._fusion_steps.setValue(50)
        btn_row.addWidget(self._btn_step)
        btn_row.addWidget(self._btn_reset)
        btn_row.addWidget(self._btn_play)
        btn_row.addWidget(QLabel("Speed:"))
        btn_row.addWidget(self._speed)
        btn_row.addStretch()
        btn_row.addWidget(QLabel("Batch steps:"))
        btn_row.addWidget(self._fusion_steps)
        btn_row.addWidget(self._btn_fusion)

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._step)

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
        self._btn_play.clicked.connect(self._toggle_play)
        self._speed.valueChanged.connect(self._apply_speed)
        self._device.currentTextChanged.connect(self._on_device_changed)
        self._preset.currentTextChanged.connect(self._on_preset_changed)

    def _refresh_presets(self) -> None:
        device = self._device.currentText() or "ITER"
        self._preset.blockSignals(True)
        self._preset.clear()
        self._preset.addItems(get_preset_names(device))
        self._preset.blockSignals(False)

    def _on_device_changed(self, name: str) -> None:
        if not name:
            return
        self._sim.set_device(name)
        self._refresh_presets()
        self._step()

    def _on_preset_changed(self, name: str) -> None:
        if not name:
            return
        self._sim.set_preset(name)
        self._step()

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self._btn_play.setText("▶ Play")
        else:
            self._apply_speed()
            self._play_timer.start()
            self._btn_play.setText("⏸ Pause")

    def _apply_speed(self) -> None:
        base_ms = 1000
        interval = max(200, int(base_ms / max(0.5, self._speed.value())))
        self._play_timer.setInterval(interval)

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
        device = self._device.currentText() or "ITER"
        preset = self._preset.currentText() or "H-mode"
        if grid != self._sim.grid_size:
            self._sim = LiveSimulation(grid_size=grid, device=device, preset=preset)
        else:
            self._sim.set_device(device)
            self._sim.set_preset(preset)
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
