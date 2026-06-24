"""Oil-water PDE and muon cycle physics panels."""

from __future__ import annotations

import json

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class OilWaterPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.addWidget(QLabel("Oil-water vapor barrier PDE solver"))

        box = QGroupBox("Parameters")
        form = QFormLayout(box)
        self._mode = QComboBox()
        self._mode.addItems(["steady", "transient", "both"])
        self._grid = QSpinBox()
        self._grid.setRange(16, 128)
        self._grid.setValue(32)
        form.addRow("Mode:", self._mode)
        form.addRow("Grid:", self._grid)

        self._btn = QPushButton("Run PDE solve")
        self._out = QPlainTextEdit()
        self._out.setReadOnly(True)

        root.addWidget(box)
        root.addWidget(self._btn)
        root.addWidget(self._out, stretch=1)
        self._btn.clicked.connect(self._run)

    def _run(self) -> None:
        from deepiri_fuselk.physics.pde_solver import (
            solve_oil_water_steady,
            solve_oil_water_transient,
        )

        mode = self._mode.currentText()
        n = self._grid.value()
        out: dict = {"mode": mode}
        self._btn.setEnabled(False)
        try:
            if mode in ("steady", "both"):
                r = solve_oil_water_steady(n_grid=n)
                out["steady"] = {
                    "converged": r.converged,
                    "residual": r.residual,
                    "iterations": r.iterations,
                }
            if mode in ("transient", "both"):
                hist = solve_oil_water_transient(n_grid=min(n, 64), t_end=1.0)
                out["transient"] = {
                    "steps": len(hist),
                    "final_n_T_wall": float(hist[-1].n_T[-1]),
                }
            self._out.setPlainText(json.dumps(out, indent=2))
        finally:
            self._btn.setEnabled(True)


class MuonPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.addWidget(QLabel("Muon rate network — photon/proton stripping trifecta"))

        self._btn = QPushButton("Run muon rate network")
        self._out = QPlainTextEdit()
        self._out.setReadOnly(True)

        root.addWidget(self._btn)
        root.addWidget(self._out, stretch=1)
        self._btn.clicked.connect(self._run)

    def _run(self) -> None:
        from deepiri_fuselk.muon import RateNetworkParams, run_rate_network

        self._btn.setEnabled(False)
        try:
            r = run_rate_network(params=RateNetworkParams(R_photon=0.5, R_proton=0.3))
            self._out.setPlainText(
                json.dumps(
                    {
                        "fusions_per_muon": r.fusions_per_muon,
                        "effective_sticking": r.effective_sticking,
                        "breakeven": r.breakeven,
                    },
                    indent=2,
                )
            )
        finally:
            self._btn.setEnabled(True)
