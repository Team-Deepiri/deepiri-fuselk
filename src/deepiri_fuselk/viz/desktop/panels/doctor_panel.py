"""System doctor / dependency health panel."""

from __future__ import annotations

import importlib

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_CORE_MODULES = [
    "numpy",
    "scipy",
    "xarray",
    "pydantic",
    "zmq",
    "pyarrow",
    "gymnasium",
    "stable_baselines3",
    "dash",
    "plotly",
    "fastapi",
    "uvicorn",
]

_DESKTOP_MODULES = [
    "PySide6",
    "PySide6.QtWebEngineWidgets",
]


class DoctorPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.addWidget(QLabel("fuselk doctor — Python module health check"))

        btn_row = QHBoxLayout()
        self._btn = QPushButton("Re-run checks")
        self._summary = QLabel()
        btn_row.addWidget(self._btn)
        btn_row.addWidget(self._summary)
        btn_row.addStretch()

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Module", "Status"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        root.addLayout(btn_row)
        root.addWidget(self._table)
        self._btn.clicked.connect(self.refresh)

    def refresh(self) -> None:
        rows: list[tuple[str, str]] = []
        ok = True
        for name in _CORE_MODULES + _DESKTOP_MODULES:
            try:
                importlib.import_module(name)
                rows.append((name, "ok"))
            except ImportError as exc:
                ok = False
                rows.append((name, f"missing: {exc}"))

        self._table.setRowCount(len(rows))
        for i, (mod, status) in enumerate(rows):
            self._table.setItem(i, 0, QTableWidgetItem(mod))
            self._table.setItem(i, 1, QTableWidgetItem(status))
        self._table.resizeColumnsToContents()
        color = "#4caf50" if ok else "#ff9800"
        self._summary.setText(f"Overall: {'PASS' if ok else 'ISSUES'}")
        self._summary.setStyleSheet(f"color: {color}; font-weight: 600;")
