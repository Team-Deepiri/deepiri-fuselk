"""Experiment catalog runner panel."""

from __future__ import annotations

import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from deepiri_fuselk.experiments.registry import load_registry
from deepiri_fuselk.experiments.runner import run_experiment


class ExperimentsPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries = load_registry()
        self._build_ui()
        self._populate_table()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["ID", "Status", "Category", "Description"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)

        btn_row = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run selected experiment")
        self._btn_refresh = QPushButton("Refresh list")
        self._btn_refresh.setObjectName("secondary")
        btn_row.addWidget(self._btn_run)
        btn_row.addWidget(self._btn_refresh)
        btn_row.addStretch()

        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setPlaceholderText("Experiment results (JSON)…")
        self._output.setMinimumHeight(140)

        root.addWidget(self._table, stretch=2)
        root.addLayout(btn_row)
        root.addWidget(self._output, stretch=1)

        self._btn_run.clicked.connect(self._run_selected)
        self._btn_refresh.clicked.connect(self._populate_table)

    def _populate_table(self) -> None:
        self._entries = load_registry()
        self._table.setRowCount(len(self._entries))
        for row, entry in enumerate(self._entries):
            for col, text in enumerate(
                [entry.id, entry.status, entry.category, entry.description]
            ):
                item = QTableWidgetItem(text)
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, entry.id)
                self._table.setItem(row, col, item)
        self._table.resizeColumnsToContents()

    def _run_selected(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            self._output.setPlainText("Select an experiment row first.")
            return
        exp_id = self._table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)
        self._btn_run.setEnabled(False)
        try:
            result = run_experiment(str(exp_id))
            self._output.setPlainText(json.dumps(result, indent=2))
        except Exception as exc:
            self._output.setPlainText(f"Error: {exc}")
        finally:
            self._btn_run.setEnabled(True)
