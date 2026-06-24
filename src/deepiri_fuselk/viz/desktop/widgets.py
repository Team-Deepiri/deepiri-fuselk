"""Reusable Qt widgets for the fuselk desktop shell."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QProgressBar,
    QSplashScreen,
    QVBoxLayout,
    QWidget,
)

from deepiri_fuselk import __version__


class SplashScreen(QSplashScreen):
    def __init__(self) -> None:
        super().__init__(_splash_pixmap())
        self.setStyleSheet(
            "QSplashScreen { color: #c8d8ff; font-size: 13px; font-family: system-ui; }"
        )
        self.showMessage(
            "Starting fusion backends…",
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            Qt.GlobalColor.white,
        )


def _splash_pixmap():
    from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPixmap

    w, h = 520, 300
    px = QPixmap(w, h)
    px.fill(QColor("#0a0c12"))
    painter = QPainter(px)
    grad = QLinearGradient(0, 0, w, h)
    grad.setColorAt(0.0, QColor("#101828"))
    grad.setColorAt(1.0, QColor("#0a0c12"))
    painter.fillRect(0, 0, w, h, grad)
    painter.setPen(QColor("#4488ff"))
    painter.setFont(QFont("system-ui", 28, QFont.Weight.Bold))
    painter.drawText(40, 90, "deepiri-fuselk")
    painter.setPen(QColor("#8899bb"))
    painter.setFont(QFont("system-ui", 11))
    painter.drawText(42, 118, "Fusion Unified Simulation, ELM Learning & Kinetics")
    painter.setPen(QColor("#556677"))
    painter.drawText(42, h - 48, f"v{__version__}  ·  loading Dash + API + Qt shell")
    painter.end()
    return px


class StatusDot(QLabel):
    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = label
        self.set_status(False, "offline")

    def set_status(self, ok: bool, detail: str = "") -> None:
        color = "#3dd68c" if ok else "#ff6b6b"
        text = detail or ("online" if ok else "offline")
        self.setText(f'<span style="color:{color}">●</span> {self._label}: {text}')
        self.setTextFormat(Qt.TextFormat.RichText)


class KpiCard(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("kpi-card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        self._title = QLabel(title)
        self._title.setObjectName("kpi-title")
        self._value = QLabel("—")
        self._value.setObjectName("kpi-value")
        self._hint = QLabel("")
        self._hint.setObjectName("kpi-hint")
        layout.addWidget(self._title)
        layout.addWidget(self._value)
        layout.addWidget(self._hint)

    def set_value(self, value: str, hint: str = "", *, alert: bool = False) -> None:
        self._value.setText(value)
        self._hint.setText(hint)
        color = "#ff6b6b" if alert else "#e8e8e8"
        self._value.setStyleSheet(f"color: {color};")


class PanelHeader(QWidget):
    def __init__(self, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("panel-header")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 8)
        layout.setSpacing(4)
        t = QLabel(title)
        t.setObjectName("panel-title")
        s = QLabel(subtitle)
        s.setObjectName("panel-subtitle")
        s.setWordWrap(True)
        layout.addWidget(t)
        layout.addWidget(s)


class PanelChrome(QWidget):
    """Native panel wrapper with title header and padded content area."""

    def __init__(
        self,
        title: str,
        subtitle: str,
        content: QWidget,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(PanelHeader(title, subtitle))
        body = QFrame()
        body.setObjectName("panel-body")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(16, 8, 16, 16)
        body_layout.addWidget(content)
        root.addWidget(body, stretch=1)


class LoadingOverlay(QWidget):
    def __init__(self, message: str = "Loading…", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("loading-overlay")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label = QLabel(message)
        label.setObjectName("loading-label")
        bar = QProgressBar()
        bar.setRange(0, 0)
        bar.setFixedWidth(220)
        bar.setObjectName("loading-bar")
        layout.addWidget(label)
        layout.addWidget(bar, alignment=Qt.AlignmentFlag.AlignHCenter)
