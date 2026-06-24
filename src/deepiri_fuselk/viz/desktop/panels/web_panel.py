"""QWebEngineView panel with loading overlay."""

from __future__ import annotations

from PySide6.QtCore import QUrl
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QStackedLayout, QWidget

from deepiri_fuselk.viz.desktop.widgets import LoadingOverlay


class WebPanel(QWidget):
    """Embeds a web view pointed at a URL or local file."""

    def __init__(self, url: str, *, loading_message: str = "Loading view…", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._url = url
        self._view = QWebEngineView(self)
        self._overlay = LoadingOverlay(loading_message, self)
        self._stack = QStackedLayout(self)
        self._stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        self._stack.addWidget(self._view)
        self._stack.addWidget(self._overlay)
        self._overlay.raise_()
        self._view.loadFinished.connect(self._on_loaded)
        self.reload()

    def _on_loaded(self, ok: bool) -> None:
        self._overlay.setVisible(not ok)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._overlay.setGeometry(self.rect())

    def reload(self) -> None:
        self._overlay.setVisible(True)
        if self._url.startswith(("http://", "https://", "file://")):
            self._view.setUrl(QUrl(self._url))
        else:
            self._view.setUrl(QUrl.fromLocalFile(self._url))

    def page(self) -> QWebEnginePage:
        return self._view.page()
