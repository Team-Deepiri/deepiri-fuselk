"""QWebEngineView panel with loading overlay.

For the tokamak "port view" panel specifically, this also runs two small
JS-push bridges (via ``page().runJavaScript``, no QWebChannel needed):

* a sim-data bridge that polls the desktop API's ``/api/sim/frame`` JSON
  endpoint and calls ``window.__fuselk_updateFrame(frame)`` in the page,
  driving the real-data plasma/divertor rendering in
  ``tokamak_viewer.html``.
* a live-feed bridge that captures frames from :class:`LiveFeedSource`
  (webcam / RTSP / MJPEG, see ``viz/live_feed.py``) and calls
  ``window.__fuselk_setLiveFrame(dataUrl)`` / ``window.__fuselk_setLiveStatus``
  so the viewer's 'live' mode shows actual video instead of a synthetic
  scene.

Both bridges are best-effort: any network/camera failure just stops that
one push, it never breaks the underlying web view.
"""

from __future__ import annotations

import json
import logging

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QStackedLayout, QWidget

from deepiri_fuselk.viz.desktop.widgets import LoadingOverlay
from deepiri_fuselk.viz.live_feed import LiveFeedSource

logger = logging.getLogger(__name__)

_TOKAMAK_MARKER = "tokamak_viewer.html"
_SIM_FRAME_POLL_MS = 400
_LIVE_FEED_POLL_MS = 90  # ~11 fps cap; actual rate also bounded by LiveFeedSource


class WebPanel(QWidget):
    """Embeds a web view pointed at a URL or local file."""

    def __init__(
        self,
        url: str,
        *,
        loading_message: str = "Loading view…",
        parent: QWidget | None = None,
        enable_sim_bridge: bool | None = None,
        enable_live_feed: bool | None = None,
        live_feed_source: int | str = 0,
    ) -> None:
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

        is_tokamak = _TOKAMAK_MARKER in url
        self._enable_sim_bridge = is_tokamak if enable_sim_bridge is None else enable_sim_bridge
        self._enable_live_feed = is_tokamak if enable_live_feed is None else enable_live_feed

        self._sim_frame_url = self._derive_sim_frame_url(url) if self._enable_sim_bridge else None
        self._net_manager: QNetworkAccessManager | None = None
        self._sim_timer: QTimer | None = None
        self._sim_reply_in_flight = False

        self._live_feed: LiveFeedSource | None = None
        self._live_timer: QTimer | None = None
        self._live_feed_opened = False
        self._live_feed_source_spec = live_feed_source

        self.reload()

        if self._enable_sim_bridge and self._sim_frame_url:
            self._start_sim_bridge()
        if self._enable_live_feed:
            self._start_live_feed_bridge()

    @staticmethod
    def _derive_sim_frame_url(url: str) -> str | None:
        """Given the tokamak viewer's static-file URL, find the API base.

        ``url`` looks like ``http://host:port/api/static/tokamak_viewer.html``;
        the sim frame endpoint lives at ``http://host:port/api/sim/frame``.
        """
        marker = "/api/static/"
        idx = url.find(marker)
        if idx == -1:
            return None
        base = url[:idx]
        return f"{base}/api/sim/frame"

    # -- sim-data bridge (part A: real-data-driven rendering) -----------------

    def _start_sim_bridge(self) -> None:
        self._net_manager = QNetworkAccessManager(self)
        self._sim_timer = QTimer(self)
        self._sim_timer.timeout.connect(self._poll_sim_frame)
        self._sim_timer.start(_SIM_FRAME_POLL_MS)

    def _poll_sim_frame(self) -> None:
        if self._sim_reply_in_flight or self._net_manager is None or self._sim_frame_url is None:
            return
        self._sim_reply_in_flight = True
        reply = self._net_manager.get(QNetworkRequest(QUrl(self._sim_frame_url)))
        reply.finished.connect(lambda: self._on_sim_frame_reply(reply))

    def _on_sim_frame_reply(self, reply: QNetworkReply) -> None:
        self._sim_reply_in_flight = False
        try:
            if reply.error() != QNetworkReply.NetworkError.NoError:
                return
            payload = bytes(reply.readAll().data())
            frame = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return
        finally:
            reply.deleteLater()
        self._push_sim_frame(frame)

    def _push_sim_frame(self, frame: dict) -> None:
        script = f"window.__fuselk_updateFrame && window.__fuselk_updateFrame({json.dumps(frame)});"
        self.page().runJavaScript(script)

    # -- live-feed bridge (part B: actual video) -------------------------------

    def _start_live_feed_bridge(self) -> None:
        self._live_feed = LiveFeedSource(self._live_feed_source_spec)
        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._poll_live_feed)
        self._live_timer.start(_LIVE_FEED_POLL_MS)

    def _poll_live_feed(self) -> None:
        if self._live_feed is None:
            return
        if not self._live_feed_opened:
            self._live_feed_opened = True
            available = self._live_feed.open()
            self._push_live_status(available, self._live_feed.status())
            if not available:
                return
        live_frame = self._live_feed.read_frame()
        if live_frame is None:
            return
        script = (
            "window.__fuselk_setLiveFrame && "
            f"window.__fuselk_setLiveFrame({json.dumps(live_frame.data_url)});"
        )
        self.page().runJavaScript(script)

    def _push_live_status(self, available: bool, status: str) -> None:
        script = (
            "window.__fuselk_setLiveStatus && "
            f"window.__fuselk_setLiveStatus({json.dumps(available)}, {json.dumps(status)});"
        )
        self.page().runJavaScript(script)

    def _on_loaded(self, ok: bool) -> None:
        self._overlay.setVisible(not ok)
        if ok and self._enable_live_feed and self._live_feed is not None:
            # Page just (re)loaded — re-announce whatever live status we know.
            self._push_live_status(self._live_feed.is_available(), self._live_feed.status())

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

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._live_feed is not None:
            self._live_feed.release()
        super().closeEvent(event)
