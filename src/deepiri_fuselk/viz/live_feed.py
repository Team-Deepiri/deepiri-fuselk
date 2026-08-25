"""Real (non-synthetic) video source for the port-view panel.

Wraps ``cv2.VideoCapture`` so the desktop GUI can stream actual camera
frames — a local webcam index, or an RTSP/MJPEG URL — into the tokamak
viewer's "live" mode. When no camera is available this degrades cleanly:
callers get ``is_available() == False`` and a clear status string instead
of a crash or a blank/broken view.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

try:
    import cv2

    _CV2_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - opencv-python missing/broken install
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = exc


@dataclass
class LiveFrame:
    """A single encoded frame ready to hand to the viewer."""

    data_url: str
    width: int
    height: int
    captured_at: float


class LiveFeedSource:
    """Captures frames from a webcam or network stream and JPEG-encodes them.

    Source can be a local camera index (``0``, ``1``, ...) or any URL string
    OpenCV's ``VideoCapture`` accepts (RTSP, MJPEG-over-HTTP, etc). Nothing
    opens the device until :meth:`open` is called, and every failure mode
    (missing opencv-python, no camera present, stream unreachable) is
    reported through :meth:`status` rather than raised.
    """

    def __init__(
        self,
        source: int | str = 0,
        *,
        max_fps: float = 12.0,
        jpeg_quality: int = 80,
    ) -> None:
        self._source = source
        self._min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._jpeg_quality = jpeg_quality
        self._cap: Any = None
        self._last_capture_t = 0.0
        self._status = "not started"
        self._opened = False

    def open(self) -> bool:
        """Try to open the underlying capture device. Never raises."""
        if cv2 is None:
            self._status = f"opencv-python unavailable: {_CV2_IMPORT_ERROR}"
            self._opened = False
            return False
        try:
            cap = cv2.VideoCapture(self._source)
        except Exception as exc:  # pragma: no cover - defensive
            self._status = f"failed to open source {self._source!r}: {exc}"
            self._opened = False
            return False
        if not cap.isOpened():
            cap.release()
            self._status = f"no live source detected at {self._source!r}"
            self._opened = False
            self._cap = None
            return False
        self._cap = cap
        self._opened = True
        self._status = f"live feed active ({self._source!r})"
        return True

    def is_available(self) -> bool:
        return self._opened and self._cap is not None

    def status(self) -> str:
        return self._status

    def read_frame(self) -> LiveFrame | None:
        """Grab and JPEG-encode the next frame, respecting the fps cap.

        Returns ``None`` when throttled, unavailable, or the device stops
        producing frames (e.g. camera unplugged) — callers should treat
        that as "no frame this tick", not necessarily "source gone".
        """
        if not self.is_available():
            return None
        now = time.monotonic()
        if now - self._last_capture_t < self._min_interval:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._status = "live source stopped producing frames"
            return None
        self._last_capture_t = now
        height, width = frame.shape[:2]
        encode_ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        )
        if not encode_ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return LiveFrame(
            data_url=f"data:image/jpeg;base64,{b64}",
            width=width,
            height=height,
            captured_at=now,
        )

    def release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # pragma: no cover - defensive
                logger.debug("LiveFeedSource release failed", exc_info=True)
        self._cap = None
        self._opened = False
        self._status = "released"
