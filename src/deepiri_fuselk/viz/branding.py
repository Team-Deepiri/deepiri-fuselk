"""Deepiri brand asset paths (from deepiri-landing)."""

from __future__ import annotations

from pathlib import Path

_BRAND_DIR = Path(__file__).resolve().parent / "static" / "branding"

LOGO_PATH = _BRAND_DIR / "deepiri_logo.png"
LOGO_SQUARED_PATH = _BRAND_DIR / "deepiri_logo_squared.png"
FAVICON_SVG_PATH = _BRAND_DIR / "deepiri_favicon.svg"


def logo_path() -> Path:
    return LOGO_PATH


def branding_dir() -> Path:
    return _BRAND_DIR


def load_logo_pixmap(size: int = 48):
    """Scaled QPixmap of the Deepiri logo, or empty pixmap if missing."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPixmap

    if not LOGO_PATH.is_file():
        return QPixmap()
    px = QPixmap(str(LOGO_PATH))
    return px.scaled(
        size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )


def app_icon():
    """Window/taskbar icon from the Deepiri logo."""
    from PySide6.QtGui import QIcon

    if LOGO_PATH.is_file():
        return QIcon(str(LOGO_PATH))
    return QIcon()
