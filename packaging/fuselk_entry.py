"""PyInstaller entrypoint for the Fuselk desktop control room."""

from __future__ import annotations

import os

os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

from deepiri_fuselk.viz.desktop.app import run_desktop_gui

if __name__ == "__main__":
    run_desktop_gui()
