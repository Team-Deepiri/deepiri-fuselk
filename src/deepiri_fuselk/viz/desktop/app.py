"""Launch the PySide6 desktop control room."""

from __future__ import annotations

import os
import sys
import time

from PySide6.QtCore import Qt

from deepiri_fuselk.viz.gui_server import start_gui_server, wait_for_servers


def run_desktop_gui(
    *,
    dash_port: int = 8050,
    api_port: int = 8765,
    host: str = "127.0.0.1",
) -> None:
    """Start Dash + API backends and open the Qt desktop shell."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        raise SystemExit(
            "PySide6 is required for the desktop GUI.\n"
            "Install with: poetry install --with desktop\n"
            "Or run: ./setup.sh"
        ) from exc

    os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

    dash_url = f"http://{host}:{dash_port}"
    api_url = f"http://{host}:{api_port}"
    align = Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter

    app = QApplication(sys.argv)
    app.setApplicationName("deepiri-fuselk")
    app.setOrganizationName("Deepiri")

    from deepiri_fuselk.viz.branding import app_icon
    from deepiri_fuselk.viz.desktop.shell import MainShell
    from deepiri_fuselk.viz.desktop.widgets import SplashScreen

    splash = SplashScreen()
    splash.show()
    app.processEvents()
    app.setWindowIcon(app_icon())

    splash.showMessage("Starting Dash control room + FastAPI…", align, Qt.GlobalColor.white)
    start_gui_server(
        dash_host=host,
        dash_port=dash_port,
        api_host=host,
        api_port=api_port,
    )

    deadline = time.time() + 90.0
    servers_ok = False
    while time.time() < deadline:
        app.processEvents()
        if wait_for_servers(
            dash_url=f"{dash_url}/",
            api_url=f"{api_url}/api/health",
            timeout=0.5,
        ):
            servers_ok = True
            break
        splash.showMessage("Waiting for fusion backends…", align, Qt.GlobalColor.white)
        time.sleep(0.15)

    splash.showMessage("Opening control room shell…", align, Qt.GlobalColor.white)
    app.processEvents()

    window = MainShell(dash_url=dash_url, api_url=api_url)
    window.show_startup_notice(servers_ok)
    window.show()
    splash.finish(window)
    sys.exit(app.exec())
