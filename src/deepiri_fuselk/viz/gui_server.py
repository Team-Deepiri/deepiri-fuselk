"""Launch Dash control room + FastAPI backend for the desktop GUI."""

from __future__ import annotations

import threading
import time

import uvicorn

from deepiri_fuselk.viz.api import create_api
from deepiri_fuselk.viz.dashboard.app import create_app


def start_gui_server(
    *,
    dash_host: str = "127.0.0.1",
    dash_port: int = 8050,
    api_host: str = "127.0.0.1",
    api_port: int = 8765,
) -> tuple[threading.Thread, threading.Thread]:
    """Start Dash dashboard and FastAPI in background daemon threads."""
    dash_app = create_app()
    api_app = create_api()

    def _run_dash() -> None:
        dash_app.run_server(host=dash_host, port=dash_port, debug=False, use_reloader=False)

    def _run_api() -> None:
        uvicorn.run(api_app, host=api_host, port=api_port, log_level="warning")

    dash_thread = threading.Thread(target=_run_dash, daemon=True, name="fuselk-dash")
    api_thread = threading.Thread(target=_run_api, daemon=True, name="fuselk-api")
    dash_thread.start()
    api_thread.start()
    return dash_thread, api_thread


def run_gui_server(
    *,
    dash_host: str = "127.0.0.1",
    dash_port: int = 8050,
    api_host: str = "127.0.0.1",
    api_port: int = 8765,
    block: bool = True,
) -> None:
    """Start backends; optionally block until threads exit."""
    dash_thread, api_thread = start_gui_server(
        dash_host=dash_host,
        dash_port=dash_port,
        api_host=api_host,
        api_port=api_port,
    )
    if block:
        while dash_thread.is_alive() and api_thread.is_alive():
            time.sleep(0.5)


def wait_for_servers(
    *,
    dash_url: str = "http://127.0.0.1:8050",
    api_url: str = "http://127.0.0.1:8765/api/health",
    timeout: float = 60.0,
) -> bool:
    """Poll until both servers respond or timeout."""
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    dash_ok = api_ok = False
    while time.time() < deadline and not (dash_ok and api_ok):
        if not dash_ok:
            try:
                urllib.request.urlopen(dash_url, timeout=1)
                dash_ok = True
            except (urllib.error.URLError, TimeoutError):
                pass
        if not api_ok:
            try:
                urllib.request.urlopen(api_url, timeout=1)
                api_ok = True
            except (urllib.error.URLError, TimeoutError):
                pass
        if not (dash_ok and api_ok):
            time.sleep(0.25)
    return dash_ok and api_ok
