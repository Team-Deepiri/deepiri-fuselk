"""Main desktop window — sidebar, telemetry bar, stacked panels."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from deepiri_fuselk import __version__
from deepiri_fuselk.viz.branding import app_icon, load_logo_pixmap
from deepiri_fuselk.viz.desktop.nav_config import NAV_ENTRIES
from deepiri_fuselk.viz.desktop.panels import (
    DoctorPanel,
    ExperimentsPanel,
    MuonPanel,
    OilWaterPanel,
    SimLabPanel,
    WebPanel,
)
from deepiri_fuselk.viz.desktop.styles import SHELL_QSS
from deepiri_fuselk.viz.desktop.widgets import PanelChrome, StatusDot


class MainShell(QMainWindow):
    def __init__(
        self,
        *,
        dash_url: str = "http://127.0.0.1:8050",
        api_url: str = "http://127.0.0.1:8765",
    ) -> None:
        super().__init__()
        self._dash_url = dash_url
        self._api_url = api_url
        self._entries = {e.key: e for e in NAV_ENTRIES}
        self.setWindowTitle("deepiri-fuselk Control Room")
        self.setMinimumSize(QSize(1200, 760))
        self.resize(1520, 940)
        self.setStyleSheet(SHELL_QSS)
        self.setWindowIcon(app_icon())
        self._stack = QStackedWidget()
        self._nav = QListWidget()
        self._nav.setObjectName("nav-list")
        self._telemetry: dict[str, QLabel | None] = {}
        self._telemetry_chips: dict[str, QFrame] = {}
        self._build_panels()
        self._build_layout()
        self._build_menu()
        self._nav.setCurrentRow(0)
        self._start_telemetry()
        self.statusBar().showMessage(f"deepiri-fuselk v{__version__} — ready")

    def _build_panels(self) -> None:
        tokamak_url = f"{self._api_url}/api/static/tokamak_viewer.html"
        native: dict[str, QWidget] = {
            "sim_lab": SimLabPanel(),
            "experiments": ExperimentsPanel(),
            "oil_water": OilWaterPanel(),
            "muon": MuonPanel(),
            "doctor": DoctorPanel(),
        }
        self._panels: dict[str, QWidget] = {
            "control_room": WebPanel(
                self._dash_url, loading_message="Connecting to live Dash control room…"
            ),
            "tokamak": WebPanel(tokamak_url, loading_message="Loading 3D tokamak viewer…"),
        }
        self._key_to_index: dict[str, int] = {}

        for entry in NAV_ENTRIES:
            if entry.is_web:
                panel = self._panels[entry.key]
            else:
                panel = PanelChrome(entry.title, entry.subtitle, native[entry.key])
            self._panels[entry.key] = panel
            idx = self._stack.addWidget(panel)
            self._key_to_index[entry.key] = idx

        self._nav.currentRowChanged.connect(self._on_nav_changed)

    def _build_layout(self) -> None:
        central = QWidget()
        central.setObjectName("shell-root")
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_top_bar())
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_sidebar())
        body.addWidget(self._stack, stretch=1)
        outer.addLayout(body, stretch=1)
        self.setCentralWidget(central)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(248)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        brand_row = QWidget()
        brand_row.setObjectName("brand-row")
        brand_layout = QHBoxLayout(brand_row)
        brand_layout.setContentsMargins(16, 16, 16, 8)
        brand_layout.setSpacing(10)
        logo = QLabel()
        logo.setObjectName("brand-logo")
        logo.setPixmap(load_logo_pixmap(40))
        logo.setFixedSize(40, 40)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(2)
        title = QLabel("deepiri-fuselk")
        title.setObjectName("brand-title")
        sub = QLabel("Fusion control room")
        sub.setObjectName("brand-sub")
        brand_text.addWidget(title)
        brand_text.addWidget(sub)
        brand_layout.addWidget(logo)
        brand_layout.addLayout(brand_text, stretch=1)
        layout.addWidget(brand_row)

        row_by_section: dict[str, int] = {}
        for entry in NAV_ENTRIES:
            if entry.section not in row_by_section:
                sec = QLabel(entry.section.upper())
                sec.setObjectName("nav-section")
                layout.addWidget(sec)
            item = QListWidgetItem(f"  {entry.icon}   {entry.label}")
            item.setData(Qt.ItemDataRole.UserRole, entry.key)
            item.setToolTip(f"{entry.subtitle}\n{entry.shortcut}")
            self._nav.addItem(item)
            row_by_section[entry.section] = self._nav.count() - 1

        layout.addWidget(self._nav, stretch=1)

        self._dash_status = StatusDot("Dash")
        self._api_status = StatusDot("API")
        footer = QWidget()
        foot_layout = QVBoxLayout(footer)
        foot_layout.setContentsMargins(18, 8, 18, 14)
        foot_layout.addWidget(self._dash_status)
        foot_layout.addWidget(self._api_status)
        layout.addWidget(footer)
        return sidebar

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("top-bar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 10, 20, 10)

        titles = QVBoxLayout()
        titles.setSpacing(2)
        self._top_title = QLabel("Live Control Room")
        self._top_title.setObjectName("top-panel-title")
        self._top_sub = QLabel("")
        self._top_sub.setObjectName("top-panel-sub")
        titles.addWidget(self._top_title)
        titles.addWidget(self._top_sub)
        layout.addLayout(titles, stretch=2)

        for key, label in [
            ("fusion", "Fusion"),
            ("disruption", "Disruption"),
            ("tbr", "TBR"),
            ("snr", "HELIX SNR"),
        ]:
            chip = self._telemetry_chip(label)
            value_label = chip.findChild(QLabel, "telemetry-value")
            self._telemetry[key] = value_label  # may be None before first update
            self._telemetry_chips[key] = chip
            layout.addWidget(chip)

        return bar

    def _telemetry_chip(self, label: str) -> QFrame:
        chip = QFrame()
        chip.setObjectName("telemetry-chip")
        lay = QVBoxLayout(chip)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(0)
        t = QLabel(label)
        t.setObjectName("telemetry-label")
        v = QLabel("—")
        v.setObjectName("telemetry-value")
        lay.addWidget(t)
        lay.addWidget(v)
        return chip

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        reset_action = QAction("Reset simulation", self)
        reset_action.setShortcut(QKeySequence("Ctrl+Shift+R"))
        reset_action.triggered.connect(self._reset_sim_via_api)
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(reset_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)

        view_menu = self.menuBar().addMenu("&View")
        reload_action = QAction("Reload web panel", self)
        reload_action.setShortcut(QKeySequence("Ctrl+L"))
        reload_action.triggered.connect(self._reload_web_panel)
        view_menu.addAction(reload_action)

        nav_menu = self.menuBar().addMenu("&Navigate")
        for i, entry in enumerate(NAV_ENTRIES, start=1):
            action = QAction(f"{entry.icon} {entry.label}", self)
            action.setShortcut(QKeySequence(f"Ctrl+{i}"))
            action.triggered.connect(lambda _c=False, k=entry.key: self._go_to(k))
            nav_menu.addAction(action)

    def _on_nav_changed(self, row: int) -> None:
        if row < 0:
            return
        key = self._nav.item(row).data(Qt.ItemDataRole.UserRole)
        self._stack.setCurrentIndex(self._key_to_index[key])
        entry = self._entries[key]
        self._top_title.setText(entry.title)
        self._top_sub.setText(entry.subtitle)
        self.statusBar().showMessage(f"{entry.label}  ·  {entry.shortcut}")

    def _go_to(self, key: str) -> None:
        for row in range(self._nav.count()):
            if self._nav.item(row).data(Qt.ItemDataRole.UserRole) == key:
                self._nav.setCurrentRow(row)
                break

    def _reload_web_panel(self) -> None:
        row = self._nav.currentRow()
        if row < 0:
            return
        key = self._nav.item(row).data(Qt.ItemDataRole.UserRole)
        panel = self._panels.get(key)
        if isinstance(panel, WebPanel):
            panel.reload()
            self.statusBar().showMessage(f"Reloaded {key}")

    def _reset_sim_via_api(self) -> None:
        try:
            req = urllib.request.Request(
                f"{self._api_url}/api/sim/reset",
                data=b"{}",
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
            self._poll_telemetry()
            self.statusBar().showMessage("Simulation reset")
        except (urllib.error.URLError, TimeoutError) as exc:
            self.statusBar().showMessage(f"Reset failed: {exc}")

    def _start_telemetry(self) -> None:
        self._poll_telemetry()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_telemetry)
        self._timer.start(2500)

    def _poll_telemetry(self) -> None:
        self._dash_status.set_status(self._ping(f"{self._dash_url}/"))
        self._api_status.set_status(self._ping(f"{self._api_url}/api/health"))
        try:
            with urllib.request.urlopen(f"{self._api_url}/api/sim/frame", timeout=3) as resp:
                frame = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return
        fusion_lbl = self._telemetry.get("fusion")
        if fusion_lbl is not None:
            fusion_lbl.setText(f"{frame['fusion_score']:.1%}")
        risk = frame["disruption_probability"]
        disruption_lbl = self._telemetry.get("disruption")
        if disruption_lbl is not None:
            disruption_lbl.setText(f"{risk:.1%}")
        chip = self._telemetry_chips.get("disruption")
        if chip is not None:
            chip.setProperty("alert", "true" if risk > 0.5 else "false")
            chip.style().unpolish(chip)
            chip.style().polish(chip)
        tbr_lbl = self._telemetry.get("tbr")
        if tbr_lbl is not None:
            tbr_lbl.setText(f"{frame['tbr']:.2f}")
        snr_lbl = self._telemetry.get("snr")
        if snr_lbl is not None:
            snr_lbl.setText(f"{frame['helix']['phase_locked_snr']:.1f}x")

    @staticmethod
    def _ping(url: str) -> bool:
        try:
            urllib.request.urlopen(url, timeout=1.5)
            return True
        except (urllib.error.URLError, TimeoutError):
            return False

    def show_startup_notice(self, servers_ok: bool) -> None:
        if servers_ok:
            self._dash_status.set_status(True)
            self._api_status.set_status(True)
            return
        QMessageBox.warning(
            self,
            "Backend startup",
            "Dash/API servers did not respond in time.\n"
            "Web panels may show a loading state — use View → Reload web panel shortly.",
        )
