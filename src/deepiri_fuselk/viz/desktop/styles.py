"""Dark control-room stylesheet for the Qt shell."""

SHELL_QSS = """
* { font-family: "Segoe UI", "Ubuntu", system-ui, sans-serif; }

QMainWindow, QWidget#shell-root {
    background-color: #0b0d14;
    color: #e8ecf4;
}

QMenuBar {
    background-color: #10141f;
    color: #c8d0e0;
    border-bottom: 1px solid #222838;
    padding: 2px 0;
}
QMenuBar::item:selected { background: #1a2540; border-radius: 4px; }
QMenu {
    background-color: #141a28;
    color: #e8ecf4;
    border: 1px solid #2a3348;
}
QMenu::item:selected { background-color: #1e3a6e; }

QWidget#sidebar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #121826, stop:1 #0d1018);
    border-right: 1px solid #222838;
}
QLabel#brand-title {
    font-size: 17px;
    font-weight: 800;
    color: #ffffff;
    padding: 18px 18px 2px 18px;
    letter-spacing: 0.3px;
}
QLabel#brand-sub {
    font-size: 10px;
    color: #6a7a99;
    padding: 0 18px 10px 18px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}
QLabel#nav-section {
    font-size: 10px;
    font-weight: 700;
    color: #4a5875;
    padding: 14px 18px 6px 18px;
    letter-spacing: 1.4px;
}
QListWidget#nav-list {
    background: transparent;
    border: none;
    padding: 0 8px 12px 8px;
    font-size: 13px;
    outline: none;
}
QListWidget#nav-list::item {
    padding: 10px 12px;
    border-radius: 8px;
    margin: 2px 0;
    color: #b8c4dc;
}
QListWidget#nav-list::item:selected {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1a3d7a, stop:1 #224f99);
    color: #ffffff;
    font-weight: 600;
}
QListWidget#nav-list::item:hover:!selected {
    background-color: #171d2c;
}

QWidget#top-bar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #141a28, stop:1 #10141f);
    border-bottom: 1px solid #222838;
    min-height: 56px;
}
QLabel#top-panel-title {
    font-size: 16px;
    font-weight: 700;
    color: #ffffff;
}
QLabel#top-panel-sub {
    font-size: 11px;
    color: #7a8aaa;
}
QLabel#telemetry-label {
    font-size: 9px;
    color: #6a7a99;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
QLabel#telemetry-value {
    font-size: 14px;
    font-weight: 700;
    color: #e8ecf4;
}
QFrame#telemetry-chip {
    background-color: #171d2c;
    border: 1px solid #2a3348;
    border-radius: 8px;
    min-width: 96px;
    max-width: 120px;
}
QFrame#telemetry-chip[alert="true"] {
    border-color: #8b3030;
    background-color: #2a1518;
}
QFrame#telemetry-chip[alert="true"] QLabel#telemetry-value {
    color: #ff8a8a;
}

QWidget#panel-header {
    background-color: transparent;
}
QLabel#panel-title {
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
}
QLabel#panel-subtitle {
    font-size: 12px;
    color: #7a8aaa;
}
QFrame#panel-body {
    background-color: #0b0d14;
}

QFrame#kpi-card {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #171d2c, stop:1 #121826);
    border: 1px solid #2a3348;
    border-radius: 10px;
}
QLabel#kpi-title {
    font-size: 10px;
    color: #6a7a99;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
QLabel#kpi-value {
    font-size: 22px;
    font-weight: 800;
    color: #e8ecf4;
}
QLabel#kpi-hint {
    font-size: 11px;
    color: #6a7a99;
}

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #4d94ff, stop:1 #3377ee);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 9px 16px;
    font-weight: 600;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #66a3ff, stop:1 #4488ff);
}
QPushButton:pressed { background-color: #2255cc; }
QPushButton:disabled { background-color: #2a3040; color: #666; }
QPushButton#secondary {
    background-color: #1e2433;
    border: 1px solid #2a3348;
    color: #c8d0e0;
}
QPushButton#secondary:hover { background-color: #252d40; }
QPushButton#accent-green {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #3dd68c, stop:1 #2bb87a);
}

QLineEdit, QSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
    background-color: #121826;
    color: #e8ecf4;
    border: 1px solid #2a3348;
    border-radius: 6px;
    padding: 7px 10px;
    selection-background-color: #1e3a6e;
}
QComboBox::drop-down { border: none; width: 24px; }
QSpinBox::up-button, QSpinBox::down-button { width: 18px; border: none; }

QTableWidget {
    background-color: #121826;
    color: #e8ecf4;
    gridline-color: #222838;
    border: 1px solid #2a3348;
    border-radius: 8px;
    alternate-background-color: #0f131c;
}
QHeaderView::section {
    background-color: #171d2c;
    color: #8a9ab8;
    border: none;
    border-bottom: 1px solid #2a3348;
    padding: 8px;
    font-weight: 600;
    font-size: 11px;
}
QTableWidget::item:selected {
    background-color: #1e3a6e;
}

QGroupBox {
    border: 1px solid #2a3348;
    border-radius: 10px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
    color: #8a9ab8;
    background-color: #10141f;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
}

QStatusBar {
    background-color: #10141f;
    color: #6a7a99;
    border-top: 1px solid #222838;
    font-size: 11px;
}
QScrollBar:vertical {
    background: #0b0d14;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #2a3348;
    border-radius: 5px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #3a4a68; }

QWidget#loading-overlay {
    background-color: rgba(11, 13, 20, 230);
}
QLabel#loading-label {
    color: #8a9ab8;
    font-size: 13px;
    padding-bottom: 8px;
}
QProgressBar#loading-bar {
    background-color: #1e2433;
    border: none;
    border-radius: 4px;
    height: 6px;
}
QProgressBar#loading-bar::chunk {
    background-color: #4488ff;
    border-radius: 4px;
}
"""
