#!/usr/bin/env bash
# deepiri-fuselk — install system deps, Python packages, and optional desktop GUI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

RUN_GUI=false
SKIP_SYSTEM=false

usage() {
  cat <<'EOF'
Usage: ./setup.sh [OPTIONS]

Install everything needed to run deepiri-fuselk (system + Python deps).

Options:
  --run            After install, launch the PySide6 desktop control room
  --skip-system    Skip apt/dnf system package install (Python only)
  -h, --help       Show this help

Examples:
  ./setup.sh
  ./setup.sh --run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN_GUI=true; shift ;;
    --skip-system) SKIP_SYSTEM=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

log() { printf '\033[1;34m[fuselk setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[fuselk setup]\033[0m %s\n' "$*" >&2; }

install_system_deps_apt() {
  log "Installing system packages (apt)…"
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    pkg-config \
    python3 \
    python3-dev \
    python3-venv \
    libhdf5-dev \
    libgl1 \
    libglib2.0-0 \
    libfontconfig1 \
    libdbus-1-3 \
    libxcb1 \
    libxcb-cursor0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xkb1 \
    libx11-xcb1 \
    libxkbcommon0 \
    libxkbcommon-x11-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libgbm1 \
    libegl1 \
    libopengl0 \
    libasound2 \
    libpulse0 \
    libpango-1.0-0 \
    libcairo2 \
    libcups2 \
    libexpat1 \
    libxshmfence1
}

install_system_deps_dnf() {
  log "Installing system packages (dnf)…"
  sudo dnf install -y \
    gcc gcc-c++ make \
    curl git \
    python3 python3-devel \
    hdf5-devel \
    mesa-libGL \
    glib2 fontconfig dbus-libs \
    libxcb libX11-xcb libxkbcommon-x11 \
    nss atk at-spi2-atk \
    libdrm mesa-libgbm \
    alsa-lib pulseaudio-libs \
    pango cairo cups-libs expat
}

install_system_deps() {
  if [[ "$SKIP_SYSTEM" == true ]]; then
    warn "Skipping system package install (--skip-system)"
    return
  fi
  if command -v apt-get >/dev/null 2>&1; then
    install_system_deps_apt
  elif command -v dnf >/dev/null 2>&1; then
    install_system_deps_dnf
  else
    warn "No supported package manager (apt/dnf). Install Qt6/WebEngine libs manually."
  fi
}

ensure_poetry() {
  if command -v poetry >/dev/null 2>&1; then
    log "Poetry found: $(poetry --version)"
    return
  fi
  log "Installing Poetry…"
  curl -sSL https://install.python-poetry.org | python3 -
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v poetry >/dev/null 2>&1; then
    echo "Poetry install failed — add ~/.local/bin to PATH and re-run." >&2
    exit 1
  fi
}

install_python_deps() {
  log "Installing Python dependencies (core + desktop)…"
  poetry config virtualenvs.in-project true --local 2>/dev/null || true
  poetry install --no-interaction --with desktop
  log "Running fuselk doctor…"
  poetry run fuselk doctor
}

fetch_datasets() {
  if [[ "${SKIP_FETCH:-}" == "1" ]]; then
    warn "Skipping data fetch (SKIP_FETCH=1)"
    return
  fi
  log "Fetching public datasets (synthetic + MIT Open Density Limit)…"
  poetry run python scripts/fetch_data.py --all --shots 100 --max-odl 50
}

launch_gui() {
  log "Launching desktop control room…"
  export QTWEBENGINE_DISABLE_SANDBOX=1
  poetry run fuselk gui
}

main() {
  log "deepiri-fuselk setup (root: $ROOT)"
  install_system_deps
  ensure_poetry
  install_python_deps
  fetch_datasets
  log "Setup complete."
  log "  Desktop GUI:  ./setup.sh --run   or   poetry run fuselk gui"
  log "  Data fetch:   python scripts/fetch_data.py --all"
  if [[ "$RUN_GUI" == true ]]; then
    launch_gui
  fi
}

main
