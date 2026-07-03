#!/usr/bin/env bash
# Build Fuselk desktop release artifact for the current OS.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/release"
export SKIP_FETCH=1
export QTWEBENGINE_DISABLE_SANDBOX=1

cd "$ROOT"
mkdir -p "$OUT"
rm -rf "$ROOT/build" "$ROOT/dist"

echo "==> Installing Poetry dependencies"
pip install poetry pyinstaller
poetry config virtualenvs.in-project true
poetry install --no-interaction --with desktop

echo "==> Running PyInstaller"
poetry run pyinstaller packaging/fuselk.spec --noconfirm --clean

case "$(uname -s)" in
  Linux)
  echo "==> Building AppImage"
  APPDIR="$ROOT/build/AppDir"
  rm -rf "$APPDIR"
  mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications"
  cp -a "$ROOT/dist/Fuselk/." "$APPDIR/usr/bin/"
  cat > "$APPDIR/fuselk.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=Fuselk
Exec=Fuselk
Icon=fuselk
Categories=Science;
EOF
  if [[ -f "$ROOT/src/deepiri_fuselk/viz/static/branding/deepiri_favicon.svg" ]]; then
    cp "$ROOT/src/deepiri_fuselk/viz/static/branding/deepiri_favicon.svg" "$APPDIR/fuselk.svg"
    cp "$APPDIR/fuselk.svg" "$APPDIR/.DirIcon"
  fi
  curl -fsSL -o /tmp/appimagetool.AppImage \
    https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
  chmod +x /tmp/appimagetool.AppImage
  ARCH=x86_64 /tmp/appimagetool.AppImage "$APPDIR" "$OUT/Fuselk-latest.AppImage"
  ;;
  Darwin)
  echo "==> Building DMG"
  hdiutil create -volname "Fuselk" -srcfolder "$ROOT/dist/Fuselk.app" -ov -format UDZO "$OUT/Fuselk-latest.dmg"
  ;;
  MINGW*|MSYS*|CYGWIN*)
  echo "==> Packaging Windows installer executable"
  cp "$ROOT/dist/Fuselk/Fuselk.exe" "$OUT/Fuselk-latest-setup.exe"
  ;;
  *)
  echo "Unsupported OS: $(uname -s)" >&2
  exit 1
  ;;
esac

echo "==> Release artifacts"
ls -la "$OUT"
