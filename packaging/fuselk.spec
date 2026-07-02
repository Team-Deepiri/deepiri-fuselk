# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Fuselk PySide6 desktop shell."""

from __future__ import annotations

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent.parent
SRC = ROOT / "src"

datas = [
    (str(ROOT / "experiments" / "registry.yaml"), "experiments"),
    (str(SRC / "deepiri_fuselk" / "viz" / "static"), "deepiri_fuselk/viz/static"),
]

for pkg in ("dash", "plotly", "flask", "werkzeug"):
    datas += collect_data_files(pkg)

hiddenimports = [
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineQuick",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "engineio.async_drivers.threading",
    "multipart",
    "sklearn.utils._typedefs",
    "sklearn.neighbors._partition_nodes",
    "pandas._libs.tslibs.timedeltas",
]
hiddenimports += collect_submodules("dash")
hiddenimports += collect_submodules("plotly")

a = Analysis(
    [str(ROOT / "packaging" / "fuselk_entry.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "jax",
        "jaxlib",
        "equinox",
        "optax",
        "jupyter",
        "notebook",
        "IPython",
        "matplotlib.tests",
        "pytest",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Fuselk",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Fuselk",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Fuselk.app",
        icon=None,
        bundle_identifier="com.deepiri.fuselk",
    )
