"""Navigation metadata for the desktop shell."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NavEntry:
    key: str
    label: str
    icon: str
    title: str
    subtitle: str
    section: str
    shortcut: str
    is_web: bool = False


NAV_ENTRIES: list[NavEntry] = [
    NavEntry(
        "control_room",
        "Control Room",
        "◉",
        "Live Control Room",
        "FusionCell dashboard — HELIX · Venturi · ELM · fuel cycle",
        "Visualization",
        "Ctrl+1",
        is_web=True,
    ),
    NavEntry(
        "tokamak",
        "3D Tokamak",
        "◎",
        "Tokamak Viewer",
        "Three.js cross-section with plasma core and O-point track",
        "Visualization",
        "Ctrl+2",
        is_web=True,
    ),
    NavEntry(
        "sim_lab",
        "Simulation Lab",
        "⏵",
        "Simulation Lab",
        "Step the live FusionCell loop and run batch scoring",
        "Simulation",
        "Ctrl+3",
    ),
    NavEntry(
        "experiments",
        "Experiments",
        "⚗",
        "Experiment Catalog",
        "Run registered fuselk experiments from the YAML registry",
        "Simulation",
        "Ctrl+4",
    ),
    NavEntry(
        "oil_water",
        "Oil-Water PDE",
        "≋",
        "Oil-Water Barrier",
        "Coupled plasma/vapor PDE steady and transient solves",
        "Physics",
        "Ctrl+5",
    ),
    NavEntry(
        "muon",
        "Muon Cycle",
        "μ",
        "Muon Fuel Cycle",
        "Photon/proton stripping rate network and breakeven",
        "Physics",
        "Ctrl+6",
    ),
    NavEntry(
        "doctor",
        "System Doctor",
        "✓",
        "System Doctor",
        "Dependency health check for core and desktop modules",
        "System",
        "Ctrl+7",
    ),
]

SECTIONS = ["Visualization", "Simulation", "Physics", "System"]
