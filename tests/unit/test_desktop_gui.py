"""Desktop GUI import smoke tests (no display required)."""

import pytest

pytest.importorskip("PySide6")


def test_desktop_panels_import():
    from deepiri_fuselk.viz.desktop.panels import (
        DoctorPanel,
        ExperimentsPanel,
        MuonPanel,
        OilWaterPanel,
        SimLabPanel,
        WebPanel,
    )

    assert all([DoctorPanel, ExperimentsPanel, MuonPanel, OilWaterPanel, SimLabPanel, WebPanel])


def test_main_shell_import():
    from deepiri_fuselk.viz.desktop.shell import MainShell

    assert MainShell is not None
