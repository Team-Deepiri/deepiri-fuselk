"""Desktop tool panels."""

from deepiri_fuselk.viz.desktop.panels.doctor_panel import DoctorPanel
from deepiri_fuselk.viz.desktop.panels.experiments_panel import ExperimentsPanel
from deepiri_fuselk.viz.desktop.panels.physics_panel import MuonPanel, OilWaterPanel
from deepiri_fuselk.viz.desktop.panels.sim_lab_panel import SimLabPanel
from deepiri_fuselk.viz.desktop.panels.web_panel import WebPanel

__all__ = [
    "DoctorPanel",
    "ExperimentsPanel",
    "MuonPanel",
    "OilWaterPanel",
    "SimLabPanel",
    "WebPanel",
]
