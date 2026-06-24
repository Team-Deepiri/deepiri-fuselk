"""Data ingestion — IMAS-compatible, HDF5, synthetic shots."""

from deepiri_fuselk.data.hdf5_store import FuselkStore, load_shot, save_shot
from deepiri_fuselk.data.imas_loader import IMASShot, load_imas_shot, synthetic_imas_shot
from deepiri_fuselk.data.schemas import DiagnosticFrame, PlasmaShot, ProfileData

__all__ = [
    "DiagnosticFrame",
    "FuselkStore",
    "IMASShot",
    "PlasmaShot",
    "ProfileData",
    "load_imas_shot",
    "load_shot",
    "save_shot",
    "synthetic_imas_shot",
]
