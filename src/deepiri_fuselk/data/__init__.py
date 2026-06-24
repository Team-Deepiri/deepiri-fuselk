"""Data ingestion — IMAS-compatible, HDF5, synthetic shots, fetch pipeline."""

from deepiri_fuselk.data.fetchers import FETCHERS, run_fetch
from deepiri_fuselk.data.hdf5_store import FuselkStore, load_shot, save_shot
from deepiri_fuselk.data.imas_loader import (
    IMASShot,
    export_imas_hdf5,
    load_imas_hdf5,
    load_imas_shot,
    shot_to_plasma_record,
    synthetic_imas_shot,
)
from deepiri_fuselk.data.normalize import odl_csv_to_shots
from deepiri_fuselk.data.notebook_loaders import (
    ensure_fetched_data,
    imas_to_synthetic_shot,
    list_shots,
    load_corpus_arrays,
    load_odl_meta,
    load_shot_by_id,
    manifest_summary,
    resolve_data_root,
    resolve_repo_root,
)
from deepiri_fuselk.data.schemas import DiagnosticFrame, PlasmaShot, ProfileData
from deepiri_fuselk.data.sources import get_source, load_catalog

__all__ = [
    "DiagnosticFrame",
    "FETCHERS",
    "FuselkStore",
    "IMASShot",
    "PlasmaShot",
    "ProfileData",
    "export_imas_hdf5",
    "get_source",
    "load_catalog",
    "load_imas_hdf5",
    "load_imas_shot",
    "load_shot",
    "load_odl_meta",
    "load_shot_by_id",
    "manifest_summary",
    "resolve_data_root",
    "resolve_repo_root",
    "ensure_fetched_data",
    "imas_to_synthetic_shot",
    "list_shots",
    "load_corpus_arrays",
    "odl_csv_to_shots",
    "run_fetch",
    "save_shot",
    "shot_to_plasma_record",
    "synthetic_imas_shot",
]
