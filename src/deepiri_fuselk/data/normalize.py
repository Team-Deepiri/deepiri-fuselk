"""Normalize external datasets into fuselk IMAS-compatible HDF5 shots."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from deepiri_fuselk.data.imas_loader import IMASShot, export_imas_hdf5
from deepiri_fuselk.data.schemas import ProfileData
from deepiri_fuselk.sim.synthetic_data_gen import generate_ece_shot


def _scalar_profile(rho: np.ndarray, center: float, edge: float) -> list[float]:
    return (center + (edge - center) * rho).tolist()


def odl_csv_to_shots(
    csv_path: Path,
    out_dir: Path,
    *,
    max_discharges: int | None = None,
    grid_size: int = 32,
) -> list[Path]:
    """
    Convert Open Density Limit CSV rows into per-discharge fuselk HDF5 shots.

    Experimental scalars become radial profiles; ECE heat fields are seeded
    deterministically per discharge so HELIX/ELM pipelines have 2D input.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    by_discharge: dict[str, list[dict[str, str]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            by_discharge.setdefault(row["discharge_ID"], []).append(row)

    discharge_ids = sorted(by_discharge.keys())
    if max_discharges is not None:
        discharge_ids = discharge_ids[:max_discharges]

    written: list[Path] = []
    rho = np.linspace(0, 1, grid_size)

    for discharge_id in discharge_ids:
        rows = sorted(by_discharge[discharge_id], key=lambda r: float(r["time"]))
        times = np.array([float(r["time"]) for r in rows])
        labels = np.array([int(float(r["density_limit_phase"])) for r in rows])
        density = np.array([float(r["density"]) for r in rows])
        ip = np.array([float(r["plasma_current"]) for r in rows])
        btor = np.array([float(r["toroidal_B_field"]) for r in rows])

        seed = abs(hash(discharge_id)) % (2**31)
        ece = generate_ece_shot(grid_size, seed=seed, island_amplitude=0.35 + labels.max() * 0.25)

        ne_center = float(density.mean()) * 1e20
        ne = ProfileData(
            radius=rho.tolist(),
            values=_scalar_profile(rho, ne_center, ne_center * 0.3),
            label="ne",
            units="m^-3",
        )
        te = ProfileData(
            radius=rho.tolist(),
            values=_scalar_profile(rho, 4e3, 8e2),
            label="Te",
            units="eV",
        )
        q = ProfileData(
            radius=rho.tolist(),
            values=_scalar_profile(rho, 1.0 + ip.mean(), 2.5 + btor.mean() * 0.1),
            label="q",
        )
        rot = ProfileData(
            radius=rho.tolist(),
            values=_scalar_profile(rho, 5e3, 1e3),
            label="omega",
            units="rad/s",
        )

        shot = IMASShot(
            shot_id=f"CMOD_{discharge_id}",
            device="Alcator C-Mod",
            time=times,
            ne_profile=ne,
            Te_profile=te,
            q_profile=q,
            rotation_profile=rot,
            heat_field=ece.heat_field,
            dataset=None,
        )
        out = export_imas_hdf5(shot, out_dir / f"CMOD_{discharge_id}.h5")
        _annotate_odl_metadata(
            out,
            discharge_id=discharge_id,
            density_limit_labels=labels,
            density_trace=density,
        )
        written.append(out)
    return written


def _annotate_odl_metadata(
    path: Path,
    *,
    discharge_id: str,
    density_limit_labels: np.ndarray,
    density_trace: np.ndarray,
) -> None:
    import h5py

    with h5py.File(path, "a") as f:
        f.attrs["source"] = "odl"
        f.attrs["discharge_id"] = discharge_id
        f.attrs["elm_event_rate"] = float(density_limit_labels.mean())
        meta = f.create_group("odl_meta")
        meta.create_dataset("density_limit_phase", data=density_limit_labels)
        meta.create_dataset("density", data=density_trace)
