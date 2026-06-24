"""IMAS-compatible data loader with HDF5 shot archive support."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from deepiri_fuselk.data.schemas import PlasmaShot, ProfileData


@dataclass
class IMASShot:
    """IMAS-style shot bundle."""

    shot_id: str
    device: str
    time: np.ndarray
    ne_profile: ProfileData
    Te_profile: ProfileData
    q_profile: ProfileData
    rotation_profile: ProfileData
    heat_field: np.ndarray
    dataset: xr.Dataset | None = None


def _profile_from_h5(group: h5py.Group, name: str) -> ProfileData:
    rho = group[f"{name}/rho"][:] if f"{name}/rho" in group else group["rho"][:]
    vals = group[f"{name}/values"][:] if f"{name}/values" in group else group["values"][:]
    units = group[f"{name}"].attrs.get("units", "")
    return ProfileData(radius=rho.tolist(), values=vals.tolist(), label=name, units=str(units))


def load_imas_hdf5(path: str | Path) -> IMASShot:
    """Load shot from fuselk HDF5 archive or IMAS-exported HDF5."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        shot_id = f.attrs.get("shot_id", path.stem)
        device = f.attrs.get("device", "unknown")
        time = np.array(f["time"]) if "time" in f else np.linspace(0, 5, 100)
        heat = np.array(f["ece/heat_field"]) if "ece/heat_field" in f else np.array(f["heat_field"])

        profiles_grp = f["profiles"] if "profiles" in f else f
        ne = (
            _profile_from_h5(profiles_grp, "ne")
            if "ne" in profiles_grp or "ne/rho" in profiles_grp
            else None
        )
        Te = (
            _profile_from_h5(profiles_grp, "Te")
            if "Te" in profiles_grp or "Te/rho" in profiles_grp
            else None
        )
        q = (
            _profile_from_h5(profiles_grp, "q")
            if "q" in profiles_grp or "q/rho" in profiles_grp
            else None
        )
        rot = (
            _profile_from_h5(profiles_grp, "omega")
            if "omega" in profiles_grp or "omega/rho" in profiles_grp
            else None
        )

    if ne is None:
        return synthetic_imas_shot(str(shot_id), size=heat.shape[0])

    ds = xr.Dataset(
        {
            "ne": ("rho", ne.values),
            "Te": ("rho", Te.values if Te else ne.values),
            "q": ("rho", q.values if q else ne.values),
        },
        coords={"rho": ne.radius, "time": time},
        attrs={"shot_id": shot_id, "device": device},
    )
    return IMASShot(
        shot_id=str(shot_id),
        device=str(device),
        time=time,
        ne_profile=ne,
        Te_profile=Te or ne,
        q_profile=q or ne,
        rotation_profile=rot or ne,
        heat_field=heat,
        dataset=ds,
    )


def export_imas_hdf5(shot: IMASShot, path: str | Path) -> Path:
    """Write shot to fuselk HDF5 archive format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["shot_id"] = shot.shot_id
        f.attrs["device"] = shot.device
        f.create_dataset("time", data=shot.time)
        f.create_dataset("heat_field", data=shot.heat_field, compression="gzip")
        grp = f.create_group("profiles")
        for label, prof in [
            ("ne", shot.ne_profile),
            ("Te", shot.Te_profile),
            ("q", shot.q_profile),
            ("omega", shot.rotation_profile),
        ]:
            pg = grp.create_group(label)
            pg.create_dataset("rho", data=np.array(prof.radius))
            pg.create_dataset("values", data=np.array(prof.values))
            pg.attrs["units"] = prof.units
    return path


def synthetic_imas_shot(shot_id: str = "SYN001", size: int = 32, seed: int = 0) -> IMASShot:
    """Generate IMAS-compatible synthetic shot."""
    from deepiri_fuselk.sim.synthetic_data_gen import generate_ece_shot

    rng = np.random.default_rng(seed)
    rho = np.linspace(0, 1, size).tolist()
    shot = generate_ece_shot(size, seed=seed)
    ne = ProfileData(radius=rho, values=(1e19 * (1 - np.array(rho))).tolist(), label="ne")
    Te = ProfileData(
        radius=rho, values=(5e3 * (1 - 0.8 * np.array(rho))).tolist(), label="Te", units="eV"
    )
    q = ProfileData(radius=rho, values=(1.0 + 2.5 * np.array(rho) ** 2).tolist(), label="q")
    rot = ProfileData(
        radius=rho,
        values=rng.uniform(1e3, 1e4, size).tolist(),
        label="omega",
        units="rad/s",
    )
    ds = xr.Dataset(
        {"ne": ("rho", ne.values), "Te": ("rho", Te.values), "q": ("rho", q.values)},
        coords={"rho": rho},
    )
    return IMASShot(
        shot_id=shot_id,
        device="DIII-D-synthetic",
        time=np.linspace(0, 5.0, 100),
        ne_profile=ne,
        Te_profile=Te,
        q_profile=q,
        rotation_profile=rot,
        heat_field=shot.heat_field,
        dataset=ds,
    )


def load_imas_shot(path: str | None = None, shot_id: str = "SYN001") -> IMASShot:
    if path is not None and Path(path).exists():
        return load_imas_hdf5(path)
    return synthetic_imas_shot(shot_id)


def shot_to_plasma_record(shot: IMASShot) -> PlasmaShot:
    meta = {"device": shot.device}
    if shot.dataset is not None:
        meta["xarray"] = json.dumps({"dims": list(shot.dataset.dims)})
    return PlasmaShot(
        shot_id=shot.shot_id,
        device=shot.device,
        profiles={"ne": shot.ne_profile, "Te": shot.Te_profile, "q": shot.q_profile},
        metadata=meta,
    )
