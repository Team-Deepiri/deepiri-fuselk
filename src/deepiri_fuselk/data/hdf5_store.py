"""HDF5 persistence for fuselk shots and simulation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from deepiri_fuselk.data.schemas import PlasmaShot


class FuselkStore:
    """Local HDF5 store for shots and simulation results."""

    def __init__(self, root: str | Path = ".fuselk-data") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def shot_path(self, shot_id: str) -> Path:
        return self.root / f"{shot_id}.h5"

    def save(self, shot: PlasmaShot, arrays: dict[str, np.ndarray] | None = None) -> Path:
        path = self.shot_path(shot.shot_id)
        with h5py.File(path, "w") as f:
            f.attrs["shot_id"] = shot.shot_id
            f.attrs["device"] = shot.device
            f.attrs["metadata"] = json.dumps(shot.metadata)
            if arrays:
                for key, arr in arrays.items():
                    f.create_dataset(key, data=arr, compression="gzip")
        return path

    def load_arrays(self, shot_id: str) -> dict[str, np.ndarray]:
        path = self.shot_path(shot_id)
        result: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                result[key] = np.array(f[key])
        return result


def save_shot(shot_id: str, arrays: dict[str, np.ndarray], root: str = ".fuselk-data") -> Path:
    store = FuselkStore(root)
    shot = PlasmaShot(shot_id=shot_id)
    return store.save(shot, arrays)


def load_shot(shot_id: str, root: str = ".fuselk-data") -> dict[str, np.ndarray]:
    return FuselkStore(root).load_arrays(shot_id)
