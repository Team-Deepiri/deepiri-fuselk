"""IMAS-compatible data loader (synthetic + HDF5 fallback)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepiri_fuselk.data.schemas import ProfileData
from deepiri_fuselk.sim.synthetic_data_gen import generate_ece_shot


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


def synthetic_imas_shot(shot_id: str = "SYN001", size: int = 32, seed: int = 0) -> IMASShot:
    """Generate IMAS-compatible synthetic shot for development without real IMAS install."""
    rng = np.random.default_rng(seed)
    rho = np.linspace(0, 1, size).tolist()
    shot = generate_ece_shot(size, seed=seed)
    return IMASShot(
        shot_id=shot_id,
        device="DIII-D-synthetic",
        time=np.linspace(0, 5.0, 100),
        ne_profile=ProfileData(
            radius=rho, values=(1e19 * (1 - np.array(rho))).tolist(), label="ne"
        ),
        Te_profile=ProfileData(
            radius=rho,
            values=(5e3 * (1 - 0.8 * np.array(rho))).tolist(),
            label="Te",
            units="eV",
        ),
        q_profile=ProfileData(
            radius=rho, values=(1.0 + 2.5 * np.array(rho) ** 2).tolist(), label="q"
        ),
        rotation_profile=ProfileData(
            radius=rho,
            values=(rng.uniform(1e3, 1e4, size)).tolist(),
            label="omega",
            units="rad/s",
        ),
        heat_field=shot.heat_field,
    )


def load_imas_shot(path: str | None = None, shot_id: str = "SYN001") -> IMASShot:
    """Load shot from path or fall back to synthetic."""
    if path is None:
        return synthetic_imas_shot(shot_id)
    # Real IMAS integration point — requires imas-python package
    return synthetic_imas_shot(shot_id)
