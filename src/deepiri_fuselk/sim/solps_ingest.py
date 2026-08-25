"""SOLPS / BOUT++ edge-plasma ingest → fuselk divertor heat maps.

Does not require a live SOLPS binary. Loads exported edge profiles (HDF5)
or builds a physically shaped synthetic SOL so Venturi / ReactorCell can
run on edge-like heat footprints instead of core ECE islands alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class EdgeProfiles:
    """1D radial edge profiles + optional 2D divertor heat map."""

    rho: np.ndarray
    ne: np.ndarray
    te: np.ndarray
    heat_flux_2d: np.ndarray
    source: str = "synthetic"

    def peak_heat(self) -> float:
        return float(np.max(self.heat_flux_mw_m2()))

    def heat_flux_mw_m2(self) -> np.ndarray:
        """Normalize stored map into MW/m²-scale values."""
        q = np.asarray(self.heat_flux_2d, dtype=np.float64)
        if float(np.max(q)) <= 5.0:
            return q * 10.0
        return q


def synthetic_solps_edge(
    grid_size: int = 32,
    *,
    lambda_q_mm: float = 2.0,
    q_peak_mw: float = 8.0,
    seed: int = 0,
) -> EdgeProfiles:
    """
    Exponential SOL heat footprint with a toroidally localized strike hotspot.

    ``lambda_q_mm`` is the heat-flux decay length (Eich scaling order ~1–3 mm
    mapped onto a unit divertor domain for the control stack).
    """
    rng = np.random.default_rng(seed)
    rho = np.linspace(0.85, 1.15, grid_size)
    # Upstream ne / Te fall across the separatrix
    ne = 1e19 * np.exp(-(rho - 0.85) / 0.12)
    te = 80.0 * np.exp(-(rho - 0.85) / 0.08)  # eV
    # 2D divertor: exponential fall-off from strike + toroidal modulation
    y = np.linspace(-1, 1, grid_size)
    x = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    lam = max(lambda_q_mm / 50.0, 0.02)  # map mm → domain units
    strike = np.exp(-np.abs(Y + 0.35) / lam) * np.exp(-(X**2) / 0.35)
    strike *= 1.0 + 0.15 * np.sin(3.0 * np.pi * X)
    heat = q_peak_mw * strike / max(float(np.max(strike)), 1e-9)
    heat += 0.05 * q_peak_mw * rng.standard_normal((grid_size, grid_size))
    heat = np.maximum(heat, 0.0)
    return EdgeProfiles(rho=rho, ne=ne, te=te, heat_flux_2d=heat, source="synthetic_solps")


def load_solps_hdf5(path: str | Path) -> EdgeProfiles:
    """
    Load edge export written by ``export_solps_hdf5``.

    Expected datasets: ``rho``, ``ne``, ``te``, ``heat_flux`` (2D).
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        rho = np.array(f["rho"]) if "rho" in f else np.linspace(0.85, 1.15, 32)
        ne = np.array(f["ne"]) if "ne" in f else 1e19 * np.ones_like(rho)
        te = np.array(f["te"]) if "te" in f else 50.0 * np.ones_like(rho)
        if "heat_flux" in f:
            heat = np.array(f["heat_flux"])
        else:
            heat = synthetic_solps_edge(len(rho)).heat_flux_2d
        source = str(f.attrs.get("source", "solps_hdf5"))
    return EdgeProfiles(rho=rho, ne=ne, te=te, heat_flux_2d=heat, source=source)


def export_solps_hdf5(profiles: EdgeProfiles, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["source"] = profiles.source
        f.create_dataset("rho", data=profiles.rho)
        f.create_dataset("ne", data=profiles.ne)
        f.create_dataset("te", data=profiles.te)
        f.create_dataset("heat_flux", data=profiles.heat_flux_2d, compression="gzip")
    return path


def edge_to_pipeline_heat(profiles: EdgeProfiles, grid_size: int | None = None) -> np.ndarray:
    """Resize divertor map for ShotPipeline / Venturi input."""
    heat = profiles.heat_flux_mw_m2()
    n = grid_size or heat.shape[0]
    if heat.shape[0] == n:
        return heat.astype(np.float64)
    from scipy.ndimage import zoom

    return zoom(heat, n / heat.shape[0], order=1).astype(np.float64)


class SOLPSIngest:
    """
    Prefer real HDF5 edge exports; fall back to synthetic SOL or stub wrapper.
    """

    def __init__(self, grid_points: int = 64) -> None:
        self.grid_points = grid_points
        self.last: EdgeProfiles | None = None

    def available(self) -> bool:
        return True

    def load(self, path: str | Path | None = None, **kwargs) -> EdgeProfiles:
        if path is not None and Path(path).exists():
            self.last = load_solps_hdf5(path)
        else:
            self.last = synthetic_solps_edge(
                grid_size=int(kwargs.get("grid_size", self.grid_points)),
                seed=int(kwargs.get("seed", 0)),
                q_peak_mw=float(kwargs.get("q_peak_mw", 8.0)),
                lambda_q_mm=float(kwargs.get("lambda_q_mm", 2.0)),
            )
        return self.last

    def feed_venturi_heat(self, grid_size: int = 32) -> np.ndarray:
        if self.last is None:
            self.load()
        assert self.last is not None
        return edge_to_pipeline_heat(self.last, grid_size)
