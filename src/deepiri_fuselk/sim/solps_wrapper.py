"""SOLPS-ITER / BOUT++ edge coupling — ingest + optional live binary stub."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deepiri_fuselk.sim.solps_ingest import EdgeProfiles, SOLPSIngest, synthetic_solps_edge


@dataclass
class SOLPSConfig:
    grid_points: int = 64
    dt: float = 1e-5
    diffusion_coeff: float = 0.5


@dataclass
class SOLPSResult:
    density: np.ndarray
    temperature: np.ndarray
    heat_flux: np.ndarray


class SOLPSWrapper:
    """
    Edge-plasma coupling entry point.

    When a SOLPS binary is not linked, ``available()`` is False for the live
    binary but ``run_step`` / ``ingest`` still produce physically shaped SOL
    heat maps via ``SOLPSIngest`` so Venturi can train/act on edge footprints.
    """

    def __init__(self, config: SOLPSConfig | None = None) -> None:
        self.config = config or SOLPSConfig()
        self._ingest = SOLPSIngest(self.config.grid_points)
        self._binary_linked = False

    def available(self) -> bool:
        return self._binary_linked

    def ingest_available(self) -> bool:
        return True

    def load_edge(self, path: str | Path | None = None, **kwargs) -> EdgeProfiles:
        return self._ingest.load(path, **kwargs)

    def run_step(self, boundary_condition: np.ndarray | None = None) -> SOLPSResult:
        if self._binary_linked:
            # Placeholder for real SOLPS IPC — not yet wired.
            pass
        profiles = self._ingest.load(seed=0) if self._ingest.last is None else self._ingest.last
        assert profiles is not None
        if boundary_condition is not None:
            n = min(len(profiles.ne), len(boundary_condition))
            profiles.ne[:n] = profiles.ne[:n] * (1 + 0.1 * boundary_condition[:n])
        return SOLPSResult(
            density=profiles.ne,
            temperature=profiles.te,
            heat_flux=np.abs(profiles.heat_flux_2d.mean(axis=0)),
        )

    def divertor_heat_map(self, grid_size: int = 32) -> np.ndarray:
        return self._ingest.feed_venturi_heat(grid_size)


__all__ = [
    "EdgeProfiles",
    "SOLPSConfig",
    "SOLPSIngest",
    "SOLPSResult",
    "SOLPSWrapper",
    "synthetic_solps_edge",
]
