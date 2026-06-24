"""Pydantic schemas for fusion diagnostic data."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator


class ProfileData(BaseModel):
    """1D radial profile (density, temperature, rotation, q)."""

    radius: list[float]
    values: list[float]
    label: str = "profile"
    units: str = ""

    @field_validator("radius", "values", mode="before")
    @classmethod
    def coerce_array(cls, v: Any) -> list[float]:
        if isinstance(v, np.ndarray):
            return v.tolist()
        return list(v)

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.radius), np.array(self.values)


class DiagnosticFrame(BaseModel):
    """Single diagnostic snapshot."""

    shot_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    diagnostic: str  # ece, sxr, bes, magnetics
    data: list[list[float]]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        return np.array(self.data, dtype=np.float64)


class PlasmaShot(BaseModel):
    """Full shot record for digital twin replay."""

    shot_id: str
    device: str = "synthetic"
    profiles: dict[str, ProfileData] = Field(default_factory=dict)
    diagnostics: list[DiagnosticFrame] = Field(default_factory=list)
    elm_events: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
