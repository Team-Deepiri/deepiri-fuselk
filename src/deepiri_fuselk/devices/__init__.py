"""Tokamak device profiles: physically-distinct geometry/limit data model.

Exposes `DeviceProfile`, `DischargePreset`, `DeviceRegistry`, and the
module-level `DEFAULT` profile (fuselk's existing implicit/hardcoded
geometry) for wiring into physics modules.
"""

from __future__ import annotations

from deepiri_fuselk.devices.presets import DischargePreset
from deepiri_fuselk.devices.profile import DeviceProfile
from deepiri_fuselk.devices.registry import DEFAULT, DeviceRegistry

__all__ = ["DeviceProfile", "DischargePreset", "DeviceRegistry", "DEFAULT"]
