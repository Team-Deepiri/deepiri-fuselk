"""Device registry: lookup of DeviceProfile + DischargePreset by device name.

Also defines the module-level ``DEFAULT`` DeviceProfile, which encodes
fuselk's *current* implicit/hardcoded geometry so that passing no ``device``
argument anywhere in the physics stack reproduces today's behavior exactly.

Provenance of DEFAULT's values (see final report for full citations):
  - minor_radius_m=1.0, major_radius_m=1.0: the only geometry-shaped constant
    that actually participates in a live physics computation is the safety
    factor profile's minor-radius default ``a=1.0`` in
    ``helix/coordinate_mapper.py:8`` (``q_profile``), reached from
    ``field_line_pitch`` (``helix/coordinate_mapper.py:28``), which is called
    by ``HelixEngine._field_line_pitch_at_o_point``
    (``helix/helix_engine.py:72``). There is no separate major-radius
    constant anywhere in the physics core (the field-line/HQRM math is
    dimensionless/normalized), so major_radius_m=1.0 is chosen to match the
    same unit-normalized convention (aspect ratio 1, i.e. treated as a unit
    torus) rather than encode an unused real-world value.
  - elongation=1.0, triangularity=0.0: no elongation/triangularity term
    exists anywhere in the current physics code (plasma cross-section is
    implicitly circular, unshaped), so these are the identity/no-shaping
    values.
  - troyon_beta_limit=3.0: matches the ``beta_limit`` default in
    ``sim/fusion_kpis.py:54`` (``mhd_stability_margin``), the one genuine
    MHD-stability-limit constant hardcoded in the current codebase.
  - max_ip_ma=1.0, max_bt_t=1.0, max_heating_mw=1.0, greenwald_limit=1.0:
    fuselk's current physics core has *no* hardcoded SI current, field,
    heating-power, or Greenwald-density constant anywhere in the modules
    this task wires (grep turned up none — see final report). These four
    fields are therefore nominal, dimensionless "1 unit" placeholders that
    are not consumed by any DEFAULT-path computation today; they exist so
    the DeviceProfile shape is uniform across DEFAULT/ITER/JET/DIII-D and so
    future wiring has a documented, harmless default to fall back to.
"""

from __future__ import annotations

from deepiri_fuselk.devices.presets import DischargePreset
from deepiri_fuselk.devices.profile import DeviceProfile

DEFAULT = DeviceProfile(
    name="DEFAULT",
    major_radius_m=1.0,
    minor_radius_m=1.0,
    elongation=1.0,
    triangularity=0.0,
    max_ip_ma=1.0,
    max_bt_t=1.0,
    max_heating_mw=1.0,
    greenwald_limit=1.0,
    troyon_beta_limit=3.0,
)

_DEFAULT_PRESETS: list[DischargePreset] = [
    DischargePreset(
        name="H-mode",
        duration_s=1.0,
        ip_waveform_ma=[(0.0, 0.0), (0.2, 1.0), (0.8, 1.0), (1.0, 0.0)],
        heating_waveform_mw=[(0.0, 0.0), (0.3, 1.0), (0.8, 1.0), (1.0, 0.0)],
        density_waveform_frac_greenwald=[(0.0, 0.2), (0.3, 0.6), (1.0, 0.6)],
        elongation_waveform=[(0.0, 1.0), (1.0, 1.0)],
    ),
]


class DeviceRegistry:
    """Lookup table mapping device name -> DeviceProfile and its presets.

    Registers "DEFAULT" plus the real devices defined in
    ``iter.py``/``jet.py``/``diiid.py``.
    """

    def __init__(self) -> None:
        from deepiri_fuselk.devices import diiid, jet
        from deepiri_fuselk.devices import iter as iter_mod

        self._profiles: dict[str, DeviceProfile] = {
            "DEFAULT": DEFAULT,
            iter_mod.ITER.name: iter_mod.ITER,
            jet.JET.name: jet.JET,
            diiid.DIII_D.name: diiid.DIII_D,
        }
        self._presets: dict[str, list[DischargePreset]] = {
            "DEFAULT": _DEFAULT_PRESETS,
            iter_mod.ITER.name: iter_mod.PRESETS,
            jet.JET.name: jet.PRESETS,
            diiid.DIII_D.name: diiid.PRESETS,
        }

    def get(self, name: str) -> DeviceProfile:
        """Return the DeviceProfile registered under ``name``.

        Raises:
            KeyError: if ``name`` is not registered.
        """
        try:
            return self._profiles[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown device '{name}'. Known devices: {self.list_devices()}"
            ) from exc

    def list_devices(self) -> list[str]:
        """Return all registered device names."""
        return list(self._profiles.keys())

    def get_presets(self, device_name: str) -> list[DischargePreset]:
        """Return the discharge presets registered for ``device_name``.

        Raises:
            KeyError: if ``device_name`` is not registered.
        """
        if device_name not in self._presets:
            raise KeyError(f"Unknown device '{device_name}'. Known devices: {self.list_devices()}")
        return self._presets[device_name]
