# Port View Radiance: Roadmap

Companion to [PORT_VIEW_RADIANCE_VISION.md](./PORT_VIEW_RADIANCE_VISION.md).
Builds on the multi-device control room work
([MULTI_DEVICE_VISION.md](./MULTI_DEVICE_VISION.md),
[MULTI_DEVICE_ROADMAP.md](./MULTI_DEVICE_ROADMAP.md)) — this assumes
`devices/` (ITER/JET/DIII-D profiles) and the real-data-driven port view from
that work are already merged. Each phase should leave `fuselk gui` and
`fuselk doctor` green and be independently mergeable.

## Phase 5 — Radiance core (Python side, no rendering changes yet)

- New module `src/deepiri_fuselk/viz/radiance.py`:
  - `bremsstrahlung_intensity(n_e, n_i, T_e) -> float` — continuum emission,
    physically standard ∝ nₑ·nᵢ·√Te form.
  - `dalpha_doppler_shift(rotation_velocity_m_s, rest_wavelength_nm=656.3) -> float`
    — relativistic-safe Doppler shift of the Dα line from real toroidal
    rotation (sourced from the Venturi controller's rotation output).
  - `blackbody_color_temperature(heat_flux_mw_m2) -> (float, float, float)` —
    continuous Planck-law RGB mapping replacing the discrete heat-color
    lookup table currently in `tokamak_viewer.html`.
  - `radiance_field(frame: SimulationFrame) -> RadianceField` — composes the
    above into a small structured object (core intensity/color, divertor
    per-plate color, line-shift factor) ready to hand to the renderer.
- `tests/viz/test_radiance.py`: unit tests for each function against known
  limiting cases (e.g. `T_e -> 0` gives near-zero continuum; zero rotation
  gives zero Doppler shift; known heat-flux values reproduce expected color
  bands at the boundaries of the old discrete lookup table, so the switch is
  a visually smooth superset of current behavior, not a regression).

**Deliverable:** `radiance.py` computing real physical values from existing
`SimulationFrame` fields, fully unit-tested, not yet wired into the viewer.

## Phase 6 — Conservation check

- `tests/viz/test_radiance_conservation.py`: integrate `radiance_field(...)`
  core intensity over an approximated plasma volume (use the device's
  major/minor radius and elongation/triangularity from `DeviceProfile` for
  the volume estimate) and assert it tracks `frame`'s `P_rad`
  (power-balance panel's radiated power) within a stated tolerance —
  across all three device profiles and all three presets.
- If the check fails outside tolerance, this phase includes tuning the
  physical constants in `radiance.py` (not adding unprincipled fudge
  factors) until it passes, or explicitly documenting why exact tracking
  isn't achievable at this fidelity and what tolerance is actually honest.

**Deliverable:** a green conservation test plus an explicit, documented
tolerance and its justification (this is the "provably the same data"
guarantee from the vision doc — treat a failing or hand-waved tolerance as
a blocker, not a nice-to-have).

## Phase 7 — Wire into the renderer

- `src/deepiri_fuselk/viz/desktop/panels/web_panel.py`: extend the existing
  frame-push path (already streaming `SimulationFrame` data per the
  multi-device work) to also push the `RadianceField` output.
- `src/deepiri_fuselk/viz/static/tokamak_viewer.html`:
  - Replace the fixed glow-shell color/opacity constants with values read
    from the pushed radiance data.
  - Replace the discrete `HEAT_STOPS` divertor color ramp with the
    continuous blackbody mapping.
  - Add a Doppler color-temperature tint to the plasma glow driven by the
    line-shift factor.
- Every remaining color/brightness constant in the file that *isn't* wired
  to radiance data should get a one-line comment explaining why it's a
  legitimate rendering choice (e.g. ambient light color) rather than a
  physics stand-in — per the vision doc's "no unexplained magic numbers"
  success criterion.

**Deliverable:** visual smoke test (screenshot per device/preset combination,
9 total) showing distinct, physically-motivated brightness/color differences.

## Phase 8 — Volumetric shape from the equilibrium field

- Currently the plasma glow is a fixed sphere. Use the same flux-surface
  geometry already computed for the equilibrium panel (elongation κ,
  triangularity δ, per-device from `MULTI_DEVICE_*`) to shape the glow mesh
  — an elongated/triangular cross-section swept around the torus, not a
  sphere — so ITER/JET/DIII-D visibly differ in plasma shape, not just
  color.
- Keep this a mesh-deformation change, not a full ray-marched volume
  renderer (see vision doc's explicit non-goals) — deform existing
  three.js geometry based on device shape parameters.

**Deliverable:** screenshot comparison showing distinctly-shaped plasma
columns across the three devices.

## Phase 9 — Physically-real disruption signature

- Replace the generic radial disruption flash with a beam-like filament
  effect aligned with the device's real field-line pitch (already computed
  in `helix/coordinate_mapper.py` per the multi-device work), fired on the
  same `disruption_probability` threshold as today.
- Confirm the filament direction/appearance visibly differs between
  low-field DIII-D and high-field ITER, consistent with the "start small,
  work up to ITER-scale" strategy framing from the tutorial.

**Deliverable:** screenshot/recording showing the disruption effect on at
least two devices, visibly distinct.

## Explicitly out of scope for this roadmap

- Full multi-wavelength spectral rendering (see vision doc non-goals).
- Research-grade radiative transfer / Monte Carlo photon transport.
- Any change to the underlying physics solvers themselves — this roadmap
  only adds a *visualization* layer reading from them.

## Cross-cutting

- `make lint test` must stay green after every phase.
- No phase changes default (no-device-selected) simulation output — only
  the port-view rendering path changes.
- Every new color/brightness constant introduced must trace to a physical
  term or be explicitly commented as a deliberate rendering choice.
