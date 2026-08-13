# Multi-Device Control Room: Implementation Plan & Deliverables

Companion to [MULTI_DEVICE_VISION.md](./MULTI_DEVICE_VISION.md) and
[MULTI_DEVICE_ROADMAP.md](./MULTI_DEVICE_ROADMAP.md). This plan maps each
roadmap phase to concrete files and deliverables in the current codebase
(paths below reflect `src/deepiri_fuselk/` as of commit `41ec8d3`).

## Phase 0 — Groundwork

**New module:** `src/deepiri_fuselk/devices/`
- `profile.py` — `DeviceProfile` dataclass: `name`, `major_radius_m`,
  `minor_radius_m`, `aspect_ratio`, `elongation`, `triangularity`,
  `max_ip_ma`, `max_bt_t`, `max_heating_mw`, `greenwald_limit`,
  `troyon_beta_limit`.
- `presets.py` — `DischargePreset` dataclass: `name` (H-mode/L-mode/density
  limit), target waveforms for Ip/heating/density/shaping over the pulse.
- `registry.py` — `DeviceRegistry` (lookup by name, list available).

**Deliverable:** unit tests in `tests/devices/test_registry.py` confirming
the default (current-behavior) device is registered and profile fields
validate (positive radii, limits within physical sanity bounds).

**Audit deliverable:** a short table (in the PR description, not a new doc)
listing every hardcoded geometry/limit constant found in `helix/`, `barrier/`,
`sim/fusion_cell.py`, `sim/reactor_cell.py` that Phase 1 will need to
parameterize.

## Phase 1 — Device-parameterized physics

**Touched files:**
- `src/deepiri_fuselk/sim/fusion_cell.py`, `sim/reactor_cell.py` — accept a
  `DeviceProfile` and read geometry/limits from it instead of module constants.
- `src/deepiri_fuselk/helix/helix_engine.py`, `helix/jax_hqrm.py` — geometry
  inputs sourced from `DeviceProfile`.
- `src/deepiri_fuselk/barrier/heat_exhaust.py` — divertor heat-flux
  calculations parameterized by device.
- New: `devices/iter.py`, `devices/jet.py`, `devices/diiid.py` — concrete
  `DeviceProfile` + preset definitions with real published geometry/limits.

**Deliverable:** `tests/devices/test_physics_parity.py` — same scenario shape
run against all three devices produces distinct, physically ordered outputs
(e.g. confinement time and disruption sensitivity scale with device size/field
in the expected direction). `poetry run pytest tests/devices/` green.

## Phase 2 — Control room device switching

**Touched files:**
- `src/deepiri_fuselk/viz/simulation_engine.py` — `SimulationFrame` gains an
  `active_device: DeviceProfile` field.
- `src/deepiri_fuselk/viz/desktop/panels/physics_panel.py`,
  `sim_lab_panel.py` — device + preset dropdowns, wired to `DeviceRegistry`.
- `src/deepiri_fuselk/viz/dashboard/figures.py` — status/trace panel gauges
  read thresholds from `active_device` instead of fixed constants.

**Deliverable:** `fuselk gui` smoke run (manual, recorded in PR description)
showing device switch changing displayed limits/gauges live.

## Phase 3 — Port view: real-data-driven rendering

**Touched files:**
- `src/deepiri_fuselk/viz/desktop/panels/web_panel.py` — pass
  `SimulationFrame` heat/disruption/ELM state through to the embedded page
  (via `QWebChannel` or periodic `runJavaScript` calls); add `mode` config
  (`'synthetic' | 'live'`).
- `src/deepiri_fuselk/viz/static/tokamak_viewer.html` — consume real heat data
  for glow/coloring; upgrade materials (bloom on hot regions, divertor
  strike-point heat coloring); add clearly-labeled disabled `'live'` mode stub.

**Deliverable:** visual smoke test — screenshot of `fuselk gui` port view
during a running H-mode pulse showing heat-reactive glow (attached to PR),
plus a code comment/doc note on how a future live feed would plug into the
`mode` seam.

## Phase 4 — Polish and parity pass

**Touched files:**
- `viz/dashboard/figures.py` — time-scrubbing on the trace panel.
- `docs/architecture.md`, `README.md` — document device-switching workflow
  and new `devices/` module.

**Deliverable:** updated docs merged; `make lint test` green.

## Cross-cutting

- `make lint test` must stay green after every phase.
- No phase changes the default (no-device-selected) behavior of existing
  scenarios — parity with current output is a hard requirement for Phase 0–1,
  verified by `tests/devices/test_physics_parity.py`'s default-device case.
- Live camera ingestion is explicitly not a deliverable of any phase above
  (see vision doc, "What this is not").
