# Multi-Device Control Room: Roadmap

Companion to [MULTI_DEVICE_VISION.md](./MULTI_DEVICE_VISION.md). Phases are
sequential; each phase should be independently mergeable and leave `fuselk gui`
and `fuselk doctor` green.

## Phase 0 — Groundwork (no behavior change)

- Introduce a `Device` profile type (geometry + engineering limits + presets)
  and a `DeviceRegistry`.
- Audit HELIX, HQRM, the oil-water barrier, Venturi RL, and Peclet extraction
  modules for hardcoded geometry/limit constants that should become
  device-parameterized inputs.
- No UI changes yet. Existing behavior is preserved by registering the current
  implicit machine as the default device.

## Phase 1 — Device-parameterized physics

- Wire `Device` profiles into the physics modules identified in Phase 0.
- Add ITER, JET, DIII-D profiles with real geometry and engineering limits
  (Ip, Bt, heating power ceilings, Greenwald density limit, Troyon beta limit).
- Add H-mode / L-mode / density-limit presets per device.
- Validate: running the same scenario shape against different device profiles
  produces physically distinct, sane outputs (confinement time, beta limits,
  disruption risk scale with device size/field as expected).

## Phase 2 — Control room device switching

- `fuselk gui`: device dropdown + preset dropdown wired to `DeviceRegistry`.
- Status panel, trace panel, and equilibrium panel read device limits for their
  gauges/thresholds instead of fixed constants.
- `SimulationFrame` carries the active device profile so downstream consumers
  (dashboard, port view) don't need a second lookup path.

## Phase 3 — Port view: real-data-driven rendering

- Replace `tokamak_viewer.html`'s fixed glow/heat constants with live
  `SimulationFrame.raw_heat` / `controlled_heat` / disruption/ELM flags.
- Upgrade three.js materials (PBR-ish shading, bloom on hot regions, divertor
  strike-point heat coloring) toward the hyperrealistic look described in the
  vision doc.
- Introduce the `mode: 'synthetic' | 'live'` seam in `web_panel.py` /
  `tokamak_viewer.html` — `'live'` is a no-op stub (clearly labeled
  unavailable) until a real feed exists.

## Phase 4 — Polish and parity pass

- Trace panel: time-scrubbing after a pulse completes, matching the reference
  UX (drag to replay equilibrium + diagnostics at any point in the run).
- Disruption-risk gauge and divertor heat-flux bar styled to read at a glance,
  consistent with the rest of the modernized control room (commit `d455e3e`).
- Update `docs/architecture.md` and `README.md` Quick Start with the new
  device-switching workflow.

## Explicitly out of scope for this roadmap

- Live camera/RTSP ingestion (no source available — tracked as a future item,
  not a phase, until a real feed exists).
- Additional devices beyond ITER/JET/DIII-D (CENTAUR-style speculative designs
  are a possible future phase, not part of this pass).
