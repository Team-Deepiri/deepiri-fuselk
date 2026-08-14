# Multi-Device Control Room: Vision

## Why

fuselk's physics pillars (HELIX, HQRM, the 6-field oil-water barrier PDE, the Muon
recycling trifecta, the Venturi hierarchical RL controller, Peclet tritium
extraction) are validated against a single implicit reactor geometry. There is no
notion of "which device" anywhere in `src/` — no per-device geometry, no
per-device operational limits, no way to point the same physics stack at a
different machine.

Public browser-based simulators (e.g. fusionsimulator.io) have shown that a
single physics core cleanly parameterized by device (DIII-D, JET, ITER, and even
speculative designs) is both pedagogically powerful and technically
straightforward when the underlying physics is 0D/scaling-law grade. fuselk's
ambition is higher-fidelity than that — but the *organizing idea* (device as a
first-class, swappable parameter set) is worth adopting regardless of fidelity
level, and the control-room UI patterns those tools use (synchronized time
traces, scrubbing, a live equilibrium cross-section, a port/camera view into the
vessel) are proven UX for exactly what fuselk's desktop shell is trying to be.

## What we're building

1. **Device abstraction.** ITER, JET, and DIII-D as first-class device profiles —
   geometry (major/minor radius, aspect ratio, elongation, triangularity),
   engineering limits (max Ip, Bt, heating power, Greenwald density limit,
   Troyon beta limit), and default discharge presets (H-mode, L-mode, density
   limit) — sitting behind one shared physics interface so existing fuselk
   modules (HELIX, HQRM, the oil-water barrier, Venturi RL) become
   device-parameterized instead of hardcoded to one implicit machine.

2. **Real-time transport physics per device.** Each device profile drives fuselk's
   existing physics engine (not a separate toy model) — 0D power balance,
   confinement scaling, pedestal/ELM dynamics, disruption risk — so that
   switching devices changes the *numbers that come out of real fuselk physics*,
   not a cosmetic skin.

3. **Hyperrealistic port view.** The current `tokamak_viewer.html` three.js panel
   is procedural and driven by fixed constants. It should instead be driven by
   the live `SimulationFrame` (`raw_heat`, `controlled_heat`, HQRM state,
   disruption/ELM flags) so the glow, divertor heat coloring, and vessel
   rendering reflect what the simulation is actually doing — with a
   `mode: 'synthetic' | 'live'` seam left in place so a real camera/RTSP feed
   can be substituted later if and when an actual video source exists. We are
   not committing to live camera ingestion now — no such source is available —
   but we are not building ourselves into a corner either.

## What this is not

- Not a rewrite of fuselk's physics fidelity down to 0D scaling laws. The
  device layer parameterizes the *existing* higher-fidelity modules; it does not
  replace HQRM or the 6-field PDE with cheaper approximations.
- Not a live-camera feature yet. That requires an actual video source
  (a real reactor test cell, a partner facility, or a webcam proxy) that does
  not currently exist. The seam is built; the feed is not.
- Not a new frontend framework. The existing PySide6 + QtWebEngine +
  three.js + Dash/Plotly stack stays; this is additive within it.

## Success looks like

- `fuselk gui` lets you pick a device (ITER / JET / DIII-D) and a preset
  (H-mode / L-mode / density limit), and the equilibrium, trace panel, status
  panel, and port view all respond with device-appropriate numbers and limits.
- The port view's visual state (glow intensity, divertor coloring, disruption
  flash) is derived from the same `SimulationFrame` data the Plotly dashboard
  already consumes — no divergent data path.
- Swapping in a live video feed later is a config change, not a rewrite.
