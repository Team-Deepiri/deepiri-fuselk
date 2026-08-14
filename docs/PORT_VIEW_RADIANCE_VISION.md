# Port View Radiance: Vision

## The gap this closes

fuselk already computes real physics: HQRM's 7×7 lock state, the 6-field
oil-water barrier PDE, the Venturi hierarchical RL controller's rotation
outputs, per-device field-line pitch (`docs/MULTI_DEVICE_*`). None of that
reaches the eye. The port view's plasma glow, divertor coloring, and
disruption flash are tuned constants — opacity values, color stops, decay
curves chosen because they look right, not because they are derived from
anything the simulation computed. This is exactly the shape of every public
browser-based tokamak simulator we've looked at (fusionsimulator.io included):
a real 0D/scaling-law physics core driving a cosmetic, hand-tuned render.

The port view and the status/trace panels currently describe **two different,
independently-tuned worlds** that happen to look loosely consistent because a
human tuned both by eye. There is no reason, mathematically, that the plasma
glow's brightness has anything to do with `P_rad` in the power-balance panel.
They could silently diverge and nothing would catch it.

## What we're building

A **radiance field** — a genuine physical quantity a camera would actually
measure — computed from fuselk's existing simulation state, and rendered
directly, replacing every hand-tuned visual constant currently in
`tokamak_viewer.html` with a value traceable back to a real physics term:

1. **Bremsstrahlung continuum** (∝ nₑ·nᵢ·√Te) — core glow brightness/color
   driven by real density and temperature, not an opacity slider.
2. **Dα line emission, Doppler-shifted** by real toroidal rotation velocity
   from the Venturi controller — the plasma glow's color temperature reflects
   actual rotation, a physical effect no public simulator bothers to render
   because their underlying physics doesn't track rotation with enough
   fidelity to make it meaningful. fuselk's does.
3. **Divertor blackbody radiation**, mapped from real heat-flux values via
   the actual Planck-law color-temperature curve, replacing the current
   discrete blue→cyan→yellow→red→white lookup table with a continuous,
   physically derived one.
4. **Runaway-electron disruption signature** — a real disruption should show
   a beam-like filament aligned with the device's actual field-line pitch
   (already computed per-device, see `MULTI_DEVICE_IMPLEMENTATION_PLAN.md`
   Phase 1), not a generic radial flash.

## Why this is the genuinely novel part

Prettier shaders are not novel — every simulator has those. What's novel is
**closing the loop so the pretty view and the physics gauges are provably the
same data.** Concretely: integrated rendered radiance over the visible plasma
volume should track `P_rad` from the status panel within a stated tolerance.
That constraint becomes a real automated test, not a vibes check — the first
time (to our knowledge) a browser-embeddable tokamak visualization has made
its render a checkable function of its physics rather than a separately-tuned
illustration of it.

## What this is not

- Not a rewrite of the rendering stack. Three.js stays; we add a radiance
  computation layer (`viz/radiance.py` + shader changes) feeding the existing
  scene, not a new engine.
- Not full spectral/multi-wavelength rendering (no plans for a true spectral
  renderer with dozens of wavelength bins). Three physically-grounded terms
  (continuum, one dominant line, blackbody) is the scope — enough to make
  color and brightness *mean something* without building a spectroscopy
  simulator inside a desktop GUI.
- Not real-time radiative transfer at research-grade accuracy. A ray-marched
  approximation sampling a coarse voxel grid derived from existing field data
  is the target — visually and structurally correct, not a substitute for an
  actual transport code's diagnostic output.

## Success looks like

- Every color and brightness value in the port view has a one-line comment
  pointing to the physics term it comes from — no unexplained magic numbers.
- A test asserts integrated rendered radiance tracks `P_rad` within tolerance
  across at least the ITER/JET/DIII-D device profiles and H-mode/L-mode/
  density-limit presets.
- The plasma glow's shape follows the real flux-surface geometry from the
  equilibrium panel, not a fixed sphere.
- Disruption visuals are recognizably different per device (a runaway beam
  along ITER's field-line pitch looks different from DIII-D's) because they're
  derived from real per-device geometry, not a shared constant.
