# deepiri-fuselk Architecture

## Modules

| Module | Purpose |
|--------|---------|
| `physics/` | Oil-water coupled PDE system, energy balance |
| `barrier/` | Vapor dynamics, tritium breeding blanket |
| `helix/` | Boozer mapping, phase-locked tracker, HQRM |
| `focal/` | Focal heat maps, spiral attention |
| `muon/` | Rate network, photon/proton stripping |
| `control/` | Traffic router, vent RL env, watchdog |
| `sim/` | Digital twin, synthetic data, domain randomizer |
| `devices/` | Device profiles (ITER/JET/DIII-D geometry, limits, presets) |
| `viz/` | Dash dashboard, HELIX/traffic viewers, PySide6 desktop control room |

## Data Flow

```
Diagnostics → HELIX/HQRM → Focal Heat Map → RL Control → Actuators
                    ↓
              Physics PDE ← Breeding Blanket ← Muon Cycle
```

See `docs/theory/` for governing equations. See
`docs/MULTI_DEVICE_VISION.md`, `MULTI_DEVICE_ROADMAP.md`, and
`MULTI_DEVICE_IMPLEMENTATION_PLAN.md` for the device-abstraction effort:
`devices/` profiles feed the same physics stack so `fuselk gui` can switch
between ITER, JET, and DIII-D (with H-mode / L-mode / density-limit presets)
without changing UI code. The desktop shell (`viz/desktop/shell.py`) also
ships an in-app tutorial (`viz/desktop/panels/tutorial_panel.py`, reachable
via Help → Tutorial…) that walks through the equilibrium, trace, status,
and port view panels.
