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
| `viz/` | Dash dashboard, HELIX/traffic viewers |

## Data Flow

```
Diagnostics → HELIX/HQRM → Focal Heat Map → RL Control → Actuators
                    ↓
              Physics PDE ← Breeding Blanket ← Muon Cycle
```

See `docs/theory/` for governing equations.
