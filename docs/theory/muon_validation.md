# Muon Trifecta Literature Validation

## Benchmark bands

| Scenario | Literature FPM band | fuselk check |
|----------|---------------------|--------------|
| Collision-only | 105–120 | `R_photon=0, R_proton=0, R_col=0.35` |
| XFEL photon assisted | 148–165 | `R_photon=0.25` |
| Photon + proton trifecta | 180–320 | `R_photon=0.25, R_proton=0.18` |

Breakeven threshold: **284 fusions/muon** (`BREAKEVEN_FUSIONS`).

## Validation (wired into core)

Every `run_stripping_trifecta()` result includes `literature_aligned` and `literature_band`.
Full band audit: `fuselk doctor --vision` or `validate_muon_trifecta()`.

```bash
fuselk doctor --vision
```

Implementation: `muon/literature_bands.py` + `muon/literature_validation.py` + `muon/stripping_orchestrator.py`.

## Trifecta orchestrator

`muon/stripping_orchestrator.py` combines:

- Photon stripping (XFEL / vortex OAM)
- Proton collision stripping
- Cyclotron resonance contribution

Experimental XFEL validation remains future lab work; simulation validates **rate-network consistency** against published schematic curves.
