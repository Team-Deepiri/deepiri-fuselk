# Changelog

## [0.2.0] - 2026-06-24

### Added
- Newton-Krylov steady-state oil-water PDE solver and transient integrator
- 4-state extended Kalman filter for O-point phase tracking
- Digital lock-in amplifier and phase-synchronous noise subtraction
- Multi-head helical spiral attention
- PPO vent circularization training via stable-baselines3
- Production ZeroMQ pub/sub diagnostic bus
- HDF5 IMAS shot archive export/import with xarray
- Extended 5-population muon rate network (BDF)
- `scripts/benchmark.py` end-to-end benchmark suite
- CLI: `fuselk benchmark`, `train vent`, `data export/import`, `sim muon`

### Removed
- Tutorial notebooks (replaced by benchmark script)

## [0.1.0] - 2026-06-24

### Added
- Initial deepiri-fuselk monorepo: HELIX, HQRM, Venturi, physics, muon, digital twin
- Control room Dash dashboard and Three.js tokamak viewer
- Experiment registry and lab/reactor runners
- GitHub Actions CI, CodeQL, DeepIRI PR template
