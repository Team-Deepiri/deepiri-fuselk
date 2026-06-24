# Changelog

## [0.5.1] - 2026-06-24

### Changed
- **VISION alignment in core paths** — rigor checks wired into `FusionCell`, `solve_oil_water_steady`, `HelixEngine`, `run_stripping_trifecta`, and `train_vent_policy` instead of a standalone validation script
- `sim/vision_alignment.py` — single audit mapping VISION.md pillars to implementation
- `fuselk doctor --vision` — module health + VISION pillar table
- Removed `scripts/validate_claims.py` and `fuselk validate claims`

## [0.5.0] - 2026-06-24

### Added
- **Rigor modules** — PDE well-posedness, muon literature bands, RL convergence, JAX HQRM benchmark
- `physics/pde_wellposedness.py` — 6-field existence/uniqueness (contraction L < 1)
- `muon/literature_validation.py` — trifecta rate bands vs schematic literature
- `control/convergence.py` — finite-horizon MDP + discretized value iteration
- `helix/jax_hqrm.py` — JAX HQRM kernel + latency benchmark
- Theory docs: `pde_wellposedness.md`, `muon_validation.md`, `rl_convergence.md`, `hqrm_benchmark.md`

### Changed
- Muon rate network: catalytic recycling + literature-calibrated yield
- Default `PDEParameters.alpha` tuned; `PDEParameters.certified()` for uniqueness proof

## [0.4.0] - 2026-06-24

### Added
- **Live simulation dashboard** — FusionCell steps in real time with KPI strip, disruption gauge, traffic map
- **`LiveSimulation`** engine + Plotly figure builders for control room
- **`FusionCell`** top-level orchestrator (reactor + oil-water TBR + brine coating + muon trifecta)
- **`stripping_orchestrator`** — unified photon + proton + cyclotron muon recycling
- **Experiment runner** — `fuselk experiments list/run`, YAML registry loaders
- **`docs/STACK.md`** — technology stack vs OpenMC/IMAS/BLUEMIRA
- CLI: `fuselk sim fusion`, `fuselk viz sim`, `fuselk experiments`

### Changed
- Dashboard dark theme, reset button, configurable refresh interval, 3D tokamak viewer link
- Typer upgraded for reliable CLI options; PyYAML added for experiment registry

## [0.3.0] - 2026-06-24

### Added
- **Jupyter notebook suite** — six vision/math/experiment notebooks under `notebooks/` (HELIX, oil–water PDEs, muon cycle, Venturi control, ReactorCell digital twin)
- Jupyter dev dependencies (`jupyter`, `ipykernel`, `matplotlib`, `nbformat`)
- **FusionKPIs** composite progress score for reproducible benchmarking
- **Trainable ELMPredictor** — logistic regression on labeled synthetic shot corpus with save/load
- **DisruptionDetector** — fuses HELIX, ELM, and MHD stability into actuation recommendations
- **Shot corpus generator** for supervised ELM training
- CLI: `fuselk reactor run`, `fuselk reactor score`, `fuselk train elm`, `fuselk sim reactor`
- Benchmark suite: `--reactor`, `--elm` targets
- Restored Jupyter notebook tutorials under `notebooks/` (replaces v0.2 benchmark-only workflow)

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
- Tutorial notebooks (replaced by benchmark script) — **restored in v0.3** under `notebooks/`

## [0.1.0] - 2026-06-24

### Added
- Initial deepiri-fuselk monorepo: HELIX, HQRM, Venturi, physics, muon, digital twin
- Control room Dash dashboard and Three.js tokamak viewer
- Experiment registry and lab/reactor runners
- GitHub Actions CI, CodeQL, DeepIRI PR template
