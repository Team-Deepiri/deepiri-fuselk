# deepiri-fuselk

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

> **Fusion Unified Simulation, ELM Learning & Kinetics** — the open-source autopilot for next-generation tokamaks.

---
<img width="560" height="560" alt="image" src="https://github.com/user-attachments/assets/5fbd2652-c39d-4e87-820e-5efa22a590a5" />


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e8c61803-e23d-4c4b-872e-d38ea81f5aaf" />


The fusion community has OpenMC (neutronics), IMAS (data), and BLUEMIRA (design). None of them solve the hardest live problem: **predicting, controlling, and preventing ELMs and disruptions in real time** while closing the tritium fuel cycle.

**deepiri-fuselk** unifies that entire stack in one repo.

Read the full manifesto: [VISION.md](VISION.md)

## Architecture

```
Diagnostics ──► HELIX Engine ──► Focal Heat Map ──► Venturi Controller ──► Actuators
     │               │                  │                    │
     │         HQRM 7x7 Lock    ELM Predictor         RMP / Pellet / Sweep
     │               │                  │                    │
     └──── Oil-Water Barrier ◄── Tritium Cycle ◄── Muon Recycling
```

| Module | What it does |
|--------|--------------|
| **HELIX** | Phase-locked helical denoising, O-point tracking, focal singularity maps |
| **HQRM** | Recursive rotating quadtree aligned to field line pitch |
| **Venturi** | Hierarchical RL: Bayesian rotational prior + fast vent circularizer |
| **Oil-Water** | Coupled plasma/vapor PDEs, tritium breeding, Peclet extraction |
| **Muon** | Rate network, photon/proton stripping, tritium capsule |
| **Digital Twin** | End-to-end shot replay with IMAS-compatible data |
| **Control Room** | Multi-panel Dash dashboard + Three.js tokamak viewer |

## Quick Start

```bash
git clone https://github.com/Team-Deepiri/deepiri-fuselk.git
cd deepiri-fuselk
poetry install

fuselk doctor
fuselk sim oil-water --steps 200
fuselk sim run --episodes 10
fuselk viz serve   # → http://127.0.0.1:8050
```

## Experiments

| ID | Status | Description |
|----|--------|-------------|
| `elm_helix_denoising` | simulation | HELIX focal precursor tracking |
| `oil_water_barrier` | simulation | Coupled plasma/vapor PDE system |
| `tritium_vapor_extraction` | simulation | Peclet-driven fuel extraction |
| `muon_photon_stripping` | speculative | XFEL + vortex OAM muon reactivation |
| `brine_wall_coating` | speculative | Salinity barrier coating (sim only) |
| `vent_circularization_rl` | simulation | Divertor heat flux symmetrization |
| `plasma_traffic_routing` | simulation | Real-time exhaust congestion routing |

Full catalog: [`experiments/registry.yaml`](experiments/registry.yaml)

## Project Layout

```
src/deepiri_fuselk/
├── data/         IMAS loader, HDF5 store, Pydantic schemas
├── physics/      Oil-water PDE solver, energy balance
├── barrier/      Vapor dynamics, breeding, phase separation, heat exhaust
├── helix/        HELIX engine, HQRM, Boozer map, JAX kernels
├── focal/        Focal heat maps, spiral attention
├── models/       ELM predictor, Bayesian rotational prior, surrogate
├── muon/         Rate network, stripping, tritium capsule
├── control/      Venturi controller, traffic router, RL env, watchdog
├── sim/          Digital twin, SOLPS wrapper, synthetic data
└── viz/          Control room dashboard, Three.js viewer
```

## Development

```bash
make lint && make test && make cov
```

PRs → `dev` branch. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

- **v0.1** — Core math, simulation, dashboard, experiment registry *(current)*
- **v0.2** — JAX GPU kernels, real IMAS shots, offline RL training
- **v0.3** — SOLPS coupling, Triton inference, real-time ZeroMQ bus
- **v1.0** — Community benchmark suite, published ELM predictor

## License

Apache 2.0
