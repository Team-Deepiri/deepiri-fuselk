# deepiri-fuselk Technology Stack

This document is the canonical reference for libraries, frameworks, and design choices.
fuselk is built to compete with OpenMC, IMAS, and BLUEMIRA — but targets **live plasma control**, not just neutronics, data schemas, or steady-state design.

## Positioning

| Tool | Strength | fuselk advantage |
|------|----------|------------------|
| **OpenMC** | Monte Carlo neutronics | Plasma transients, ELM prediction, RL control loop |
| **IMAS** | Standard data model | ML inference + visualization + actuation on top of IMAS |
| **BLUEMIRA** | Reactor systems design | Real-time disruption dynamics + divertor traffic routing |

## Core Language

- **Python 3.11+** — fusion community standard, rich ML ecosystem
- **Poetry** — dependency management, reproducible installs
- **Optional JAX** — HQRM coordinate maps, differentiable physics (GPU/XLA)
- **Optional PyTorch** — pulled via stable-baselines3 for PPO policies

## Scientific Computing

| Library | Role in fuselk |
|---------|----------------|
| **NumPy / SciPy** | PDE solvers, rate networks (BDF), Kalman tracking |
| **xarray** | Labeled multi-D plasma profiles, IMAS shot bundles |
| **h5py** | HDF5 shot archives (IMAS-compatible roundtrip) |
| **Pydantic v2** | Diagnostic schemas, shot metadata validation |

## Machine Learning & Control

| Library | Role in fuselk |
|---------|----------------|
| **scikit-learn-style logistic** | Trainable ELM predictor (built-in, no sklearn dep) |
| **stable-baselines3** | PPO vent circularization agent |
| **Gymnasium** | `VentCircularizerEnv` RL environment |
| **Bayesian prior module** | Rotational regime classifier (custom, PyMC-compatible path) |

Future integrations (optional groups in `pyproject.toml`):

- **JAX + equinox + optax** — HQRM kernels, autodiff PDEs
- **PyTorch** — spiral attention ViT, Triton inference path

## Real-Time & Infrastructure

| Library | Role in fuselk |
|---------|----------------|
| **ZeroMQ + PyArrow** | Diagnostic pub/sub bus (numpy serialization) |
| **FastAPI + uvicorn** | REST hooks for remote control room |
| **Dash + Plotly** | 6-panel control room dashboard |
| **PySide6 + Qt WebEngine** | Native desktop shell embedding Dash + Three.js |
| **Three.js** | Browser tokamak / field-line viewer |

## Visualization Philosophy

One repo, one control room:

1. **Raw diagnostics** — noisy ECE/SXR heat fields
2. **HELIX output** — denoised focal singularity map
3. **Traffic router** — divertor heat flux arrows
4. **Disruption gauge** — ELM probability + recommended action
5. **Fuel cycle** — TBR, muon gain, Peclet extraction status

## Module → Stack Map

```
data/       xarray, h5py, pydantic
physics/    numpy, scipy (Newton + transient PDE)
helix/      numpy, optional JAX
focal/      numpy (lock-in, spiral attention)
models/     numpy (ELM, disruption detector, Bayesian prior)
control/    gymnasium, stable-baselines3, zmq
muon/       scipy.integrate (rate network ODE)
barrier/    numpy (breeding, vapor, brine coating sim)
sim/        orchestrates all modules
viz/        dash, plotly
experiments/ YAML registry + Python runners
```

## What We Deliberately Avoid

- **Heavy CFD coupling in v1** — SOLPS/BOUT++ are stub wrappers; reduced-order first
- **Proprietary data formats** — IMAS HDF5 is the interchange layer
- **Notebook-only tutorials** — `scripts/benchmark.py` + CLI for reproducible runs

## Install Profiles

```bash
poetry install                    # core: sim + ML + viz
poetry install --with desktop     # PySide6 control room
poetry install --with jax         # GPU HQRM kernels
poetry install --with torch       # explicit torch (SB3 already pulls it)
```

## Benchmark Targets (community reproducibility)

Run `fuselk benchmark` or `python scripts/benchmark.py --all`:

- Oil-water PDE steady convergence + Peclet criterion
- HELIX mean SNR over 20 synthetic frames
- Muon rate network fusions/muon with photon+proton stripping
- Reactor cell fusion score (composite KPI)
- ELM predictor train accuracy on labeled corpus
