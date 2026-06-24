# deepiri-fuselk Vision

**Fusion Unified Simulation, ELM Learning & Kinetics**

We are building the open-source autopilot for next-generation tokamaks — not another data schema, not another steady-state design tool, but a **live, intelligent, end-to-end fusion control platform**.

## The Problem

Tokamaks face three existential bottlenecks:

1. **ELMs & disruptions** — sudden instabilities that terminate pulses and damage hardware
2. **Divertor melt** — localized heat exhaust that destroys tungsten tiles
3. **Tritium self-sufficiency** — no commercial fusion without a closed fuel cycle

Existing tools address pieces. None unify them.

## The fuselk Answer

```
Raw Diagnostics ──► HELIX Engine ──► Focal Heat Map ──► Venturi Controller ──► Actuators
       │                  │                  │                    │
       │            HQRM 7x7 Lock     ELM Probability      RMP / Pellet / Sweep
       │                  │                  │                    │
       └────────── Oil-Water Barrier ◄── Tritium Cycle ◄── Muon Recycling
```

### HELIX — Helical Extraction & Locked-In Isotropy eXclusion

Phase-locked noise cancellation that tracks the rotating magnetic island O-point, unwraps the helix into Boozer coordinates, and produces a crystalline focal heat map — the "telescope for the plasma's soul."

### HQRM — Helical Quadtree Rotational Mapper

Recursive rotating squares that align to field line pitch, subdivide on magnetic shear, and lock a 7x7 kernel when variance drops below 7%. Solves the differential geometry problem in <1ms on GPU via JAX.

### Venturi Controller

Hierarchical hybrid RL:
- **Slow loop (~100ms):** Bayesian neural net with rotational axis awareness sets policy priors
- **Fast loop (~1ms):** PPO agent circularizes divertor exhaust via strike-point sweeping
- **Watchdog:** PID safety override when engineering limits breach

### Oil-Water Barrier

Coupled plasma/vapor PDEs modeling phase-separated edge transport, tritium breeding in lithium vapor, and Peclet-driven fuel extraction — our own mathematical framework for self-sustaining fuel cycles.

### Muon Recycling Trifecta

Photon stripping (XFEL + vortex OAM), proton collision stripping, and cyclotron resonance — targeting N_fus,μ > 284 for catalytic breakeven.

## How We Beat OpenMC / IMAS / BLUEMIRA

| Tool | Scope | fuselk advantage |
|------|-------|------------------|
| OpenMC | Neutronics | We own plasma transients + real-time control |
| IMAS | Data standard | We use IMAS but add inference + visualization + RL |
| BLUEMIRA | Reactor design | We target live disruption dynamics + digital twin |

## Roadmap

- **v0.1 (now):** Core math, simulation, dashboard, experiment registry
- **v0.2:** JAX GPU kernels, IMAS shot loader, offline RL on DIII-D synthetic data
- **v0.3:** SOLPS/BOUT++ coupling, real-time bus with ZeroMQ, Triton inference
- **v1.0:** Community-validated ELM predictor, published benchmark suite

## Join Us

This is frontier science made open. PRs welcome → `dev` branch.
