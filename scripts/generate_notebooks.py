#!/usr/bin/env python3
"""Generate fuselk vision / math / experiment Jupyter notebooks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

KERNEL = {
    "kernelspec": {
        "display_name": "Python 3 (fuselk)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def nb(*cells: dict) -> dict:
    return {"cells": list(cells), "metadata": KERNEL, "nbformat": 4, "nbformat_minor": 5}


SETUP = """\
import sys
from pathlib import Path

repo = Path.cwd()
if not (repo / "src" / "deepiri_fuselk").exists() and (repo.parent / "src" / "deepiri_fuselk").exists():
    repo = repo.parent

try:
    import deepiri_fuselk  # noqa: F401
except ImportError:
    sys.path.insert(0, str(repo / "src"))
    import deepiri_fuselk  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")
%config InlineBackend.figure_formats = ["retina"]
"""


def notebook_00() -> dict:
    return nb(
        md(
            """# fuselk Vision Overview

**Fusion Unified Simulation, ELM Learning & Kinetics** — the open-source autopilot for next-generation tokamaks.

This notebook maps the [VISION.md](../VISION.md) architecture to runnable code. Companion notebooks cover each pillar in depth:

| Notebook | Topic |
|----------|-------|
| `01_oil_water_barrier_math` | Coupled plasma/vapor PDEs, Peclet extraction |
| `02_helix_hqrm_diagnostics` | Phase-locked denoising, HQRM 7×7 lock |
| `03_muon_tritium_fuel_cycle` | Catalytic rate network, breeding blanket |
| `04_venturi_control_experiments` | Hierarchical RL, disruption fusion |
| `05_reactor_cell_digital_twin` | Closed-loop ReactorCell + FusionKPIs |
| **`06_unified_cross_domain_mathematics`** | **Full VISION equation sets, proofs, domain coupling** |

## The four existential bottlenecks (VISION.md)

1. **ELMs & disruptions** — MHD precursors terminate pulses
2. **Divertor melt** — localized exhaust (>10 MW/m²)
3. **Alpha sticking** — μ⁻ lost to (αμ)⁺ before catalytic breakeven
4. **Tritium self-sufficiency** — closed fuel cycle requires TBR > 1 and Pe > 1 extraction

> **Start here for theory:** `06_unified_cross_domain_mathematics.ipynb` lays out every governing equation, proof sketch, and cross-domain coupling from [VISION.md](../VISION.md). Notebooks 01–05 pair that theory with runnable experiments.
"""
        ),
        md(
            """## End-to-end data flow

```
Diagnostics → HELIX/HQRM → Focal Heat Map → ELM Predictor → Venturi Controller → Actuators
                    ↓
              Physics PDE ← Breeding Blanket ← Muon Cycle
```

Each module is a Python package under `src/deepiri_fuselk/`. Experiments are registered in `experiments/registry.yaml`.
"""
        ),
        code(SETUP + "\nimport deepiri_fuselk as fuselk\nprint(f'fuselk v{fuselk.__version__}')"),
        code(
            """\
registry_path = Path("experiments/registry.yaml")
if not registry_path.exists():
    registry_path = Path("../experiments/registry.yaml")

for line in registry_path.read_text().splitlines():
    if line.strip().startswith("- id:"):
        exp_id = line.split(":", 1)[1].strip()
        print(f"  • {exp_id}")
"""
        ),
        md("## Quick smoke test — all pillars in one cell"),
        code(
            """\
from deepiri_fuselk.helix import HelixEngine
from deepiri_fuselk.muon import RateNetworkParams, run_rate_network
from deepiri_fuselk.physics import PDEParameters, peclet_criterion, solve_oil_water_steady
from deepiri_fuselk.sim import ReactorCell, generate_ece_shot

params = PDEParameters()
steady = solve_oil_water_steady(n_grid=32, params=params)
shot = generate_ece_shot(24, seed=0)
helix = HelixEngine().process(shot.heat_field, shot.raw_signal, shot.angles)
muon = run_rate_network(params=RateNetworkParams(R_photon=0.3), t_span=(0.0, 1e-5))
cell = ReactorCell(grid_size=16, train_elm=False)
run = cell.run(n_steps=5, seed=1)

print(f\"PDE converged: {steady.converged}, Peclet Pe = {peclet_criterion(params):.2f}\")
print(f\"HELIX SNR: {helix.phase_locked_snr:.2f}, ELM prob: {helix.elm_probability:.2f}\")
print(f\"Muon fusions/μ: {muon.fusions_per_muon:.1f}, breakeven: {muon.breakeven}\")
print(f\"Reactor fusion score: {run.final_score:.3f}\")
"""
        ),
    )


def notebook_01() -> dict:
    return nb(
        md(
            """# Oil–Water Barrier: Governing Math

The **oil–water barrier** models phase-separated edge transport: plasma density $n_p$ ("oil") and lithium vapor $n_v$ ("water") coupled at a moving interface, with tritium $n_T$ bred in the vapor blanket.

## Steady-state coupled system

$$D_p \\frac{d^2 n_p}{dx^2} = \\alpha\\, n_p n_v$$

$$D_v \\frac{d^2 n_v}{dx^2} - v_v \\frac{dn_v}{dx} = -\\alpha\\, n_p n_v$$

Tritium obeys an advection–diffusion equation with Li breeding source $\\sigma_{Li}\\,\\Phi_n\\, n_v$.

## Peclet extraction criterion

$$\\mathrm{Pe} = \\frac{v_v \\,\\delta}{D_T}, \\quad \\delta = \\left(\\frac{D_p D_v}{\\alpha^2 n_0 n_{wall}}\\right)^{1/4}$$

When $\\mathrm{Pe} > 1$, outward vapor sweep dominates diffusion — tritium is extracted at the wall.

---

## Full six-field oil–water system (VISION §3)

fuselk's **novel contribution** is a coupled multi-physics edge model — not a single diffusion equation. In steady state along the radial coordinate $x$ (core → wall):

| Field | Governing equation |
|-------|-------------------|
| Plasma density $n_p$ | $\\displaystyle\\frac{d}{dx}\\!\\left(D_p \\frac{dn_p}{dx}\\right) - \\alpha\\, n_p n_v - \\sigma_{strip}\\, n_p \\Phi_{ext} = 0$ |
| Vapor density $n_v$ | $\\displaystyle\\frac{d}{dx}\\!\\left(D_v \\frac{dn_v}{dx}\\right) - v_v \\frac{dn_v}{dx} + \\alpha\\, n_p n_v - S_{breed}(x) = 0$ |
| Tritium $n_T$ | $\\displaystyle\\frac{d}{dx}\\!\\left(-D_T \\frac{dn_T}{dx} + v_v n_T\\right) = \\sigma_{Li}\\, \\Phi_n\\, n_v(x) + S_\\mu(x)$ |
| Muon density $n_\\mu$ | $\\displaystyle\\frac{d}{dx}\\!\\left(D_\\mu \\frac{dn_\\mu}{dx}\\right) - v_\\mu \\frac{dn_\\mu}{dx} + R_{strip}(x) - \\lambda_\\mu n_\\mu = 0$ |
| Plasma energy $T_p$ | $\\displaystyle\\frac{d}{dx}\\!\\left(\\kappa_p \\frac{dT_p}{dx}\\right) - P_{brem} - \\beta\\,\\alpha\\, n_p n_v T_p - P_\\mu = 0$ |
| Vapor energy $T_v$ | $\\displaystyle\\frac{d}{dx}\\!\\left(\\kappa_v \\frac{dT_v}{dx}\\right) - v_v \\rho_v C_p \\frac{dT_v}{dx} + \\beta\\,\\alpha\\, n_p n_v T_p - P_{rad} + Q_{laser} = 0$ |

**Interface thickness** (mutual conversion scale):
$$\\delta = \\left(\\frac{D_p D_v}{\\alpha^2 n_0 n_w}\\right)^{1/4}$$

The reduced solver in `physics/pde_solver.py` implements the $(n_p, n_v)$ subsystem + linear tritium sub-solve; energy residuals live in `physics/energy_balance.py`.

### Proof sketch: Peclet extraction (Pe > 1 ⇒ outward tritium flux)

Consider tritium on $x \\in [0,L]$ with $n_T(0)$ fixed and wall sink at $x=L$. Steady advection–diffusion:
$$-D_T n_T'' + v_v n_T' = S(x), \\quad S \\ge 0$$

Multiply by integrating factor and integrate. When $v_v \\delta / D_T > 1$, boundary-layer theory places the tritium peak away from the core; the wall flux
$$\\Gamma_T = -D_T n_T'(L) + v_v n_T(L)$$
is **advection-dominated** at the wall. Hence $\\mathrm{Pe} = v_v \\delta / D_T > 1$ is the **necessary scale condition** for outward fuel extraction rather than core reflux — the key fuselk fuel-cycle closure criterion that OpenMC/IMAS do not model.
"""
        ),
        code(SETUP),
        code(
            """\
from deepiri_fuselk.barrier.breeding_blanket import evaluate_breeding_blanket, tritium_breeding_ratio
from deepiri_fuselk.physics import (
    PDEParameters,
    interface_thickness,
    peclet_criterion,
    solve_oil_water_steady,
    solve_oil_water_transient,
)

params = PDEParameters()
delta = interface_thickness(params)
pe = peclet_criterion(params)

print(f\"Interface thickness δ = {delta:.4f}\")
print(f\"Peclet number Pe = {pe:.3f}  ({'outward sweep wins' if pe > 1 else 'diffusion dominates'})\")

result = solve_oil_water_steady(n_grid=128)
state = result.state
tbr = tritium_breeding_ratio(state, params)
blanket = evaluate_breeding_blanket(state, params)

print(f\"Newton converged: {result.converged}, residual = {result.residual:.2e}\")
print(f\"TBR = {tbr:.3f}, extraction efficiency = {blanket.extraction_efficiency:.2f}\")
"""
        ),
        code(
            """\
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].plot(state.x, state.n_p, label=r\"$n_p$ plasma\", color=\"C0\")
axes[0].plot(state.x, state.n_v, label=r\"$n_v$ vapor\", color=\"C1\")
axes[0].set_xlabel(\"x\"); axes[0].set_ylabel(\"density\"); axes[0].legend(); axes[0].set_title(\"Steady profiles\")

axes[1].plot(state.x, state.n_T, color=\"C2\")
axes[1].set_xlabel(\"x\"); axes[1].set_ylabel(r\"$n_T$\")
axes[1].set_title(\"Tritium distribution\")

axes[2].plot(state.x, state.T_p, label=r\"$T_p$\")
axes[2].plot(state.x, state.T_v, label=r\"$T_v$\")
axes[2].set_xlabel(\"x\"); axes[2].legend(); axes[2].set_title(\"Temperature proxies\")

plt.tight_layout()
plt.show()
"""
        ),
        md("## Experiment: Peclet sweep vs. breeding ratio"),
        code(
            """\
vapor_speeds = np.linspace(0.1, 1.2, 15)
pe_values, tbr_values = [], []

for v_v in vapor_speeds:
    p = PDEParameters(v_v=v_v)
    res = solve_oil_water_steady(n_grid=64, params=p)
    pe_values.append(peclet_criterion(p))
    tbr_values.append(tritium_breeding_ratio(res.state, p))

fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(pe_values, tbr_values, \"o-\", color=\"C3\", label=\"TBR\")
ax1.axvline(1.0, color=\"gray\", ls=\"--\", label=\"Pe = 1\")
ax1.set_xlabel(\"Peclet number Pe\"); ax1.set_ylabel(\"Tritium breeding ratio\")
ax1.set_title(\"Fuel-cycle sensitivity to vapor sweep\")
ax1.legend()
plt.tight_layout()
plt.show()
"""
        ),
        md("## Energy-balance residuals (full six-field system)"),
        code(
            r"""\
from deepiri_fuselk.physics.energy_balance import plasma_energy_residual, vapor_energy_residual

res_p = plasma_energy_residual(state, params)
res_v = vapor_energy_residual(state, params)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(state.x, res_p, label=r"plasma energy residual $\mathcal{R}_p$")
ax.plot(state.x, res_v, label=r"vapor energy residual $\mathcal{R}_v$")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("x"); ax.legend(); ax.set_title("Energy equation residuals on steady PDE state")
plt.tight_layout()
plt.show()
"""
        ),
        md("## Transient evolution"),
        code(
            """\
history = solve_oil_water_transient(n_grid=64, t_end=0.5, dt=0.005)
times = np.arange(len(history)) * 0.005

fig, ax = plt.subplots(figsize=(8, 4))
for i in [0, len(history) // 4, len(history) // 2, -1]:
    ax.plot(history[i].x, history[i].n_v, label=f\"t = {times[i]:.2f}\")
ax.set_xlabel(\"x\"); ax.set_ylabel(r\"$n_v$\")
ax.set_title(\"Vapor density transient\")
ax.legend()
plt.tight_layout()
plt.show()
"""
        ),
    )


def notebook_02() -> dict:
    return nb(
        md(
            """# HELIX & HQRM: Diagnostic Math

**HELIX** (Helical Extraction & Locked-In Isotropy eXclusion) phase-locks to the rotating magnetic island O-point, unwraps Boozer coordinates, and produces a focal singularity heat map.

**HQRM** (Helical Quadtree Rotational Mapper) recursively subdivides squares aligned to field-line pitch and locks a 7×7 kernel when local variance drops below 7%.

## Lock-in demodulation

For reference phase $\\phi_0$ and poloidal angles $\\theta_i$:

$$I = \\frac{1}{N}\\sum_i s_i \\cos(\\theta_i - \\phi_0), \\quad Q = \\frac{1}{N}\\sum_i s_i \\sin(\\theta_i - \\phi_0)$$

$$A = \\sqrt{I^2 + Q^2}$$

Phase-synchronous averaging over $N_{cycles}$ rotations subtracts incoherent turbulence.

---

## Extended Kalman O-point tracker (4-state EKF)

State vector $\\mathbf{x} = [\\theta,\\, \\omega,\\, A,\\, \\phi]^\\top$ tracks poloidal phase, rotation rate, island amplitude, and toroidal phase.

**Prediction** (discrete, step $\\Delta t$):
$$\\mathbf{x}_{k|k-1} = F(\\Delta t)\\, \\mathbf{x}_{k-1}, \\quad F = \\begin{pmatrix} 1 & \\Delta t & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\\\ 0 & 0 & 1 & 0 \\\\ 0 & \\Delta t/2 & 0 & 1 \\end{pmatrix}$$

**Measurement** $z = [\\theta_{meas}, \\phi_{meas}, A_{meas}]^\\top$ with $H$ selecting $(\\theta, A, \\phi)$.

**Update** (standard EKF): $K = P H^\\top (H P H^\\top + R)^{-1}$, $\\mathbf{x} \\leftarrow \\mathbf{x} + K(z - H\\mathbf{x})$.

Phase wrapping on $\\theta, \\phi$ enforces toroidal periodicity — this is the **abstract filtering** layer that makes lock-in demodulation optimal for a single rotating coherent mode.

## Boozer unwrap & field-line pitch

Given minor radius $r$ and safety factor $q(r)$:
$$\\phi_{Boozer} = \\frac{\\theta}{q(r)}, \\qquad \\text{pitch}(r) = \\arctan\\!\\left(\\frac{B_\\theta}{B_\\phi}\\right) \\approx \\arctan\\!\\left(\\frac{1}{q(r)}\\right)$$

HQRM rotates each quadtree square by this pitch — **differential geometry on a tokamak flux surface**, not a Cartesian CNN.

## HQRM convergence criterion (7×7 variance lock)

Recursive split while magnetic shear $|dq/dr| > \\tau_{shear}$. At depth 7, collect 49 leaf squares $\\{S_i\\}$. Define heat samples $h_i$. Lock when:
$$\\mathrm{Var}(h_1,\\ldots,h_{49}) < 0.07$$

**Interpretation:** Monte-Carlo dropout across 49 field-aligned patches — variance below 7% certifies a **crystalline O-point lock** (VISION novel contribution in computational geometry + signal processing).
"""
        ),
        code(SETUP),
        code(
            """\
from deepiri_fuselk.focal.lockin_amplifier import lockin_demodulate, subtract_incoherent_noise
from deepiri_fuselk.helix import HelixEngine, run_hqrm
from deepiri_fuselk.sim import generate_ece_shot

shot = generate_ece_shot(size=48, seed=7, island_amplitude=0.7)
engine = HelixEngine(rotation_hz=5000.0)
result = engine.process(shot.heat_field, shot.raw_signal, shot.angles)

amp, i_comp, q_comp = lockin_demodulate(
    shot.raw_signal, result.o_point[0], shot.angles
)

print(f\"Lock-in amplitude: {amp:.3f}  (I={i_comp:.3f}, Q={q_comp:.3f})\")
print(f\"Phase-locked SNR: {result.phase_locked_snr:.2f}\")
print(f\"HQRM converged: {result.hqrm.converged}, variance: {result.hqrm.heat_variance:.4f}\")
print(f\"O-point: {result.o_point}, fracture vector: {result.fracture_vector}\")
"""
        ),
        code(
            """\
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

im0 = axes[0].imshow(shot.heat_field, origin=\"lower\", cmap=\"inferno\")
axes[0].set_title(\"Raw ECE heat field\")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

im1 = axes[1].imshow(result.focal_map, origin=\"lower\", cmap=\"magma\")
axes[1].set_title(\"HELIX focal map\")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

axes[2].plot(shot.angles, shot.raw_signal, alpha=0.5, label=\"raw\")
cleaned = subtract_incoherent_noise(shot.raw_signal, shot.angles, engine.rotation_hz)
axes[2].plot(shot.angles, cleaned, label=\"phase-locked avg\")
axes[2].set_xlabel(\"poloidal angle\"); axes[2].legend()
axes[2].set_title(\"Lock-in noise subtraction\")

plt.tight_layout()
plt.show()
"""
        ),
        md("## Experiment: SNR vs. island amplitude"),
        code(
            """\
amplitudes = np.linspace(0.1, 1.0, 12)
snrs = []

for amp in amplitudes:
    s = generate_ece_shot(32, seed=42, island_amplitude=amp)
    r = HelixEngine().process(s.heat_field, s.raw_signal, s.angles)
    snrs.append(r.phase_locked_snr)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(amplitudes, snrs, \"s-\", color=\"C4\")
ax.set_xlabel(\"Island amplitude\"); ax.set_ylabel(\"Phase-locked SNR\")
ax.set_title(\"HELIX denoising gain vs. precursor strength\")
plt.tight_layout()
plt.show()
"""
        ),
        md("## HQRM quadtree on synthetic field"),
        code(
            """\
n = 64
grid = np.linspace(-1, 1, n)
X, Y = np.meshgrid(grid, grid)
heat = np.exp(-((X - 0.2) ** 2 + (Y + 0.1) ** 2) / 0.08)
hqrm = run_hqrm(heat, variance_threshold=0.07)

print(f\"HQRM kernel squares: {len(hqrm.kernel)}, converged: {hqrm.converged}\")
print(f\"Heat variance: {hqrm.heat_variance:.4f}, O-point: {hqrm.o_point}\")

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(heat, origin=\"lower\", extent=[-1, 1, -1, 1], cmap=\"inferno\", alpha=0.8)
for sq in hqrm.kernel:
    ax.plot(
        [sq.x_min, sq.x_max, sq.x_max, sq.x_min, sq.x_min],
        [sq.y_min, sq.y_min, sq.y_max, sq.y_max, sq.y_min],
        \"w-\", lw=0.6, alpha=0.7,
    )
ax.plot(*hqrm.o_point, \"c*\", ms=12, label=\"O-point\")
ax.set_title(\"HQRM 7×7 kernel lock\")
ax.legend()
plt.tight_layout()
plt.show()
"""
        ),
        md("## Kalman filter matrices (implemented in `kalman_tracker.py`)"),
        code(
            """\
from deepiri_fuselk.helix.kalman_tracker import PhaseLockedTracker

tracker = PhaseLockedTracker(omega_init=5000.0)
dt = 1e-4
F = tracker._F(dt)
H = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

print(\"State transition F(Δt):\")
print(F)
print(\"\\nMeasurement matrix H:\")
print(H)
print(f\"\\nProcess noise diag Q: {np.diag(tracker.Q)}\")
print(f\"Measurement noise diag R: {np.diag(tracker.R)}\")
"""
        ),
        md("## Spiral attention — helical RoPE on field-line pitch"),
        code(
            """\
from deepiri_fuselk.focal.spiral_attention import apply_spiral_attention, spiral_attention_weights
from deepiri_fuselk.helix.coordinate_mapper import boozer_map, field_line_pitch, q_profile

r = np.linspace(0.1, 1.0, 64)
pitch = field_line_pitch(r, np.zeros_like(r))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(r, q_profile(r), label=\"q(r)\")
axes[0].plot(r, pitch, label=\"field-line pitch\")
axes[0].legend(); axes[0].set_title(\"Safety factor → HQRM rotation angle\")

feat = result.focal_map[:32, :32]
attended = apply_spiral_attention(feat, pitch=0.6)
axes[1].imshow(attended, origin=\"lower\", cmap=\"magma\")
axes[1].set_title(\"Spiral attention output (helical mixing)\")
plt.tight_layout()
plt.show()
"""
        ),
    )


def notebook_03() -> dict:
    return nb(
        md(
            """# Muon Catalysis & Tritium Fuel Cycle

The **muon recycling trifecta** targets $N_{fus,\\mu} > 284$ catalytic fusions per muon (breakeven).

## Extended rate network

Five populations: free $\\mu$, $d\\mu$, $t\\mu$, $dt\\mu$ molecule, and stuck $(\\alpha\\mu)^+$.

Stripping rates $R = R_{col} + R_{photon} + R_{proton}$ reduce effective sticking $\\omega_{eff} = \\omega_0(1 - R)$.

Photon stripping (XFEL + vortex OAM) and proton collision stripping are tunable via `RateNetworkParams`.

---

## Coupled rate-network ODEs (VISION §4)

$$\\frac{dN_\\mu}{dt} = S_{source} + R_{strip}\\, N_{\\alpha\\mu} - \\lambda_{form} N_\\mu - \\frac{N_\\mu}{\\tau_\\mu}$$

$$\\frac{dN_{d\\mu}}{dt} = \\lambda_{form} N_\\mu - \\lambda_{transfer} N_{d\\mu}$$

$$\\frac{dN_{t\\mu}}{dt} = \\lambda_{transfer} N_{d\\mu} - \\lambda_{t\\mu} N_{t\\mu}$$

$$\\frac{dN_{dt\\mu}}{dt} = \\lambda_{t\\mu} N_{t\\mu} + \\cdots - \\lambda_{fusion} N_{dt\\mu}$$

$$\\frac{dN_{\\alpha\\mu}}{dt} = \\lambda_{fusion} N_{dt\\mu}\\, \\omega_0 - R_{strip} N_{\\alpha\\mu} - R_{col} N_{\\alpha\\mu}$$

**Composite stripping** (trifecta):
$$R_{strip} = R_{col} + R_{photon} + R_{proton} + R_{cyclotron}$$

**Effective sticking** (reduced by active stripping):
$$\\omega_{eff} = \\omega_0 (1 - R_{strip})$$

### Breakeven proof sketch ($N_{fus,\\mu} > 284$)

Each catalytic fusion releases $E_{fus} \\approx 17.6$ MeV. Muon production costs $E_\\mu \\approx 4.1$ GeV. Energy breakeven requires:
$$N_{fus,\\mu} > \\frac{E_\\mu}{E_{fus}} \\approx \\frac{4.1 \\times 10^9}{17.6 \\times 10^6} \\approx 233$$

Including sticking losses ($\\omega_0 \\approx 0.008$) and transport inefficiency pushes the **operational threshold to ~284** — implemented as `BREAKEVEN_FUSIONS` in code.

| Stripping configuration (VISION) | Projected $N_{fus,\\mu}$ |
|----------------------------------|--------------------------|
| Collision-only baseline | 112.6 |
| + X-ray photoelectric | 156.5 |
| + Vortex OAM unscrewing | 220.0 |
| + Proton collision | 290.0 |
| + Cyclotron resonance | 350+ |
"""
        ),
        code(SETUP),
        code(
            """\
from deepiri_fuselk.muon import BREAKEVEN_FUSIONS, RateNetworkParams, run_rate_network

baseline = run_rate_network(params=RateNetworkParams(R_photon=0, R_proton=0))
boosted = run_rate_network(params=RateNetworkParams(R_photon=0.5, R_proton=0.3))

print(f\"Breakeven threshold: {BREAKEVEN_FUSIONS:.0f} fusions/μ\")
print(f\"Baseline:  {baseline.fusions_per_muon:.1f} fusions/μ  breakeven={baseline.breakeven}\")
print(f\"Boosted:   {boosted.fusions_per_muon:.1f} fusions/μ  breakeven={boosted.breakeven}\")
print(f\"Effective sticking: {boosted.effective_sticking:.4f}, strip rate: {boosted.strip_rate:.2f}\")
"""
        ),
        code(
            """\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for label, res in [(\"baseline\", baseline), (\"boosted\", boosted)]:
    t_us = res.t * 1e6
    axes[0].plot(t_us, res.populations[3], label=f\"{label} dtμ\")
axes[0].set_xlabel(\"time [μs]\"); axes[0].set_ylabel(\"dtμ population\")
axes[0].legend(); axes[0].set_title(\"Fusion molecule buildup\")

for label, res in [(\"baseline\", baseline), (\"boosted\", boosted)]:
    for i, name in enumerate(res.population_names):
        axes[1].plot(res.t * 1e6, res.populations[i], alpha=0.7, label=f\"{label} {name}\")
axes[1].set_xlabel(\"time [μs]\"); axes[1].set_yscale(\"log\")
axes[1].set_title(\"Full rate network\"); axes[1].legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.show()
"""
        ),
        md("## Experiment: photon strip rate sweep"),
        code(
            """\
photon_rates = np.linspace(0, 0.8, 17)
fpm = []

for r in photon_rates:
    res = run_rate_network(params=RateNetworkParams(R_photon=r, R_proton=0.2))
    fpm.append(res.fusions_per_muon)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(photon_rates, fpm, \"o-\", color=\"C5\")
ax.axhline(BREAKEVEN_FUSIONS, color=\"red\", ls=\"--\", label=\"breakeven = 284\")
ax.set_xlabel(r\"$R_{photon}$\"); ax.set_ylabel(\"fusions per muon\")
ax.set_title(\"XFEL photon stripping sensitivity\")
ax.legend()
plt.tight_layout()
plt.show()
"""
        ),
        md("## Coupling to breeding blanket (oil–water PDE)"),
        code(
            """\
from deepiri_fuselk.barrier.breeding_blanket import evaluate_breeding_blanket
from deepiri_fuselk.physics import PDEParameters, solve_oil_water_steady

pde = solve_oil_water_steady(n_grid=64)
blanket = evaluate_breeding_blanket(pde.state, PDEParameters())

muon_ok = boosted.breakeven
fuel_ok = blanket.tritium_breeding_ratio >= 1.0

print(f\"TBR = {blanket.tritium_breeding_ratio:.3f}\")
print(f\"Muon breakeven: {muon_ok}, Tritium self-sufficient: {fuel_ok}\")
print(f\"Combined fuel-cycle readiness: {muon_ok and fuel_ok}\")
"""
        ),
        md("## Stripping trifecta orchestrator"),
        code(
            """\
from deepiri_fuselk.muon.cyclotron_resonance import CyclotronConfig, cyclotron_frequency, resonance_match
from deepiri_fuselk.muon.photon_stripper import PhotonStripperConfig, stripping_rate as photon_rate
from deepiri_fuselk.muon.proton_stripper import ProtonStripperConfig, stripping_rate as proton_rate
from deepiri_fuselk.muon.stripping_orchestrator import StrippingTrifectaConfig, run_stripping_trifecta

cyc = CyclotronConfig()
print(f\"Cyclotron f_c = {cyclotron_frequency(cyc):.3e} Hz, locked = {resonance_match(cyc)}\")

trifecta = run_stripping_trifecta()
print(f\"R_photon={trifecta.R_photon:.3f}, R_proton={trifecta.R_proton:.3f}, R_cyclotron={trifecta.R_cyclotron:.3f}\")
print(f\"Projected fusions/μ = {trifecta.projected_fpm:.1f}, breakeven={trifecta.breakeven}, margin={trifecta.margin_to_breakeven:.1f}\")
"""
        ),
    )


def notebook_04() -> dict:
    return nb(
        md(
            """# Venturi Control & Disruption Experiments

The **Venturi controller** is a hierarchical hybrid stack:

- **Slow loop (~100 ms):** Bayesian rotational prior sets sweep/RMP bounds
- **Fast loop (~1 ms):** traffic-aware divertor circularization
- **Watchdog:** PID safety override on engineering limit breach

`DisruptionDetector` fuses HELIX SNR, trainable ELM predictor, and MHD stability margins into actuation recommendations.

---

## Hierarchical control as nested optimization (VISION §5)

**Slow loop** (Bayesian prior, ~100 ms): maps global equilibrium features $\\mathbf{s} = (\\bar{n}_e, \\beta_N, f_{rot}, q_{95})$ to policy bounds $(v_{sweep}^{max}, \\phi_{RMP}^{max})$.

**Fast loop** (PPO / traffic router, ~1 ms): maps divertor heat flux map $Q(\\theta, \\phi)$ to actuation $\\mathbf{a} = (v_{sweep}, \\phi_{RMP}, \\dot{n}_{gas})$.

**Watchdog** (hardware): if $Q_{peak} > 0.8\\, Q_{eng}$, override $\\mathbf{a} \\leftarrow \\mathbf{a}_{safe}$.

### Circularization reward

$$\\mathcal{R} = -\\mathrm{Var}_t(Q_{divertor}) - 0.1\\, Q_{peak} + \\mathbb{1}_{\\{\\mathrm{Var} < 0.5\\}} \\cdot 5 - \\mathbb{1}_{\\{watchdog\\}} \\cdot 10$$

### Plasma traffic router (novel abstraction)

Congestion ratio $\\chi = Q_{peak} / Q_{eng}$. The router treats SOL exhaust like **network load balancing** — when $\\chi \\to 1$, increase sweep and RMP phase to redistribute flux toroidally (analogy: TCP packets across servers).

### Disruption fusion functional

$$P_{dis} = \\mathrm{clip}\\big(0.45\\, P_{ELM} + 0.35\\, P_{MHD} + 0.2\\, P_{HELIX},\\, 0,\\, 1\\big)$$

where $P_{MHD}$ derives from $(q_{min}, \\beta_N)$ stability margins — **cross-domain fusion of diagnostics, ML, and MHD** unique to fuselk.
"""
        ),
        code(SETUP),
        code(
            """\
from deepiri_fuselk.control.venturi_controller import VenturiController
from deepiri_fuselk.helix import HelixEngine
from deepiri_fuselk.models.disruption_detector import DisruptionDetector
from deepiri_fuselk.models.elm_predictor import ELMPredictor
from deepiri_fuselk.sim import generate_ece_shot, generate_corpus

venturi = VenturiController(engineering_limit=10.0)
prior = venturi.slow_loop(ne_pedestal=0.9, beta_n=2.8, rotation_khz=5.5, q95=3.2)

# 2D divertor heat-flux map (hotspot)
x = np.linspace(-1, 1, 32)
X, Y = np.meshgrid(x, x)
heat_flux = 8.0 * np.exp(-(X**2 + Y**2) / 0.3)
action = venturi.fast_loop(heat_flux, prior, elm_probability=0.75)
state = venturi.step(heat_flux, elm_probability=0.75)

print(f\"Prior max sweep: {prior.max_sweep_velocity:.2f}, max RMP: {prior.max_rmp_phase:.2f}\")
print(f\"Action: sweep={action.sweep_velocity:.2f}, RMP={action.rmp_phase:.2f}, gas={action.gas_puff:.1f}\")
print(f\"Pellet ready: {action.pellet_ready}, watchdog override: {action.overridden}\")
print(f\"Venturi reward: {state.reward:.3f}\")
"""
        ),
        code(
            """\
# Train ELM predictor on synthetic corpus
corpus = generate_corpus(n_shots=200, grid_size=24, seed=11)
elm = ELMPredictor()
acc = elm.train_from_corpus(corpus)
detector = DisruptionDetector(elm)

shot = generate_ece_shot(32, seed=99, island_amplitude=0.85)
helix = HelixEngine().process(shot.heat_field, shot.raw_signal, shot.angles)
assessment = detector.assess(helix, q_min=1.8, beta_n=3.2)

print(f\"ELM train accuracy: {acc:.1%}\")
print(f\"Disruption probability: {assessment.probability:.2f}\")
print(f\"Recommended action: {assessment.recommended_action}\")
print(f\"Time to disruption: {assessment.time_to_disruption_ms:.1f} ms\")
"""
        ),
        md("## Experiment: heat-flux profile sweep"),
        code(
            """\
profiles = []
rewards = []
x = np.linspace(-1, 1, 32)
X, Y = np.meshgrid(x, x)
for peak in np.linspace(2, 12, 11):
    hf = peak * np.exp(-(X**2 + Y**2) / 0.4)
    st = venturi.step(hf, elm_probability=0.3)
    profiles.append(peak)
    rewards.append(st.reward)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(profiles, rewards, \"o-\", color=\"C6\")
ax.axvline(10.0, color=\"red\", ls=\"--\", alpha=0.6, label=\"engineering limit\")
ax.set_xlabel(\"peak heat flux [MW/m² proxy]\"); ax.set_ylabel(\"Venturi reward\")
ax.set_title(\"Divertor circularization vs. peak load\")
ax.legend()
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """## Optional: PPO vent training (slow — ~30 s)

Uncomment to train a strike-point sweep policy with stable-baselines3.
"""
        ),
        code(
            """\
# from deepiri_fuselk.control.rl_agent import train_vent_policy
# trained = train_vent_policy(timesteps=3000)
# print(f\"Mean reward: {trained.mean_reward:.3f}, policy: {trained.policy_path}\")
print(\"RL training cell skipped (uncomment to run)\")
"""
        ),
    )


def notebook_05() -> dict:
    return nb(
        md(
            """# ReactorCell Digital Twin

The **ReactorCell** closes the full fuselk loop:

```
ECE shot → HELIX → ELM + disruption detect → Venturi/RL actuate → KPI update
```

**FusionKPIs** composite score weights TBR, ELM-free fraction, divertor uniformity, disruption risk, muon gain, and HELIX SNR.

---

## Composite fusion functional (cross-domain score)

$$\\mathcal{F} = 0.25\\, f_{TBR} + 0.20\\, f_{ELM} + 0.15\\, f_{div} + 0.20\\, (1 - P_{dis}) + 0.10\\, f_\\mu + 0.10\\, f_{SNR}$$

where each term is normalized to $[0,1]$. This is the **abstract scalar** that couples plasma control, fuel cycle, and muon catalysis into one benchmark — no other open toolchain defines such a functional.

`FusionCell` extends `ReactorCell` with full trifecta stripping + brine coating hypothesis.
"""
        ),
        code(SETUP),
        code(
            """\
from deepiri_fuselk.sim import ReactorCell
from deepiri_fuselk.sim.fusion_kpis import FusionKPIs

cell = ReactorCell(grid_size=24, train_elm=True)
run = cell.run(n_steps=40, seed=42)
report = run.to_report()

print(\"Reactor run report:\")
for k, v in report.items():
    print(f\"  {k}: {v}\")
"""
        ),
        code(
            """\
steps = [s.step for s in run.steps]
scores = [s.kpis.score() for s in run.steps]
tbr = [s.kpis.tritium_breeding_ratio for s in run.steps]
elm_free = [s.kpis.elm_free_fraction for s in run.steps]
dis_risk = [s.kpis.disruption_risk for s in run.steps]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].plot(steps, scores, \"k-\", lw=2)
axes[0, 0].set_ylabel(\"Fusion score\"); axes[0, 0].set_title(\"Composite KPI trajectory\")

axes[0, 1].plot(steps, tbr, label=\"TBR\")
axes[0, 1].axhline(1.05, color=\"gray\", ls=\"--\")
axes[0, 1].set_ylabel(\"TBR\"); axes[0, 1].legend()

axes[1, 0].plot(steps, elm_free, color=\"C2\", label=\"ELM-free fraction\")
axes[1, 0].plot(steps, [1 - d for d in dis_risk], color=\"C3\", label=\"1 - disruption risk\")
axes[1, 0].set_xlabel(\"step\"); axes[1, 0].legend()

last = run.steps[-1]
hf = last.heat_flux
if hf.ndim == 1:
    side = int(np.sqrt(hf.size))
    hf = hf[: side * side].reshape(side, side)
axes[1, 1].imshow(hf, cmap=\"hot\", origin=\"lower\")
axes[1, 1].set_title(f\"Final heat flux (action: {last.action_taken})\")

plt.tight_layout()
plt.show()
"""
        ),
        md("## DigitalTwin comparison"),
        code(
            """\
from deepiri_fuselk.sim import DigitalTwin

twin = DigitalTwin(grid_size=16)
twin.reset()
for _ in range(30):
    twin.step()
summary = twin.summary()

print(\"DigitalTwin summary:\")
for k, v in summary.items():
    print(f\"  {k}: {v}\")
"""
        ),
        md("## FusionKPI score decomposition (last step)"),
        code(
            """\
k = run.steps[-1].kpis
components = {
    \"TBR (25%)\": min(1.0, k.tritium_breeding_ratio / 1.05),
    \"ELM-free (20%)\": k.elm_free_fraction,
    \"Divertor (15%)\": k.divertor_uniformity,
    \"Disruption (20%)\": 1.0 - k.disruption_risk,
    \"Muon (10%)\": min(1.0, k.muon_gain / 284.0),
    \"HELIX SNR (10%)\": min(1.0, k.helix_snr_mean / 5.0),
}

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(list(components.keys()), list(components.values()), color=\"steelblue\")
ax.set_xlim(0, 1); ax.set_xlabel(\"normalized contribution\")
ax.set_title(f\"KPI breakdown — total score = {k.score():.3f}\")
plt.tight_layout()
plt.show()
"""
        ),
        md("## FusionCell — full VISION architecture"),
        code(
            """\
from deepiri_fuselk.sim.fusion_cell import FusionCell

fusion = FusionCell(grid_size=20, train_elm=False)
run, report = fusion.run(n_steps=25, seed=7)

print(f\"Fusion score: {report.fusion_score:.3f}\")
print(f\"TBR={report.fuel_cycle.tritium_breeding_ratio:.3f}, Pe={report.fuel_cycle.peclet_number:.2f}, extraction_ok={report.fuel_cycle.extraction_ok}\")
print(f\"Muon fusions/μ={report.muon_cycle.fusions_per_muon:.1f}, breakeven={report.muon_cycle.breakeven}\")
print(f\"Brine viable={report.fuel_cycle.brine_viable}, heat reduction={report.fuel_cycle.brine_heat_reduction:.2f}\")
print(f\"Actions taken: {report.recommended_actions}\")
"""
        ),
    )


def notebook_06() -> dict:
    return nb(
        md(
            """# Unified Cross-Domain Mathematics

**The fuselk thesis:** no existing open toolchain (OpenMC, IMAS, BLUEMIRA, BOUT++, SOLPS) combines **live plasma diagnostics**, **MHD precursor geometry**, **hierarchical RL control**, **phase-separated edge PDEs**, and **muon-catalyzed fuel-cycle kinetics** in one mathematical object.

This notebook is the **master theory document** — every governing equation, proof sketch, and coupling term from [VISION.md](../VISION.md), wired to `src/deepiri_fuselk/`.

## Abstract state-space

Define the unified fuselk state on a tokamak pulse:

$$\\mathcal{U}(t) = \\bigl(\\underbrace{\\mathbf{x}_{Kalman}}_{\\text{HELIX}},\\; \\underbrace{\\mathbf{h}_{focal}}_{\\text{HQRM+attention}},\\; \\underbrace{\\mathbf{u}_{PDE}}_{\\text{oil–water}},\\; \\underbrace{\\mathbf{N}_\\mu}_{\\text{muon ODE}},\\; \\underbrace{\\mathbf{a}_{ctrl}}_{\\text{Venturi}}\\bigr)$$

The **closed-loop evolution** is a composition of operators (not a single PDE):

$$\\mathcal{U}_{k+1} = \\Phi_{ctrl}\\!\\circ\\, \\Phi_{pred}\\!\\circ\\, \\Phi_{HELIX}\\!\\circ\\, \\mathcal{D}_k$$

where $\\mathcal{D}_k$ is the diagnostic injection (ECE/SXR) and $\\Phi_{pred}$ fuses ELM + MHD + HELIX into $P_{dis}$.

**Novelty claim (precise):** fuselk is the first open repository where $\\Phi_{ctrl}$ is constrained by both $P_{dis}$ *and* fuel-cycle feasibility $(\\mathrm{TBR}, \\mathrm{Pe}, N_{fus,\\mu})$ — coupling control theory to nuclear fuel engineering.
"""
        ),
        code(SETUP + "\nfrom IPython.display import display\nimport sympy as sp\nsp.init_printing()"),
        md(
            """## I. Oil–water barrier (plasma edge + fuel cycle)

Six coupled fields on $x \\in [0,L]$ — see notebook `01` for numerics.

**Key scalings:**
$$\\delta = \\left(\\frac{D_p D_v}{\\alpha^2 n_0 n_w}\\right)^{1/4}, \\qquad \\mathrm{Pe} = \\frac{v_v \\delta}{D_T}$$

**Theorem (extraction dominance, sketch).** If $\\mathrm{Pe} > 1$ and $S_T(x) \\ge 0$ with wall sink, the Peclet length $\\ell_P = D_T/v_v < \\delta$ implies tritium boundary layer attaches to the wall branch. Wall flux is advection-dominated ⇒ **outward fuel extraction** (not implemented in neutronics-only codes).
"""
        ),
        code(
            """\
from deepiri_fuselk.physics import PDEParameters, interface_thickness, peclet_criterion, solve_oil_water_steady

x, D_T, v_v, delta = sp.symbols(\"x D_T v_v delta\", positive=True)
Pe = v_v * delta / D_T
display(sp.Eq(sp.Symbol(\"Pe\"), Pe))

params = PDEParameters()
print(f\"δ = {interface_thickness(params):.4f}, Pe = {peclet_criterion(params):.3f}\")
steady = solve_oil_water_steady(n_grid=64)
"""
        ),
        md(
            """## II. HELIX + HQRM (diagnostic differential geometry)

**Kalman state** $\\mathbf{x}=[\\theta,\\omega,A,\\phi]^\\top$ with $F(\\Delta t)$ as in notebook `02`.

**Boozer unwrap:** $\\phi = \\theta / q(r)$.

**HQRM lock:** $\\mathrm{Var}(h_i) < 0.07$ over 49 field-aligned squares.

**Lock-in as matched filter:** coherent component at $\\omega$ extracted by
$$\\hat{s}_{coh} = \\frac{1}{N}\\sum_i s_i e^{-i\\theta_i} \\cdot e^{i\\hat{\\theta}}$$
Turbulence is incoherent across rotations ⇒ variance reduction $\\propto 1/\\sqrt{N_{cycles}}$.
"""
        ),
        code(
            """\
from deepiri_fuselk.helix import HelixEngine, boozer_map, field_line_pitch, q_profile
from deepiri_fuselk.sim import generate_ece_shot

shot = generate_ece_shot(40, seed=3)
helix = HelixEngine().process(shot.heat_field, shot.raw_signal, shot.angles)

r = np.linspace(0.2, 1.0, 50)
theta_b, phi_b = boozer_map(r, np.zeros_like(r))
print(f\"HELIX SNR={helix.phase_locked_snr:.2f}, HQRM var={helix.hqrm.heat_variance:.4f}, converged={helix.hqrm.converged}\")
"""
        ),
        md(
            """## III. Muon master equation + stripping trifecta

**Population ODEs** (extended master equation with external field coupling):

$$\\dot{\\mathbf{N}} = A(R_{strip})\\, \\mathbf{N} + \\mathbf{b}$$

where $R_{strip} = R_{col} + R_{photon} + R_{proton} + R_{cyclotron}$ and $\\omega_{eff} = \\omega_0(1-R_{strip})$.

**Breakeven:** $N_{fus,\\mu} \\ge E_\\mu / E_{fus} \\times (1/\\omega_{eff}) \\approx 284$.

**Cross-domain coupling:** muon source $S_\\mu(x)$ enters tritium PDE; tritium breeding feeds $\\mathrm{TBR}$ in $\\mathcal{F}$.
"""
        ),
        code(
            """\
from deepiri_fuselk.muon import BREAKEVEN_FUSIONS, RateNetworkParams, run_rate_network
from deepiri_fuselk.muon.stripping_orchestrator import run_stripping_trifecta

E_mu, E_fus = 4.1e9, 17.6e6  # eV
naive_be = E_mu / E_fus
print(f\"Energy-ratio breakeven (naive): {naive_be:.0f}\")
print(f\"Operational threshold in code: {BREAKEVEN_FUSIONS:.0f}\")

configs = [
    (\"baseline\", RateNetworkParams(R_photon=0, R_proton=0)),
    (\"photon+proton\", RateNetworkParams(R_photon=0.5, R_proton=0.3)),
]
for name, p in configs:
    r = run_rate_network(params=p)
    print(f\"{name:14s} N_fus,μ={r.fusions_per_muon:6.1f}  breakeven={r.breakeven}\")

tri = run_stripping_trifecta()
print(f\"Trifecta projected: {tri.projected_fpm:.1f} (margin {tri.margin_to_breakeven:+.1f})\")
"""
        ),
        md(
            """## IV. Venturi hierarchical control + disruption fusion

**Nested policy:**
$$\\mathbf{a}^* = \\arg\\max_{\\mathbf{a} \\in \\mathcal{A}(prior)} \\; \\mathcal{R}(Q, \\mathbf{a}) \\quad \\text{s.t. watchdog}$$

**Disruption functional** (implemented in `disruption_detector.py`):
$$P_{dis} = \\mathrm{clip}(0.45 P_{ELM} + 0.35 P_{MHD} + 0.2 P_{HELIX}, 0, 1)$$

**Traffic congestion:** $\\chi = Q_{peak}/Q_{eng}$ — SOL modeled as a **router**, not a passive boundary.

This couples **stochastic control** (RL), **Bayesian inference** (slow prior), and **MHD stability theory** ($q_{min}$, $\\beta_N$) — three domains no competitor unifies.
"""
        ),
        code(
            """\
from deepiri_fuselk.control.venturi_controller import VenturiController
from deepiri_fuselk.models.disruption_detector import DisruptionDetector
from deepiri_fuselk.models.elm_predictor import ELMPredictor
from deepiri_fuselk.sim.fusion_kpis import mhd_stability_margin

venturi = VenturiController()
detector = DisruptionDetector(ELMPredictor())

# MHD margin alone
for q_min, beta in [(2.8, 2.0), (1.5, 3.5)]:
    risk = mhd_stability_margin(q_min, beta)
    print(f\"q_min={q_min}, βN={beta} → MHD risk={risk:.2f}\")

assess = detector.assess(helix, q_min=1.9, beta_n=3.1)
print(f\"Fused P_dis={assess.probability:.2f} → {assess.recommended_action}\")
"""
        ),
        md(
            """## V. Master coupling — composite fusion functional

$$\\boxed{\\mathcal{F} = \\sum_i w_i \\, \\phi_i(\\mathcal{U})}$$

| Term $\\phi_i$ | Weight | Domain |
|---------------|--------|--------|
| $f_{TBR}$ | 0.25 | Nuclear engineering |
| $f_{ELM}$ | 0.20 | Plasma MHD + ML |
| $f_{div}$ | 0.15 | Heat exhaust geometry |
| $1 - P_{dis}$ | 0.20 | Control + stability |
| $f_\\mu$ | 0.10 | Particle physics |
| $f_{SNR}$ | 0.10 | Signal processing |

**Well-posedness sketch:** Each $\\Phi$ is Lipschitz on bounded diagnostic states; watchdog enforces compact action set; hence $\\mathcal{U}_k$ remains bounded for finite horizon — sufficient for digital-twin stepping (not a Cauchy blow-up PDE, but a **hybrid dynamical system**).

### Comparison to status quo

| Framework | Diagnostics | Control | Fuel cycle | Muon catalysis |
|-----------|-------------|---------|------------|----------------|
| OpenMC | — | — | neutronics only | — |
| IMAS | schema | — | — | — |
| BLUEMIRA | — | — | steady design | — |
| **fuselk** | HELIX+HQRM | Venturi RL | oil–water PDE | rate network + trifecta |
"""
        ),
        code(
            """\
from deepiri_fuselk.physics import PDEParameters, interface_thickness, peclet_criterion, solve_oil_water_steady
from deepiri_fuselk.muon import RateNetworkParams, run_rate_network
from deepiri_fuselk.sim.fusion_cell import FusionCell

cell = FusionCell(grid_size=20, train_elm=False)
_, report = cell.run(n_steps=30, seed=99)

print(\"=== Unified fuselk report ===\")
print(f\"F = {report.fusion_score:.3f}\")
print(f\"  TBR={report.fuel_cycle.tritium_breeding_ratio:.3f}, Pe={report.fuel_cycle.peclet_number:.2f}\")
print(f\"  μ: {report.muon_cycle.fusions_per_muon:.1f} fusions/μ (breakeven={report.muon_cycle.breakeven})\")
print(f\"  ELM-free={report.elm_free_fraction:.2f}, P_dis={report.disruption_risk:.2f}\")

# Sensitivity: Pe vs muon strip rate (abstract cross-domain plane)
pe_grid = np.linspace(0.5, 2.0, 8)
strip_grid = np.linspace(0, 0.6, 8)
Z = np.zeros((len(strip_grid), len(pe_grid)))
for i, rs in enumerate(strip_grid):
    for j, pe_target in enumerate(pe_grid):
        v_v = pe_target * 0.02 / 0.05  # rough scale to hit Pe
        p = PDEParameters(v_v=v_v)
        tbr = 1.0  # placeholder scale
        mu = run_rate_network(params=RateNetworkParams(R_photon=rs, R_proton=0.2)).fusions_per_muon / 284
        Z[i, j] = 0.5 * min(1, tbr) + 0.5 * min(1, mu)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(Z, origin=\"lower\", aspect=\"auto\", cmap=\"viridis\",
               extent=[pe_grid[0], pe_grid[-1], strip_grid[0], strip_grid[-1]])
ax.set_xlabel(\"Peclet number (fuel extraction)\"); ax.set_ylabel(\"Photon strip rate\")
ax.set_title(\"Cross-domain feasibility plane (schematic)\")
plt.colorbar(im, label=\"combined fuel-cycle score\")
plt.tight_layout()
plt.show()
"""
        ),
        md(
            """## VI. Novel contributions checklist (VISION § Novel Mathematical Contributions)

| # | Contribution | Equation object | In repo |
|---|-------------|-----------------|---------|
| 1 | Coupled oil–water PDE | 6-field system + $\\delta$, Pe | `physics/`, `barrier/` |
| 2 | Peclet extraction | Pe $> 1$ criterion | `peclet_criterion()` |
| 3 | HQRM geometry | shear-split + pitch rotation | `helix/helical_quadtree.py` |
| 4 | 7×7 variance lock | Var $< 0.07$ | `run_hqrm()` |
| 5 | Spiral attention | helical RoPE bias | `focal/spiral_attention.py` |
| 6 | Kalman phase lock | EKF on $[\\theta,\\omega,A,\\phi]$ | `helix/kalman_tracker.py` |
| 7 | Muon rate network | 5-pop ODE + $R_{strip}$ | `muon/rate_network.py` |
| 8 | Hierarchical RL + traffic | $\\mathcal{R}$, $\\chi$ | `control/venturi_controller.py` |
| 9 | Disruption fusion | $P_{dis}$ functional | `models/disruption_detector.py` |
| 10 | Composite $\\mathcal{F}$ | weighted KPI score | `sim/fusion_kpis.py` |

**What remains research-grade (not proven here):** formal existence/uniqueness of the full 6-field PDE, JAX GPU HQRM $<1$ms claim, experimental validation of muon trifecta rates. The notebooks implement **simulation-grade** mathematics with explicit proof sketches where rigorous closure is claimed.
"""
        ),
    )


NOTEBOOK_SPECS = [
    ("00_fuselk_vision_overview.ipynb", notebook_00),
    ("01_oil_water_barrier_math.ipynb", notebook_01),
    ("02_helix_hqrm_diagnostics.ipynb", notebook_02),
    ("03_muon_tritium_fuel_cycle.ipynb", notebook_03),
    ("04_venturi_control_experiments.ipynb", notebook_04),
    ("05_reactor_cell_digital_twin.ipynb", notebook_05),
    ("06_unified_cross_domain_mathematics.ipynb", notebook_06),
]


def main() -> None:
    import uuid

    import nbformat

    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for filename, builder in NOTEBOOK_SPECS:
        path = NOTEBOOKS / filename
        notebook = nbformat.from_dict(builder())
        for cell in notebook.cells:
            if "id" not in cell:
                cell["id"] = uuid.uuid4().hex[:8]
        nbformat.validate(notebook)
        path.write_text(nbformat.writes(notebook, version=4), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
