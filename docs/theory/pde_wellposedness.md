# Six-Field PDE Well-Posedness

## Field vector

$$\mathbf{u} = (n_p,\; n_v,\; n_T,\; n_\mu,\; T_p,\; T_v)$$

| Field | Equation type | Role |
|-------|---------------|------|
| $n_p$ | Parabolic + reaction | Plasma ("oil") density |
| $n_v$ | Parabolic + advection + reaction | Vapor ("water") barrier |
| $n_T$ | Elliptic (steady) / parabolic | Tritium breeding |
| $n_\mu$ | Parabolic + source | Muon catalyst |
| $T_p$ | Parabolic energy | Core temperature |
| $T_v$ | Parabolic energy | Barrier cooling |

## Existence (local)

Sufficient conditions implemented in `physics/pde_wellposedness.py`:

1. **Strictly positive diffusion** $D_p, D_v, D_T, \kappa_p, \kappa_v > 0$
2. **Lipschitz reaction** $\alpha n_p n_v$ on the box $0 \le n_p \le n_0$, $0 \le n_v \le n_{wall}$

$$L_{react} = \alpha \max(n_0, n_{wall})$$

3. **Energy dissipation** $\kappa_p, \kappa_v > 0$ for parabolic temperature equations

By standard semilinear parabolic theory (Ladyzhenskaya–Solonnikov–Uraltseva), local existence in time follows.

## Uniqueness (steady state)

For the steady coupled $(n_p, n_v)$ subsystem, define the fixed-point operator $F$ on interior nodal values. With effective Laplacian spectral gap $\lambda_{min} \sim \pi^2 D_{min} / L^2$:

$$L_{contract} = \frac{L_{react}}{\lambda_{min}}$$

**Unique steady state** when $L_{contract} < 1$ (Banach contraction).

Default `PDEParameters()` satisfies this; run:

```bash
fuselk validate claims --pde
```

## Tritium sub-problem

Given $(n_p, n_v)$, the tritium equation is **linear elliptic** with upwind advection:

$$\nabla \cdot (-D_T \nabla n_T + v_v n_T) = \sigma_{Li} \Phi_n n_v$$

Unique solution exists for $v_v > 0$ with Dirichlet/outflow BCs.

## Code

```python
from deepiri_fuselk.physics.pde_wellposedness import verify_wellposedness
report = verify_wellposedness()
assert report.passed
```
