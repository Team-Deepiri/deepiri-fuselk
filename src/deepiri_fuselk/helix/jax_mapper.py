"""JAX-accelerated coordinate transforms with NumPy fallback."""

from __future__ import annotations

import numpy as np

_JAX_AVAILABLE = False
try:
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]


def boozer_map_fast(
    R: np.ndarray, Z: np.ndarray, q0: float = 1.0, q95: float = 3.5
) -> tuple[np.ndarray, np.ndarray]:
    """Boozer map — JAX-jitted when available."""
    if _JAX_AVAILABLE and jax is not None and jnp is not None:

        @jax.jit
        def _map(r, z):
            q = q0 + (q95 - q0) * (r / 1.0) ** 2
            theta = jnp.arctan2(z, r)
            phi = theta / jnp.maximum(q, 1e-6)
            return theta, phi

        theta, phi = _map(jnp.asarray(R), jnp.asarray(Z))
        return np.asarray(theta), np.asarray(phi)

    r = np.sqrt(R**2 + Z**2)
    q = q0 + (q95 - q0) * (r / 1.0) ** 2
    theta = np.arctan2(Z, R)
    phi = theta / np.maximum(q, 1e-6)
    return theta, phi


def jax_available() -> bool:
    return _JAX_AVAILABLE
