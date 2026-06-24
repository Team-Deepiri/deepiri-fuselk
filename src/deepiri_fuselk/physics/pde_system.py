"""Coupled oil-water plasma / vapor PDE system."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PDEParameters:
    """Physical parameters for the 1D oil-water barrier model."""

    D_p: float = 0.05
    D_v: float = 0.10
    D_T: float = 0.02
    alpha: float = 1.0
    v_v: float = 0.5
    n0: float = 1.0
    n_wall: float = 1.0
    sigma_li: float = 0.3
    phi_n: float = 1.0
    kappa_p: float = 0.01
    kappa_v: float = 0.5


@dataclass
class PDEState:
    """Field variables on a 1D spatial grid."""

    x: np.ndarray
    n_p: np.ndarray
    n_v: np.ndarray
    n_T: np.ndarray
    T_p: np.ndarray
    T_v: np.ndarray


def interface_thickness(params: PDEParameters) -> float:
    """Characteristic interfacial thickness delta."""
    return (params.D_p * params.D_v / (params.alpha**2 * params.n0 * params.n_wall)) ** 0.25


def peclet_criterion(params: PDEParameters) -> float:
    """Peclet number for tritium extraction; > 1 means outward sweep dominates."""
    delta = interface_thickness(params)
    return params.v_v * delta / params.D_T


def mass_flux(params: PDEParameters, state: PDEState) -> np.ndarray:
    """Total particle flux — should be constant in steady state."""
    dn_p = np.gradient(state.n_p, state.x)
    dn_v = np.gradient(state.n_v, state.x)
    return params.D_p * dn_p + params.D_v * dn_v - params.v_v * state.n_v


def tritium_source(params: PDEParameters, state: PDEState) -> np.ndarray:
    """Neutron breeding source term sigma_Li * Phi_n * n_v."""
    return params.sigma_li * params.phi_n * state.n_v


def bremsstrahlung_power(n_p: np.ndarray, T_p: np.ndarray) -> np.ndarray:
    """Simplified bremsstrahlung cooling ~ n_p^2 sqrt(T_p)."""
    return n_p**2 * np.sqrt(np.maximum(T_p, 1e-6))


def radiative_cooling(n_v: np.ndarray, T_v: np.ndarray) -> np.ndarray:
    """Line radiation from vapor barrier."""
    return 0.1 * n_v * T_v**1.5
