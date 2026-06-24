"""Digital lock-in amplifier for coherent island extraction."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


def lockin_demodulate(
    signal: np.ndarray,
    reference_phase: float,
    angles: np.ndarray,
    harmonics: int = 1,
) -> tuple[float, float, float]:
    """
    Digital lock-in at reference phase.

    Returns (amplitude, in_phase, quadrature) of coherent component.
    """
    ref = np.cos(harmonics * (angles - reference_phase))
    ref_q = np.sin(harmonics * (angles - reference_phase))
    i_component = float(np.dot(signal, ref) / max(len(signal), 1))
    q_component = float(np.dot(signal, ref_q) / max(len(signal), 1))
    amplitude = float(np.sqrt(i_component**2 + q_component**2))
    return amplitude, i_component, q_component


def hilbert_envelope(signal: np.ndarray) -> np.ndarray:
    """Analytic signal envelope via Hilbert transform."""
    analytic = hilbert(signal)
    return np.abs(analytic)


def subtract_incoherent_noise(
    signal: np.ndarray,
    angles: np.ndarray,
    rotation_hz: float,
    dt: float = 1e-4,
) -> np.ndarray:
    """Remove broadband noise via phase-synchronous averaging over N cycles."""
    omega = 2 * np.pi * rotation_hz
    n_bins = len(signal)
    accumulated = np.zeros(n_bins)
    n_cycles = 8
    for k in range(n_cycles):
        phase_shift = omega * dt * k * n_bins
        shifted_angles = (angles + phase_shift) % (2 * np.pi)
        for i, a in enumerate(shifted_angles):
            idx = int(np.argmin(np.abs(angles - a)))
            accumulated[i] += signal[idx]
    return accumulated / n_cycles
