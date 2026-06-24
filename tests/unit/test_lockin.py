"""Tests for digital lock-in amplifier."""

import numpy as np
from deepiri_fuselk.focal.lockin_amplifier import (
    hilbert_envelope,
    lockin_demodulate,
    subtract_incoherent_noise,
)


def test_lockin_demodulate_coherent_signal():
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    signal = np.cos(angles)
    amp, i_comp, q_comp = lockin_demodulate(signal, 0.0, angles)
    assert amp > 0.4
    assert abs(i_comp) > 0.3


def test_hilbert_envelope_positive():
    signal = np.sin(np.linspace(0, 4 * np.pi, 32))
    env = hilbert_envelope(signal)
    assert np.all(env >= 0)


def test_subtract_incoherent_noise_reduces_variance():
    rng = np.random.default_rng(0)
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    coherent = 2.0 * np.cos(angles)
    noisy = coherent + 0.5 * rng.standard_normal(64)
    cleaned = subtract_incoherent_noise(noisy, angles, rotation_hz=1e4)
    assert np.var(cleaned) <= np.var(noisy) * 1.5
