"""Tests for HELIX engine and HQRM."""

import numpy as np
from deepiri_fuselk.focal.focal_heatmap import focal_heatmap, singularity_gradient
from deepiri_fuselk.helix.coordinate_mapper import boozer_map, q_profile
from deepiri_fuselk.helix.helical_quadtree import run_hqrm
from deepiri_fuselk.helix.kalman_tracker import PhaseLockedTracker


def test_q_profile_monotonic():
    r = np.linspace(0, 1, 20)
    q = q_profile(r)
    assert q[0] < q[-1]


def test_boozer_map_shapes():
    R = np.linspace(0.5, 1.5, 16)
    Z = np.linspace(-0.5, 0.5, 16)
    theta, phi = boozer_map(R, Z)
    assert theta.shape == R.shape
    assert phi.shape == R.shape


def test_hqrm_returns_result():
    heat = np.random.default_rng(0).random((32, 32))
    result = run_hqrm(heat, shear_threshold=0.01)
    assert result.o_point[0] != 0 or result.o_point[1] != 0
    assert result.heat_variance >= 0


def test_focal_heatmap_shape():
    signal = np.random.default_rng(1).random(32)
    angles = np.linspace(0, 2 * np.pi, 32)
    tracker = PhaseLockedTracker()
    hm = focal_heatmap(signal, angles, tracker, size=32)
    assert hm.shape == (32, 32)


def test_singularity_gradient():
    hm = np.zeros((16, 16))
    hm[8, 8] = 10.0
    gx, gy = singularity_gradient(hm)
    assert isinstance(gx, float)
    assert isinstance(gy, float)
