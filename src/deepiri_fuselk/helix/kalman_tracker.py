"""Full 4-state Kalman filter for rotating magnetic island O-point."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackerState:
    theta: float
    phi: float
    omega: float
    amplitude: float
    covariance: np.ndarray | None = None


class PhaseLockedTracker:
    """
    Extended Kalman filter tracking [theta, omega, amplitude, phi].

    State transition models toroidal rotation; measurements from ECE/SXR arrays.
    """

    def __init__(self, omega_init: float = 5e3) -> None:
        self.dim = 4
        self.x = np.array([0.0, omega_init, 0.0, 0.0])  # theta, omega, amp, phi
        self.P = np.diag([0.1, 1e6, 0.5, 0.1])
        self.Q = np.diag([1e-6, 1e4, 1e-4, 1e-6])
        self.R = np.diag([0.01, 0.01, 0.05])

    @property
    def state(self) -> TrackerState:
        return TrackerState(
            theta=float(self.x[0]),
            phi=float(self.x[3]),
            omega=float(self.x[1]),
            amplitude=float(self.x[2]),
            covariance=self.P.copy(),
        )

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(4)
        F[0, 1] = dt
        F[3, 1] = dt * 0.5
        return F

    def predict(self, dt: float) -> TrackerState:
        F = self._F(dt)
        self.x = F @ self.x
        self.x[0] = self.x[0] % (2 * np.pi)
        self.x[3] = self.x[3] % (2 * np.pi)
        self.P = F @ self.P @ F.T + self.Q
        return self.state

    def update(
        self,
        measurement_theta: float,
        measurement_phi: float,
        amplitude: float,
    ) -> TrackerState:
        z = np.array([measurement_theta, measurement_phi, amplitude])
        H = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ])
        y = z - H @ self.x
        y[0] = (y[0] + np.pi) % (2 * np.pi) - np.pi
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return self.state

    def sample_at_phase(self, signal: np.ndarray, angles: np.ndarray) -> float:
        """Phase-synchronous lock-in sample at predicted O-point."""
        theta = self.x[0]
        diff = np.abs((angles - theta + np.pi) % (2 * np.pi) - np.pi)
        idx = int(np.argmin(diff))
        window = max(1, len(signal) // 16)
        lo = max(0, idx - window)
        hi = min(len(signal), idx + window + 1)
        return float(np.mean(signal[lo:hi]))

    def estimate_rotation_from_signal(self, signal: np.ndarray, angles: np.ndarray) -> float:
        """Estimate dominant rotation frequency via FFT peak."""
        if len(signal) < 8:
            return self.x[1]
        fft = np.abs(np.fft.rfft(signal - np.mean(signal)))
        freqs = np.fft.rfftfreq(len(signal), d=float(angles[1] - angles[0]) / (2 * np.pi))
        if len(fft) < 2:
            return self.x[1]
        peak_idx = int(np.argmax(fft[1:]) + 1)
        return float(2 * np.pi * freqs[peak_idx])
