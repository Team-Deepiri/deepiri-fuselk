"""Phase-locked Kalman tracker for rotating O-point."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackerState:
    theta: float
    phi: float
    omega: float
    amplitude: float


class PhaseLockedTracker:
    """Track island O-point rotation with a simple Kalman filter."""

    def __init__(self, omega_init: float = 5e3) -> None:
        self.state = TrackerState(theta=0.0, phi=0.0, omega=omega_init, amplitude=0.0)
        self._process_noise = 1e-4
        self._measure_noise = 1e-2

    def predict(self, dt: float) -> TrackerState:
        self.state.theta += self.state.omega * dt
        self.state.phi += self.state.omega * dt * 0.5
        return self.state

    def update(
        self, measurement_theta: float, measurement_phi: float, amplitude: float
    ) -> TrackerState:
        gain = self._process_noise / (self._process_noise + self._measure_noise)
        self.state.theta += gain * (measurement_theta - self.state.theta)
        self.state.phi += gain * (measurement_phi - self.state.phi)
        self.state.amplitude = 0.9 * self.state.amplitude + 0.1 * amplitude
        return self.state

    def sample_at_phase(self, signal: np.ndarray, angles: np.ndarray) -> float:
        """Phase-synchronous sample: extract signal at predicted O-point."""
        diff = np.abs(angles - self.state.theta)
        idx = int(np.argmin(diff))
        return float(signal[idx])
