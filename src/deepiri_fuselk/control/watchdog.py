"""Safety watchdog — PID override when heat flux exceeds limits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WatchdogAction:
    override: bool
    safe_sweep_velocity: float
    safe_rmp_phase: float


class SafetyWatchdog:
    """Override RL actions when engineering limits are breached."""

    def __init__(self, limit_fraction: float = 0.8, engineering_limit: float = 10.0) -> None:
        self.limit_fraction = limit_fraction
        self.engineering_limit = engineering_limit
        self.trigger_count = 0

    def check(self, peak_flux: float, proposed_action: np.ndarray) -> WatchdogAction:
        threshold = self.limit_fraction * self.engineering_limit
        if peak_flux > threshold:
            self.trigger_count += 1
            return WatchdogAction(
                override=True,
                safe_sweep_velocity=0.8,
                safe_rmp_phase=0.5,
            )
        return WatchdogAction(
            override=False,
            safe_sweep_velocity=float(proposed_action[0]),
            safe_rmp_phase=float(proposed_action[1]),
        )
