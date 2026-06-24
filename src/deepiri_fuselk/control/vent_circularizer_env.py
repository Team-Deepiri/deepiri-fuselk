"""Gymnasium environment for vent circularization RL."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class VentCircularizerEnv(gym.Env):
    """RL environment: circularize divertor heat flux via strike-point sweeping."""

    metadata = {"render_modes": []}

    def __init__(self, grid_size: int = 16, max_steps: int = 200) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.observation_space = spaces.Box(0, 10, (grid_size, grid_size), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )
        self._heat = np.zeros((grid_size, grid_size), dtype=np.float32)
        self._step_count = 0
        self._phase = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        cx, cy = self.grid_size // 2, self.grid_size // 2
        self._heat = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self._heat[cx, cy] = 8.0 + rng.random() * 2.0
        self._step_count = 0
        self._phase = 0.0
        return self._heat.copy(), {}

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        sweep_velocity, rmp_phase = float(action[0]), float(action[1])
        self._phase += sweep_velocity * 0.5 + rmp_phase * 0.1

        y, x = np.ogrid[: self.grid_size, : self.grid_size]
        cx = self.grid_size / 2 + 3 * np.cos(self._phase)
        cy = self.grid_size / 2 + 3 * np.sin(self._phase)
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        self._heat = 5.0 * np.exp(-r2 / 8.0).astype(np.float32)

        variance = float(np.var(self._heat))
        peak = float(np.max(self._heat))
        reward = -variance - 0.1 * peak
        if variance < 0.5:
            reward += 10.0

        self._step_count += 1
        terminated = self._step_count >= self.max_steps
        return self._heat.copy(), reward, terminated, False, {"variance": variance}
