"""Deploy trained vent policies in control loops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deepiri_fuselk.control.rl_agent import load_policy
from deepiri_fuselk.control.venturi_controller import VenturiController, VenturiState


@dataclass
class HybridControlResult:
    venturi: VenturiState
    rl_reward: float
    final_heat: np.ndarray
    policy_active: bool


class HybridPolicyRunner:
    """Combine Venturi hierarchical control with trained PPO divertor policy."""

    def __init__(self, policy_path: Path | None = None) -> None:
        self.venturi = VenturiController()
        self.policy_path = Path(policy_path) if policy_path else None
        self._model = None
        self._rl_env = None
        self._rl_grid_size: int | None = None
        if self.policy_path and self.policy_path.exists():
            self._model = load_policy(self.policy_path)

    @property
    def has_policy(self) -> bool:
        return self._model is not None

    def _ensure_rl_env(self, grid_size: int):
        from deepiri_fuselk.control.vent_circularizer_env import VentCircularizerEnv

        if self._rl_env is None or self._rl_grid_size != grid_size:
            self._rl_env = VentCircularizerEnv(grid_size=grid_size)
            self._rl_grid_size = grid_size
        self._rl_env.reset()
        return self._rl_env

    def step(
        self,
        heat_flux: np.ndarray,
        elm_probability: float = 0.0,
        rl_steps: int = 1,
    ) -> HybridControlResult:
        vent_state = self.venturi.step(heat_flux, elm_probability=elm_probability)
        rl_reward = 0.0
        final_heat = heat_flux.copy()

        if self._model is not None:
            env = self._ensure_rl_env(heat_flux.shape[0])
            obs = heat_flux.astype(np.float32)
            for _ in range(rl_steps):
                action, _ = self._model.predict(obs, deterministic=True)
                obs, r, _, _, _ = env.step(action)
                rl_reward += r
            final_heat = obs

        return HybridControlResult(
            venturi=vent_state,
            rl_reward=rl_reward,
            final_heat=final_heat,
            policy_active=self.has_policy,
        )

    def train_and_attach(self, timesteps: int = 10_000, save_path: Path | None = None) -> Path:
        from deepiri_fuselk.control.rl_agent import train_vent_policy

        path = save_path or Path(".fuselk-data/policies/vent_ppo")
        result = train_vent_policy(timesteps=timesteps, save_path=path)
        if result.policy_path:
            self.policy_path = result.policy_path
            self._model = load_policy(self.policy_path)
            self._rl_env = None
        return self.policy_path or path
