"""RL agent training stub."""

from __future__ import annotations

import numpy as np

from deepiri_fuselk.control.vent_circularizer_env import VentCircularizerEnv


def train_random_baseline(episodes: int = 5, steps: int = 50) -> float:
    """Run random-policy baseline and return mean reward."""
    env = VentCircularizerEnv()
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        for _ in range(steps):
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
    return float(np.mean(rewards))
