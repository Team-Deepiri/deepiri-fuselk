"""PPO training and deployment for vent circularization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from deepiri_fuselk.control.convergence import ConvergenceReport, verify_rl_convergence
from deepiri_fuselk.control.vent_circularizer_env import VentCircularizerEnv

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    _SB3 = True
except ImportError:
    _SB3 = False
    PPO = None  # type: ignore[misc, assignment]


@dataclass
class TrainResult:
    mean_reward: float
    policy_path: Path | None
    timesteps: int
    convergence: ConvergenceReport | None = None


def train_vent_policy(
    timesteps: int = 20_000,
    save_path: str | Path = ".fuselk-data/policies/vent_ppo",
    grid_size: int = 16,
    seed: int = 42,
    verify_convergence: bool = False,
) -> TrainResult:
    """Train PPO policy for divertor vent circularization."""
    convergence = verify_rl_convergence(grid_size=grid_size) if verify_convergence else None
    if not _SB3:
        return TrainResult(
            mean_reward=train_random_baseline(episodes=10),
            policy_path=None,
            timesteps=0,
            convergence=convergence,
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(lambda: VentCircularizerEnv(grid_size=grid_size), n_envs=4, seed=seed)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        seed=seed,
        device="cpu",
    )
    model.learn(total_timesteps=timesteps)

    eval_env = VentCircularizerEnv(grid_size=grid_size)
    rewards = []
    for ep in range(10):
        obs, _ = eval_env.reset(seed=seed + ep)
        ep_r = 0.0
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, _ = eval_env.step(action)
            ep_r += r
            if done:
                break
        rewards.append(ep_r)

    model.save(str(save_path))
    return TrainResult(
        mean_reward=float(np.mean(rewards)),
        policy_path=save_path,
        timesteps=timesteps,
        convergence=convergence,
    )


def load_policy(path: str | Path):
    if not _SB3:
        raise RuntimeError("stable-baselines3 required; pip install stable-baselines3")
    return PPO.load(str(path))


def run_policy(
    policy_path: str | Path,
    steps: int = 200,
    grid_size: int = 16,
) -> tuple[float, np.ndarray]:
    """Run trained policy; return total reward and final heat map."""
    model = load_policy(policy_path)
    env = VentCircularizerEnv(grid_size=grid_size)
    obs, _ = env.reset()
    total = 0.0
    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, _ = env.step(action)
        total += r
        if done:
            obs, _ = env.reset()
    return total, obs.copy()


def train_random_baseline(episodes: int = 5, steps: int = 50) -> float:
    """Random policy baseline for comparison."""
    env = VentCircularizerEnv()
    rewards = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0
        for _ in range(steps):
            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
    return float(np.mean(rewards))
