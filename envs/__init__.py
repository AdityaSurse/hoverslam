"""Gymnasium environments for rocket landing."""

from gymnasium.envs.registration import register
from envs.landing_env import DEFAULT_REWARD_WEIGHTS, HoverSlamEnv

register(
    id="HoverSlam-v0",
    entry_point="envs.landing_env:HoverSlamEnv",
    max_episode_steps=1500,
)

__all__ = ["HoverSlamEnv", "DEFAULT_REWARD_WEIGHTS"]
