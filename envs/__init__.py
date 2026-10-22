"""Gymnasium environments for rocket landing."""

from gymnasium.envs.registration import register
from envs.landing_env import (
    DEFAULT_REWARD_WEIGHTS,
    DEFAULT_SENSOR_SIGMAS,
    DEFAULT_START_RANGES,
    EASY_START_RANGES,
    MAX_WIND_FORCE,
    WIDE_START_RANGES,
    HoverSlamEnv,
)

register(
    id="HoverSlam-v0",
    entry_point="envs.landing_env:HoverSlamEnv",
    max_episode_steps=1500,
)

__all__ = [
    "HoverSlamEnv",
    "DEFAULT_REWARD_WEIGHTS",
    "MAX_WIND_FORCE",
    "DEFAULT_SENSOR_SIGMAS",
    "DEFAULT_START_RANGES",
    "WIDE_START_RANGES",
    "EASY_START_RANGES",
]
