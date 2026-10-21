"""Unit tests for HoverSlam Gymnasium environment."""

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

import envs
from envs.landing_env import DEFAULT_REWARD_WEIGHTS, HoverSlamEnv
from sim.rocket import (
    DRY_MASS,
    GIMBAL_MAX,
    IDX_M,
    IDX_THETA,
    IDX_VX,
    IDX_VY,
    IDX_X,
    IDX_Y,
    MIN_THROTTLE,
)


def test_gym_check_env():
    """Verify HoverSlamEnv satisfies Gymnasium interface via check_env."""
    env = gym.make("HoverSlam-v0")
    check_env(env.unwrapped)
    env.close()


def test_registered_env_make():
    """Verify HoverSlam-v0 is registered and instantiable via gym.make."""
    env = gym.make("HoverSlam-v0")
    obs, info = env.reset(seed=123)
    assert obs.shape == (10,)
    assert obs.dtype == np.float32
    assert "landed" in info
    assert "distance_from_pad" in info
    env.close()


def test_action_mapping():
    """Verify action mapping logic:
    - a[0] < -0.5 -> throttle = 0.0 (engine off)
    - a[0] >= -0.5 -> mapped to [MIN_THROTTLE, 1.0]
    - a[1] mapped to [-GIMBAL_MAX, GIMBAL_MAX]
    """
    env = HoverSlamEnv()

    # Engine off
    th_off, _ = env._map_action([-0.6, 0.0])
    assert th_off == 0.0

    # Exactly at boundary -0.5 -> MIN_THROTTLE
    th_min, _ = env._map_action([-0.5, 0.0])
    assert th_min == pytest.approx(MIN_THROTTLE, abs=1e-5)

    # Maximum action +1.0 -> 1.0
    th_max, g_max = env._map_action([1.0, 1.0])
    assert th_max == pytest.approx(1.0, abs=1e-5)
    assert g_max == pytest.approx(GIMBAL_MAX, abs=1e-5)

    # Minimum gimbal -1.0 -> -GIMBAL_MAX
    _, g_min = env._map_action([0.0, -1.0])
    assert g_min == pytest.approx(-GIMBAL_MAX, abs=1e-5)

    env.close()


def test_vertical_only_mode():
    """Verify vertical_only locks gimbal, tilt, and lateral position to zero."""
    env = HoverSlamEnv(vertical_only=True)
    obs, info = env.reset(seed=42)

    assert env.state[IDX_X] == 0.0
    assert env.state[IDX_VX] == 0.0
    assert env.state[IDX_THETA] == 0.0

    # Command full gimbal deflection
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step([1.0, 1.0])
        assert env.state[IDX_X] == 0.0
        assert env.state[IDX_VX] == 0.0
        assert env.state[IDX_THETA] == 0.0
        assert env.last_gimbal == 0.0

    env.close()


def test_soft_landing_terminal_reward():
    """Verify that touching down within soft landing criteria awards the +100 bonus."""
    env = HoverSlamEnv()
    env.reset(seed=42)

    # Place rocket just above ground moving very slowly upright
    env.state = np.array([0.0, 0.01, 0.0, -1.0, 0.0, 0.0, DRY_MASS + 50.0], dtype=np.float64)
    env.prev_error = env._compute_error(env.state)

    # Step down to trigger touchdown
    obs, reward, terminated, truncated, info = env.step([-1.0, 0.0])

    assert terminated is True
    assert info["landed"] is True
    # Reward should contain the +100 soft landing bonus
    assert reward > 90.0
    env.close()


def test_crash_terminal_reward():
    """Verify that touching down at excessive vertical speed incurs a crash penalty."""
    env = HoverSlamEnv()
    env.reset(seed=42)

    # Place rocket just above ground dropping at -40 m/s
    env.state = np.array([0.0, 0.01, 0.0, -40.0, 0.0, 0.0, DRY_MASS + 50.0], dtype=np.float64)
    env.prev_error = env._compute_error(env.state)

    obs, reward, terminated, truncated, info = env.step([-1.0, 0.0])

    assert terminated is True
    assert info["landed"] is False
    # Crash penalty is -100 - 1.0 * vy_abs
    assert reward < -100.0
    env.close()


def test_out_of_bounds_terminal():
    """Verify that flying out of bounds terminates with penalty."""
    env = HoverSlamEnv()
    env.reset(seed=42)

    # Place rocket beyond lateral boundary (|x| > 200)
    env.state = np.array([210.0, 100.0, 10.0, 0.0, 0.0, 0.0, DRY_MASS + 50.0], dtype=np.float64)
    obs, reward, terminated, truncated, info = env.step([0.0, 0.0])

    assert terminated is True
    assert info["landed"] is False
    assert reward < -50.0
    env.close()


def test_rgb_array_rendering():
    """Verify render_mode='rgb_array' generates a valid uint8 image."""
    env = HoverSlamEnv(render_mode="rgb_array")
    env.reset(seed=42)

    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8
    env.close()
