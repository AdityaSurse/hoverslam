"""Unit tests for the HoverSlam cascaded PID controller."""

import gymnasium as gym
import numpy as np
import pytest

import envs
from controllers.pid import DEFAULT_PID_GAINS, HoverSlamPID
from sim.rocket import DRY_MASS, FUEL_MASS


def test_pid_unnormalize_obs():
    """Verify unnormalization transforms normalized observation back to physical units."""
    controller = HoverSlamPID()

    # Construct mock normalized observation
    # [x/50, y/300, vx/20, vy/80, sin(th), cos(th), omega, fuel_frac, a0, a1]
    mock_obs = np.array([
        0.2,   # x = 10.0 m
        0.5,   # y = 150.0 m
        -0.1,  # vx = -2.0 m/s
        -0.5,  # vy = -40.0 m/s
        0.0,   # sin(0) = 0
        1.0,   # cos(0) = 1
        0.02,  # omega = 0.02 rad/s
        0.8,   # fuel_frac = 0.8
        0.0,
        0.0,
    ], dtype=np.float32)

    physical = controller.unnormalize_obs(mock_obs)
    assert physical["x"] == pytest.approx(10.0, abs=1e-5)
    assert physical["y"] == pytest.approx(150.0, abs=1e-5)
    assert physical["vx"] == pytest.approx(-2.0, abs=1e-5)
    assert physical["vy"] == pytest.approx(-40.0, abs=1e-5)
    assert physical["theta"] == pytest.approx(0.0, abs=1e-5)
    assert physical["omega"] == pytest.approx(0.02, abs=1e-5)
    assert physical["m"] == pytest.approx(DRY_MASS + 0.8 * FUEL_MASS, abs=1e-5)


def test_pid_action_bounds_and_freefall():
    """Verify controller keeps engine off during high free fall and clamps actions."""
    controller = HoverSlamPID()

    # High altitude, slow descent: stopping distance is tiny compared to 300 m
    high_obs = np.array([0.0, 1.0, 0.0, -0.1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    action = controller.compute_action(high_obs)

    assert not controller.ignited
    # action[0] == -1.0 means engine completely off
    assert action[0] == -1.0
    assert -1.0 <= action[1] <= 1.0


def test_pid_nominal_landing():
    """Verify PID controller successfully lands from nominal starting conditions."""
    env = gym.make("HoverSlam-v0")
    controller = HoverSlamPID()

    # Test 3 deterministic seeds with randomize_start=False
    for seed in [1, 42, 99]:
        obs, info = env.reset(seed=seed)
        controller.reset()

        while True:
            action = controller.compute_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        assert info["landed"] is True
        assert abs(info["touchdown_vy"]) < 2.0
        assert info["distance_from_pad"] < 5.0
        assert abs(info["touchdown_tilt"]) < np.radians(5.0)

    env.close()
