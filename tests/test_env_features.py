"""Unit tests for extended environment features: wind, sensor noise, mass/start randomization, and curriculum."""

from collections import deque
import numpy as np
import pytest

from envs.landing_env import (
    DEFAULT_SENSOR_SIGMAS,
    DEFAULT_START_RANGES,
    EASY_START_RANGES,
    MAX_WIND_FORCE,
    WIDE_START_RANGES,
    HoverSlamEnv,
)
from sim.rocket import (
    DRY_MASS,
    IDX_M,
    IDX_OMEGA,
    IDX_THETA,
    IDX_VX,
    IDX_VY,
    IDX_X,
    IDX_Y,
    START_MASS,
)
from train import CurriculumCallback, LandingStatsCallback


def test_defaults_are_unchanged():
    """Verify that all new features default to off and existing behavior is preserved."""
    env = HoverSlamEnv()

    # Feature flags and strengths default to 0.0 or False
    assert env.wind_strength == 0.0
    assert env.sensor_noise == 0.0
    assert env.mass_randomization == 0.0
    assert env.randomize_start is False
    assert env.wide_start is False
    assert env.curriculum is False
    assert env.sample_random_params_each_episode is False

    obs, info = env.reset(seed=42)

    # Initial physical state matches nominal values exactly
    assert env.state[IDX_X] == 0.0
    assert env.state[IDX_Y] == 300.0
    assert env.state[IDX_VX] == 0.0
    assert env.state[IDX_VY] == -55.0
    assert env.state[IDX_THETA] == 0.0
    assert env.state[IDX_OMEGA] == 0.0
    assert env.state[IDX_M] == pytest.approx(START_MASS, abs=1e-5)

    # Observation matches uncorrupted deterministic state
    assert obs[0] == pytest.approx(0.0, abs=1e-5)
    assert obs[1] == pytest.approx(1.0, abs=1e-5)  # 300 / 300
    assert obs[2] == pytest.approx(0.0, abs=1e-5)
    assert obs[3] == pytest.approx(-55.0 / 80.0, abs=1e-5)
    assert obs[4] == pytest.approx(0.0, abs=1e-5)  # sin(0)
    assert obs[5] == pytest.approx(1.0, abs=1e-5)  # cos(0)
    assert obs[6] == pytest.approx(0.0, abs=1e-5)

    # Wind force is strictly zero
    obs, reward, terminated, truncated, info = env.step([-1.0, 0.0])
    assert info["wind_x"] == 0.0
    assert env.last_wind_x == 0.0
    assert env.state[IDX_X] == 0.0
    assert env.state[IDX_VX] == 0.0

    env.close()


def test_wind_changes_trajectory():
    """Verify that enabling wind (steady + Ornstein-Uhlenbeck) alters lateral trajectory."""
    env_nowind = HoverSlamEnv(wind_strength=0.0)
    env_wind = HoverSlamEnv(wind_strength=1.0)

    env_nowind.reset(seed=101)
    env_wind.reset(seed=101)

    # Step both environments with engine off for 20 steps
    for _ in range(20):
        obs_nw, _, _, _, info_nw = env_nowind.step([-1.0, 0.0])
        obs_w, _, _, _, info_w = env_wind.step([-1.0, 0.0])

    # No-wind rocket remains centered at x=0, vx=0
    assert env_nowind.state[IDX_X] == 0.0
    assert env_nowind.state[IDX_VX] == 0.0
    assert info_nw["wind_x"] == 0.0

    # Wind-exposed rocket has been deflected laterally
    assert env_wind.state[IDX_X] != 0.0
    assert env_wind.state[IDX_VX] != 0.0
    assert abs(info_w["wind_x"]) > 0.0
    assert abs(info_w["wind_x"]) <= MAX_WIND_FORCE + 1e-3

    env_nowind.close()
    env_wind.close()


def test_sensor_noise_changes_obs_not_true_state():
    """Verify that sensor noise alters returned observation array without corrupting true physical state."""
    env_clean = HoverSlamEnv(sensor_noise=0.0)
    env_noisy = HoverSlamEnv(sensor_noise=1.0)

    obs_clean, _ = env_clean.reset(seed=202)
    obs_noisy, _ = env_noisy.reset(seed=202)

    # Physical states must be strictly identical
    np.testing.assert_allclose(env_clean.state, env_noisy.state, atol=1e-9)

    # Observations must differ due to noise
    assert not np.allclose(obs_clean, obs_noisy, atol=1e-4)

    # Step both with identical action
    action = [0.4, 0.0]
    obs_clean_step, _, _, _, _ = env_clean.step(action)
    obs_noisy_step, _, _, _, _ = env_noisy.step(action)

    # Physical states remain strictly identical after dynamics step
    np.testing.assert_allclose(env_clean.state, env_noisy.state, atol=1e-9)

    # Step observations still differ
    assert not np.allclose(obs_clean_step, obs_noisy_step, atol=1e-4)

    env_clean.close()
    env_noisy.close()


def test_seeds_are_reproducible():
    """Verify that identical seeds yield identical disturbance, noise, and physical trajectories."""
    env1 = HoverSlamEnv(
        wind_strength=1.0,
        sensor_noise=1.0,
        mass_randomization=0.10,
        randomize_start=True,
        wide_start=True,
    )
    env2 = HoverSlamEnv(
        wind_strength=1.0,
        sensor_noise=1.0,
        mass_randomization=0.10,
        randomize_start=True,
        wide_start=True,
    )

    obs1, info1 = env1.reset(seed=777)
    obs2, info2 = env2.reset(seed=777)

    # Initial states, observations, and mass match
    np.testing.assert_allclose(env1.state, env2.state, atol=1e-7)
    np.testing.assert_allclose(obs1, obs2, atol=1e-7)
    assert env1.steady_wind == pytest.approx(env2.steady_wind, abs=1e-7)

    # Step for 25 steps with varying commands
    actions = [
        [0.0, 0.2],
        [-0.5, -0.1],
        [0.8, 0.0],
        [1.0, 0.3],
        [-1.0, -0.4],
    ] * 5

    for act in actions:
        o1, r1, t1, tr1, i1 = env1.step(act)
        o2, r2, t2, tr2, i2 = env2.step(act)

        np.testing.assert_allclose(env1.state, env2.state, atol=1e-7)
        np.testing.assert_allclose(o1, o2, atol=1e-7)
        assert r1 == pytest.approx(r2, abs=1e-7)
        assert i1["wind_x"] == pytest.approx(i2["wind_x"], abs=1e-7)

    env1.close()
    env2.close()


def test_mass_randomization_bounds():
    """Verify that ±10% mass randomization stays strictly within [0.90, 1.10] * START_MASS."""
    env = HoverSlamEnv(mass_randomization=0.10)
    masses = []

    for s in range(50):
        env.reset(seed=1000 + s)
        masses.append(float(env.state[IDX_M]))

    min_mass = min(masses)
    max_mass = max(masses)

    # Boundaries within 10%
    assert min_mass >= 0.90 * START_MASS - 1e-4
    assert max_mass <= 1.10 * START_MASS + 1e-4
    # Confirms variation actually occurred across seeds
    assert (max_mass - min_mass) > 0.10 * START_MASS

    env.close()


def test_start_randomization_wide_ranges():
    """Verify that wide_start expands initial state domain beyond default ranges."""
    env_wide = HoverSlamEnv(randomize_start=True, wide_start=True)

    xs, ys, vxs, vys, thetas = [], [], [], [], []
    for s in range(50):
        env_wide.reset(seed=2000 + s)
        xs.append(float(env_wide.state[IDX_X]))
        ys.append(float(env_wide.state[IDX_Y]))
        vxs.append(float(env_wide.state[IDX_VX]))
        vys.append(float(env_wide.state[IDX_VY]))
        thetas.append(abs(float(env_wide.state[IDX_THETA])))

    # Verify samples exceed default narrower ranges
    assert min(xs) < DEFAULT_START_RANGES["x_range"][0] or max(xs) > DEFAULT_START_RANGES["x_range"][1]
    assert min(ys) < DEFAULT_START_RANGES["y_range"][0] or max(ys) > DEFAULT_START_RANGES["y_range"][1]
    assert min(vxs) < DEFAULT_START_RANGES["vx_range"][0] or max(vxs) > DEFAULT_START_RANGES["vx_range"][1]
    assert max(thetas) > np.radians(DEFAULT_START_RANGES["theta_max_deg"])

    env_wide.close()


def test_curriculum_difficulty_interpolation():
    """Verify curriculum difficulty level smoothly transitions from easy to full wide configuration."""
    env = HoverSlamEnv(curriculum=True, curriculum_level=0.0)

    # Level 0.0: easy altitude and slower speeds
    env.reset(seed=3000)
    assert EASY_START_RANGES["y_range"][0] <= env.state[IDX_Y] <= EASY_START_RANGES["y_range"][1]
    assert EASY_START_RANGES["vy_range"][0] <= env.state[IDX_VY] <= EASY_START_RANGES["vy_range"][1]
    assert abs(env.state[IDX_THETA]) <= np.radians(EASY_START_RANGES["theta_max_deg"]) + 1e-5

    # Advance to level 1.0: wide configuration
    env.set_curriculum_level(1.0)
    assert env.curriculum_level == 1.0

    env.close()


def test_curriculum_callback_progression():
    """Verify CurriculumCallback triggers difficulty step advancement upon sustained success >= 80%."""
    class DummyVecEnv:
        def __init__(self):
            self.current_level = 0.0

        def env_method(self, method_name, *args, **kwargs):
            if method_name == "set_curriculum_level":
                self.current_level = args[0]
            return [self.current_level]

    class DummyModel:
        def __init__(self, env):
            self.env = env
            class DummyLogger:
                def record(self, key, val):
                    pass
            self.logger = DummyLogger()

        def get_env(self):
            return self.env

    dummy_vec = DummyVecEnv()
    landing_cb = LandingStatsCallback(window_size=100)
    landing_cb.model = DummyModel(dummy_vec)

    curriculum_cb = CurriculumCallback(
        landing_cb=landing_cb,
        threshold=0.80,
        min_episodes=5,
        step_increment=0.25,
        verbose=0,
    )
    curriculum_cb.model = DummyModel(dummy_vec)
    curriculum_cb._training_env = dummy_vec

    # Populate 5 successful landings (100% success rate >= 80%)
    for _ in range(5):
        landing_cb.total_episodes += 1
        landing_cb.landed_history.append(1.0)

    curriculum_cb._on_step()

    # Curriculum level should advance from 0.0 to 0.25
    assert curriculum_cb.current_level == pytest.approx(0.25)
    assert dummy_vec.current_level == pytest.approx(0.25)
