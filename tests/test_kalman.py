"""Unit tests for the linear Kalman filter and its environment integration."""

from __future__ import annotations

import numpy as np
import pytest

from envs.landing_env import (
    DEFAULT_SENSOR_SIGMAS,
    HoverSlamEnv,
)
from estimation.kalman import (
    DEFAULT_KALMAN_COVARIANCES,
    HoverSlamKalmanFilter,
)
from sim.rocket import (
    DT,
    IDX_OMEGA,
    IDX_THETA,
    IDX_VX,
    IDX_VY,
    IDX_X,
    IDX_Y,
    reset as sim_reset,
    step as sim_step,
)


def test_free_fall_filtered_error_clearly_lower_than_raw() -> None:
    """Requirement 3: On a free-fall trajectory with noise,

    the filtered error must be clearly lower than the raw noise error.
    """
    rng = np.random.default_rng(2026)
    noise_scale = 1.0
    num_steps = 150  # 3.0 seconds of free fall

    kf = HoverSlamKalmanFilter(dt=DT, sensor_noise=noise_scale, enable_drag=True)

    # Initial state: altitude 300m, descending at -50 m/s
    true_state = sim_reset(
        seed=42,
        randomize=True,
        y_range=(295.0, 305.0),
        vy_range=(-55.0, -45.0),
        x_range=(-20.0, 20.0),
        vx_range=(-3.0, 3.0),
        theta_max_deg=3.0,
        omega_range=(-0.02, 0.02),
    )

    raw_errors = []
    filtered_errors = []

    for step_idx in range(num_steps):
        # Construct true normalized observation
        obs_clean = np.array([
            true_state[IDX_X] / 50.0,
            true_state[IDX_Y] / 300.0,
            true_state[IDX_VX] / 20.0,
            true_state[IDX_VY] / 80.0,
            float(np.sin(true_state[IDX_THETA])),
            float(np.cos(true_state[IDX_THETA])),
            float(true_state[IDX_OMEGA]),
            1.0,
            0.0,
            0.0,
        ], dtype=np.float32)

        # Apply sensor noise identical to envs/landing_env.py
        sigmas = noise_scale * DEFAULT_SENSOR_SIGMAS
        noise = rng.normal(0.0, sigmas, size=obs_clean.shape).astype(np.float32)
        raw_obs = obs_clean + noise

        # Unnormalized raw measurement in physical SI units
        raw_x = float(raw_obs[0]) * 50.0
        raw_y = float(raw_obs[1]) * 300.0
        raw_vx = float(raw_obs[2]) * 20.0
        raw_vy = float(raw_obs[3]) * 80.0
        raw_theta = float(np.arctan2(raw_obs[4], raw_obs[5]))
        raw_omega = float(raw_obs[6])
        raw_meas = np.array([raw_x, raw_y, raw_vx, raw_vy, raw_theta, raw_omega])

        ground_truth_6 = np.array([
            true_state[IDX_X],
            true_state[IDX_Y],
            true_state[IDX_VX],
            true_state[IDX_VY],
            true_state[IDX_THETA],
            true_state[IDX_OMEGA],
        ])

        if step_idx == 0:
            filtered_obs = kf.reset_from_obs(raw_obs)
        else:
            # Free fall (zero commanded action)
            filtered_obs = kf.filter_obs(raw_obs, action=np.zeros(2))

        filtered_x = float(filtered_obs[0]) * 50.0
        filtered_y = float(filtered_obs[1]) * 300.0
        filtered_vx = float(filtered_obs[2]) * 20.0
        filtered_vy = float(filtered_obs[3]) * 80.0
        filtered_theta = float(np.arctan2(filtered_obs[4], filtered_obs[5]))
        filtered_omega = float(filtered_obs[6])
        filtered_est = np.array([
            filtered_x, filtered_y, filtered_vx, filtered_vy, filtered_theta, filtered_omega
        ])

        raw_errors.append(np.abs(raw_meas - ground_truth_6))
        filtered_errors.append(np.abs(filtered_est - ground_truth_6))

        # Advance true physics in free fall (action = [0, 0])
        true_state = sim_step(true_state, [0.0, 0.0], dt=DT, enable_drag=True, burn_fuel=False)

    raw_mae = np.mean(raw_errors, axis=0)
    filtered_mae = np.mean(filtered_errors, axis=0)

    # Position error reduction
    pos_raw_mae = float(np.mean(raw_mae[:2]))
    pos_filt_mae = float(np.mean(filtered_mae[:2]))
    assert pos_filt_mae < 0.40 * pos_raw_mae, (
        f"Filtered position error ({pos_filt_mae:.3f}m) must be < 40% of raw ({pos_raw_mae:.3f}m)"
    )

    # Velocity error reduction
    vel_raw_mae = float(np.mean(raw_mae[2:4]))
    vel_filt_mae = float(np.mean(filtered_mae[2:4]))
    assert vel_filt_mae < 0.65 * vel_raw_mae, (
        f"Filtered velocity error ({vel_filt_mae:.3f} m/s) must be < 65% of raw ({vel_raw_mae:.3f} m/s)"
    )

    # Overall 6-state error reduction
    total_raw_err = float(np.mean(raw_mae))
    total_filt_err = float(np.mean(filtered_mae))
    error_reduction = (1.0 - total_filt_err / total_raw_err) * 100.0
    assert error_reduction > 50.0, f"Overall error reduction {error_reduction:.1f}% must exceed 50%"


def test_env_use_kalman_flag_default_off() -> None:
    """Verify that use_kalman defaults to False and preserves backwards compatibility."""
    env = HoverSlamEnv()
    assert env.use_kalman is False
    assert env.kalman_filter is None
    env.close()


def test_env_use_kalman_filters_observations_without_touching_true_state() -> None:
    """Verify that when use_kalman=True, the agent receives filtered observations

    while env.state (ground truth) remains strictly untouched.
    """
    env_raw = HoverSlamEnv(sensor_noise=0.5, use_kalman=False, randomize_start=True)
    env_kal = HoverSlamEnv(sensor_noise=0.5, use_kalman=True, randomize_start=True)

    obs_raw, info_raw = env_raw.reset(seed=12345)
    obs_kal, info_kal = env_kal.reset(seed=12345)

    # Both environments initialized from same seed must have identical physical state
    np.testing.assert_allclose(env_raw.state, env_kal.state, atol=1e-10)

    # But the observations differ because env_kal filtered the initial noisy observation
    assert env_kal.use_kalman is True
    assert env_kal.kalman_filter is not None

    # Step both with identical action
    act = np.array([0.5, 0.0], dtype=np.float32)
    next_obs_raw, _, _, _, _ = env_raw.step(act)
    next_obs_kal, _, _, _, _ = env_kal.step(act)

    # True state must still match perfectly
    np.testing.assert_allclose(env_raw.state, env_kal.state, atol=1e-10)

    # Filtered observation is smoother / different from noisy observation
    assert not np.allclose(next_obs_raw, next_obs_kal)

    env_raw.close()
    env_kal.close()


def test_kalman_zero_noise_tracks_true_state() -> None:
    """When sensor_noise=0, filter estimates must closely match perfect observations."""
    env = HoverSlamEnv(sensor_noise=0.0, use_kalman=True, randomize_start=True)
    obs, _ = env.reset(seed=42)

    for _ in range(50):
        act = np.array([0.0, 0.0], dtype=np.float32)
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            break

        # Check that filtered position matches true position within small tolerance
        x_true = env.state[IDX_X]
        y_true = env.state[IDX_Y]
        x_est = float(obs[0]) * 50.0
        y_est = float(obs[1]) * 300.0

        assert abs(x_est - x_true) < 0.5
        assert abs(y_est - y_true) < 1.0

    env.close()


def test_kalman_covariance_shrinks() -> None:
    """Verify that estimation covariance P shrinks over successive measurement updates."""
    kf = HoverSlamKalmanFilter(dt=DT, sensor_noise=1.0)
    init_trace = np.trace(kf.covariance)

    # Run 50 measurement steps with zero motion
    for _ in range(50):
        kf.predict(action=None)
        kf.update(np.zeros(6))

    final_trace = np.trace(kf.covariance)
    assert final_trace < init_trace * 0.5, (
        f"Final covariance trace ({final_trace:.3f}) must be < 50% of initial ({init_trace:.3f})"
    )


def test_kalman_covariances_dict_structure() -> None:
    """Verify that DEFAULT_KALMAN_COVARIANCES contains all required keys with comments."""
    assert "measurement_noise" in DEFAULT_KALMAN_COVARIANCES
    assert "process_noise" in DEFAULT_KALMAN_COVARIANCES
    assert "initial_covariance" in DEFAULT_KALMAN_COVARIANCES

    for key in ["x", "y", "vx", "vy", "theta", "omega"]:
        assert key in DEFAULT_KALMAN_COVARIANCES["measurement_noise"]
        assert key in DEFAULT_KALMAN_COVARIANCES["process_noise"]
        assert key in DEFAULT_KALMAN_COVARIANCES["initial_covariance"]
