"""Linear Kalman filter for 2D rocket state estimation in HoverSlam."""

from __future__ import annotations

import numpy as np

from sim.rocket import (
    A,
    CD,
    DT,
    G,
    HEIGHT,
    L,
    RHO,
    START_MASS,
    T_MAX,
    process_action,
)

# -----------------------------------------------------------------------------
# Default Noise Covariances
# -----------------------------------------------------------------------------
# Measurement noise variances in physical SI units, precisely matched to
# DEFAULT_SENSOR_SIGMAS in envs/landing_env.py at nominal sensor_noise = 1.0:
#
#   Observation Signal           Normalized Sigma  Physical Scale  Physical Std Dev (sigma)   Measurement Variance (R)
#   --------------------------   ----------------  --------------  ------------------------   ------------------------
#   obs[0]: x / 50.0             0.020             50.0 m          1.00 m                     1.000000 m^2
#   obs[1]: y / 300.0            0.010             300.0 m         3.00 m                     9.000000 m^2
#   obs[2]: vx / 20.0            0.020             20.0 m/s        0.40 m/s                   0.160000 (m/s)^2
#   obs[3]: vy / 80.0            0.010             80.0 m/s        0.80 m/s                   0.640000 (m/s)^2
#   obs[4,5]: sin/cos(theta)     0.015             1.0 rad         ~0.015 rad (~0.86 deg)     0.000225 rad^2
#   obs[6]: omega                0.025             1.0 rad/s       0.025 rad/s                0.000625 (rad/s)^2
#
DEFAULT_KALMAN_COVARIANCES: dict[str, dict[str, float]] = {
    "measurement_noise": {
        "x": 1.0,           # [m^2] Lateral position measurement variance
        "y": 9.0,           # [m^2] Altitude measurement variance
        "vx": 0.16,         # [(m/s)^2] Horizontal velocity measurement variance
        "vy": 0.64,         # [(m/s)^2] Vertical velocity measurement variance
        "theta": 0.000225,  # [rad^2] Pitch attitude measurement variance (~0.86 deg std)
        "omega": 0.000625,  # [(rad/s)^2] Angular rate measurement variance
    },
    # Process noise variances capturing unmodeled dynamics: atmospheric wind gusts,
    # aerodynamic turbulence, mass depletion drift, and engine thrust jitter.
    "process_noise": {
        "x": 0.005,         # [m^2]
        "y": 0.005,         # [m^2]
        "vx": 0.10,         # [(m/s)^2]
        "vy": 0.10,         # [(m/s)^2]
        "theta": 0.0001,    # [rad^2]
        "omega": 0.001,     # [(rad/s)^2]
    },
    # Initial state estimation covariance P0 for cold-start initialization
    "initial_covariance": {
        "x": 1.0,
        "y": 9.0,
        "vx": 0.16,
        "vy": 0.64,
        "theta": 0.000225,
        "omega": 0.000625,
    },
}


class HoverSlamKalmanFilter:
    """Linear Kalman filter estimating [x, y, vx, vy, theta, omega] from noisy observations.

    Uses the same gravity-aware, constant-acceleration kinematic model as the rocket simulation,
    with discrete semi-implicit Euler integration matching sim/rocket.py:
        vx_{k} = vx_{k-1} + ax * dt
        vy_{k} = vy_{k-1} + ay * dt
        omega_{k} = omega_{k-1} + alpha * dt
        x_{k} = x_{k-1} + vx_{k} * dt = x_{k-1} + vx_{k-1} * dt + ax * dt^2
        y_{k} = y_{k-1} + vy_{k} * dt = y_{k-1} + vy_{k-1} * dt + ay * dt^2
        theta_{k} = theta_{k-1} + omega_{k} * dt = theta_{k-1} + omega_{k-1} * dt + alpha * dt^2
    """

    STATE_DIM: int = 6  # [x, y, vx, vy, theta, omega]

    def __init__(
        self,
        dt: float = DT,
        sensor_noise: float = 1.0,
        enable_drag: bool = True,
        nominal_mass: float = START_MASS,
        cov_config: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.dt = float(dt)
        self.sensor_noise = float(sensor_noise)
        self.enable_drag = bool(enable_drag)
        self.nominal_mass = float(nominal_mass)

        cfg = DEFAULT_KALMAN_COVARIANCES.copy()
        if cov_config is not None:
            for k, sub in cov_config.items():
                if k in cfg:
                    cfg[k].update(sub)
                else:
                    cfg[k] = sub.copy()

        self.cov_config = cfg

        # Base covariance matrices
        self.R_base = np.diag([
            cfg["measurement_noise"]["x"],
            cfg["measurement_noise"]["y"],
            cfg["measurement_noise"]["vx"],
            cfg["measurement_noise"]["vy"],
            cfg["measurement_noise"]["theta"],
            cfg["measurement_noise"]["omega"],
        ]).astype(np.float64)

        self.Q = np.diag([
            cfg["process_noise"]["x"],
            cfg["process_noise"]["y"],
            cfg["process_noise"]["vx"],
            cfg["process_noise"]["vy"],
            cfg["process_noise"]["theta"],
            cfg["process_noise"]["omega"],
        ]).astype(np.float64)

        self.P0 = np.diag([
            cfg["initial_covariance"]["x"],
            cfg["initial_covariance"]["y"],
            cfg["initial_covariance"]["vx"],
            cfg["initial_covariance"]["vy"],
            cfg["initial_covariance"]["theta"],
            cfg["initial_covariance"]["omega"],
        ]).astype(np.float64)

        # Transition matrix F: 6x6
        self.F = np.eye(self.STATE_DIM, dtype=np.float64)
        self.F[0, 2] = self.dt
        self.F[1, 3] = self.dt
        self.F[4, 5] = self.dt

        # Measurement matrix H: 6x6 (direct observation)
        self.H = np.eye(self.STATE_DIM, dtype=np.float64)

        # Internal state vector [x, y, vx, vy, theta, omega]
        self.x = np.zeros(self.STATE_DIM, dtype=np.float64)
        self.P = self.P0.copy()

    def _get_measurement_covariance(self) -> np.ndarray:
        """Compute R scaled by current sensor_noise level."""
        if self.sensor_noise <= 0.0:
            # Numerical regularization floor when sensor noise is zero
            return 1e-6 * np.eye(self.STATE_DIM, dtype=np.float64)
        return (self.sensor_noise ** 2) * self.R_base

    def reset(
        self,
        initial_state: np.ndarray | None = None,
        sensor_noise: float | None = None,
    ) -> np.ndarray:
        """Reset the filter state and covariance."""
        if sensor_noise is not None:
            self.sensor_noise = float(sensor_noise)

        if initial_state is not None:
            arr = np.asarray(initial_state, dtype=np.float64)
            if arr.shape[0] >= self.STATE_DIM:
                self.x = arr[:self.STATE_DIM].copy()
            elif arr.shape[0] == 4:
                # [x, y, vx, vy] provided
                self.x = np.zeros(self.STATE_DIM, dtype=np.float64)
                self.x[:4] = arr
            else:
                raise ValueError(f"Initial state dimension {arr.shape[0]} not supported.")
        else:
            self.x = np.zeros(self.STATE_DIM, dtype=np.float64)

        self.P = self.P0.copy()
        return self.x.copy()

    def reset_from_obs(
        self,
        raw_obs: np.ndarray,
        sensor_noise: float | None = None,
    ) -> np.ndarray:
        """Initialize filter directly from a Gym 10D observation vector."""
        if sensor_noise is not None:
            self.sensor_noise = float(sensor_noise)

        z = self._unnormalize_obs(raw_obs)
        self.x = z.copy()
        self.P = self.P0.copy()
        return self._normalize_state(raw_obs)

    def _unnormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert 10D normalized observation to physical 6D state measurement."""
        x_meas = float(obs[0]) * 50.0
        y_meas = float(obs[1]) * 300.0
        vx_meas = float(obs[2]) * 20.0
        vy_meas = float(obs[3]) * 80.0
        sin_th = float(obs[4])
        cos_th = float(obs[5])
        theta_meas = float(np.arctan2(sin_th, cos_th))
        omega_meas = float(obs[6])

        return np.array([x_meas, y_meas, vx_meas, vy_meas, theta_meas, omega_meas], dtype=np.float64)

    def _normalize_state(self, raw_obs: np.ndarray) -> np.ndarray:
        """Construct 10D normalized observation with Kalman-filtered estimates."""
        filtered = np.empty(10, dtype=np.float32)
        filtered[0] = np.float32(self.x[0] / 50.0)
        filtered[1] = np.float32(self.x[1] / 300.0)
        filtered[2] = np.float32(self.x[2] / 20.0)
        filtered[3] = np.float32(self.x[3] / 80.0)
        filtered[4] = np.float32(np.sin(self.x[4]))
        filtered[5] = np.float32(np.cos(self.x[4]))
        filtered[6] = np.float32(self.x[5])
        filtered[7] = raw_obs[7]
        filtered[8] = raw_obs[8]
        filtered[9] = raw_obs[9]
        return filtered

    def predict(
        self,
        action: np.ndarray | list[float] | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Propagate state and covariance forward by dt using semi-implicit Euler dynamics."""
        step_dt = self.dt if dt is None else float(dt)

        # 1. Compute control and aerodynamic accelerations
        x, y, vx, vy, theta, omega = self.x

        if action is not None:
            throttle, delta = process_action(action)
            T = throttle * T_MAX
            m = self.nominal_mass
            thrust_angle = theta + delta
            Fx = T * float(np.sin(thrust_angle))
            Fy = T * float(np.cos(thrust_angle)) - m * G
            tau = -L * T * float(np.sin(delta))
            I = m * (HEIGHT ** 2) / 12.0
            ax = Fx / m
            ay = Fy / m
            alpha = tau / I
        else:
            # Gravity-aware free-fall model
            ax = 0.0
            ay = -G
            alpha = 0.0

        if self.enable_drag:
            speed = float(np.hypot(vx, vy))
            drag_mag = 0.5 * RHO * CD * A * speed
            drag_x = drag_mag * vx
            drag_y = drag_mag * vy
            ax -= drag_x / self.nominal_mass
            ay -= drag_y / self.nominal_mass

        # Semi-implicit Euler integration vector u:
        # vx_{k} = vx_{k-1} + ax * dt
        # x_{k} = x_{k-1} + vx_{k} * dt = x_{k-1} + vx_{k-1}*dt + ax * dt^2
        u = np.array([
            ax * (step_dt ** 2),
            ay * (step_dt ** 2),
            ax * step_dt,
            ay * step_dt,
            alpha * (step_dt ** 2),
            alpha * step_dt,
        ], dtype=np.float64)

        # Update F for custom dt if necessary
        if step_dt != self.dt:
            F = np.eye(self.STATE_DIM, dtype=np.float64)
            F[0, 2] = step_dt
            F[1, 3] = step_dt
            F[4, 5] = step_dt
        else:
            F = self.F

        # Propagate state and error covariance
        self.x = F @ self.x + u
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Perform measurement update step from a physical 6D measurement vector."""
        z = np.asarray(measurement, dtype=np.float64)
        if z.shape[0] == 4:
            # 4D measurement [x, y, vx, vy] provided
            z_full = np.zeros(self.STATE_DIM, dtype=np.float64)
            z_full[:4] = z
            z_full[4] = self.x[4]
            z_full[5] = self.x[5]
            z = z_full

        R = self._get_measurement_covariance()

        # Measurement innovation with angular wrapping on pitch angle
        y_innov = z - self.H @ self.x
        # Wrap theta error to [-pi, pi]
        y_innov[4] = float((y_innov[4] + np.pi) % (2.0 * np.pi) - np.pi)

        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_innov

        # Joseph form covariance update for numerical symmetry and positive definiteness
        I_KH = np.eye(self.STATE_DIM, dtype=np.float64) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        return self.x.copy()

    def filter_obs(
        self,
        raw_obs: np.ndarray,
        action: np.ndarray | list[float] | None = None,
        dt: float | None = None,
    ) -> np.ndarray:
        """Run complete predict + update step and return filtered 10D observation."""
        self.predict(action=action, dt=dt)
        z = self._unnormalize_obs(raw_obs)
        self.update(z)
        return self._normalize_state(raw_obs)

    @property
    def state(self) -> np.ndarray:
        """Current 6D state estimate [x, y, vx, vy, theta, omega]."""
        return self.x.copy()

    @property
    def pos_vel(self) -> np.ndarray:
        """Current 4D position and velocity estimate [x, y, vx, vy]."""
        return self.x[:4].copy()

    @property
    def covariance(self) -> np.ndarray:
        """Current 6x6 estimation error covariance matrix."""
        return self.P.copy()
