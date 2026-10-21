"""Classical cascaded PID controller baseline for HoverSlam environment."""

from __future__ import annotations

import numpy as np

from sim.rocket import (
    DRY_MASS,
    DT,
    FUEL_MASS,
    G,
    GIMBAL_MAX,
    MIN_THROTTLE,
    T_MAX,
)

# -----------------------------------------------------------------------------
# Default Controller Gains and Tuning Parameters
# -----------------------------------------------------------------------------
DEFAULT_PID_GAINS: dict[str, float] = {
    # --- Altitude & Velocity Controller (Hoverslam Burn) ---
    "a_brake": 5.2,           # [m/s^2] Target nominal braking deceleration during retro-propulsion
    "safety_margin": 1.05,    # [-] Kinematic stopping distance ignition multiplier
    "v_land": 0.4,            # [m/s] Desired terminal touchdown descent speed offset
    "kp_vy": 3.5,             # [s^-1] Proportional gain on vertical velocity tracking error
    "ki_vy": 0.05,            # [s^-2] Integral gain on vertical velocity tracking error
    "kd_vy": 0.15,            # [-] Derivative gain on vertical velocity tracking error
    "i_vy_limit": 4.0,        # [m/s^2] Anti-windup clamping on vertical acceleration integral

    # --- Horizontal Outer-Loop Controller (Position to Tilt) ---
    "kp_x": 0.015,            # [rad/m] Proportional gain converting lateral offset x to target pitch
    "kd_x": 0.055,            # [rad/(m/s)] Derivative damping gain on lateral velocity vx
    "max_tilt_deg": 12.0,     # [deg] Maximum allowed pitch angle during descent
    "min_tilt_ground_deg": 3.5, # [deg] Residual tilt limit near ground to brake horizontal velocity
    "align_alt": 20.0,        # [m] Altitude below which commanded tilt tapers down

    # --- Attitude Inner-Loop Controller (Tilt to Gimbal) ---
    "kp_theta": 2.5,          # [rad/rad] Proportional gain on pitch angle error (theta - theta_target)
    "kd_theta": 1.8,          # [rad/(rad/s)] Derivative damping gain on angular velocity omega (critically damped)
    "ki_theta": 0.02,         # [rad/(rad*s)] Integral gain on pitch angle error
    "i_theta_limit": 0.08,    # [rad] Anti-windup clamping on attitude error integral
}


class HoverSlamPID:
    """Cascaded PID controller for the 2D rocket landing environment.

    Receives the 10-dimensional normalized observation vector from HoverSlamEnv,
    un-normalizes physical states, and outputs continuous action commands:
    action[0] in [-1.0, 1.0] for engine throttle and action[1] in [-1.0, 1.0] for nozzle gimbal.
    """

    def __init__(
        self,
        gains: dict[str, float] | None = None,
        dt: float = DT,
    ) -> None:
        self.gains = DEFAULT_PID_GAINS.copy()
        if gains is not None:
            self.gains.update(gains)

        self.dt = dt

        # Internal controller states
        self.ignited = False
        self.i_vy_err = 0.0
        self.prev_vy_err = 0.0
        self.i_theta_err = 0.0
        self.prev_theta_err = 0.0

    def reset(self) -> None:
        """Reset internal integrator and controller state for a new episode."""
        self.ignited = False
        self.i_vy_err = 0.0
        self.prev_vy_err = 0.0
        self.i_theta_err = 0.0
        self.prev_theta_err = 0.0

    def unnormalize_obs(self, obs: np.ndarray) -> dict[str, float]:
        """Convert the normalized Gym observation vector back to SI physical state variables.

        Observation definition from envs/landing_env.py:
            [0]: x / 50.0
            [1]: y / 300.0
            [2]: vx / 20.0
            [3]: vy / 80.0
            [4]: sin(theta)
            [5]: cos(theta)
            [6]: omega
            [7]: fuel_fraction = (m - DRY_MASS) / FUEL_MASS
            [8]: prev_action[0]
            [9]: prev_action[1]
        """
        x = float(obs[0]) * 50.0
        y = float(obs[1]) * 300.0
        vx = float(obs[2]) * 20.0
        vy = float(obs[3]) * 80.0
        sin_theta = float(obs[4])
        cos_theta = float(obs[5])
        theta = float(np.arctan2(sin_theta, cos_theta))
        omega = float(obs[6])
        fuel_fraction = float(obs[7])
        m = DRY_MASS + fuel_fraction * FUEL_MASS

        return {
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "theta": theta,
            "omega": omega,
            "m": m,
        }

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """Compute normalized [throttle, gimbal] action from observation vector.

        Returns:
            np.ndarray: shape (2,) in [-1.0, 1.0].
        """
        state = self.unnormalize_obs(obs)
        x = state["x"]
        y = state["y"]
        vx = state["vx"]
        vy = state["vy"]
        theta = state["theta"]
        omega = state["omega"]
        m = state["m"]

        a_brake = self.gains["a_brake"]
        v_land = self.gains["v_land"]
        margin = self.gains["safety_margin"]

        # --- 1. Hoverslam Ignition Condition ---
        # Stopping distance to decelerate from vy < 0 to 0 at a_brake
        if vy < 0.0:
            s_stop = (vy ** 2) / (2.0 * a_brake) * margin
        else:
            s_stop = 0.0

        if not self.ignited and y <= s_stop and vy < -v_land:
            self.ignited = True

        # --- 2. Altitude & Velocity Control ---
        if not self.ignited:
            # Engine remains off during free fall
            action_0 = -1.0
        else:
            # Target descent velocity profile as a function of altitude
            target_speed = float(np.sqrt(max(0.0, 2.0 * a_brake * max(0.0, y)))) + v_land
            v_target = -target_speed

            # Velocity tracking error (positive when falling faster than planned)
            vy_err = v_target - vy

            # PID on vertical velocity error
            self.i_vy_err += vy_err * self.dt
            self.i_vy_err = float(np.clip(
                self.i_vy_err,
                -self.gains["i_vy_limit"],
                self.gains["i_vy_limit"],
            ))
            d_vy_err = (vy_err - self.prev_vy_err) / self.dt
            self.prev_vy_err = vy_err

            a_pid = (
                a_brake
                + self.gains["kp_vy"] * vy_err
                + self.gains["ki_vy"] * self.i_vy_err
                + self.gains["kd_vy"] * d_vy_err
            )

            # Required thrust to offset gravity and supply commanded acceleration along thrust axis
            cos_tilt = max(0.5, float(np.cos(theta)))
            total_accel_req = G + a_pid
            T_req = (m * total_accel_req) / cos_tilt

            # Convert required thrust to throttle fraction [MIN_THROTTLE, 1.0]
            throttle_raw = T_req / T_MAX
            throttle = float(np.clip(throttle_raw, MIN_THROTTLE, 1.0))

            # Linearly map throttle from [MIN_THROTTLE, 1.0] back to action_0 in [-0.5, 1.0]
            norm_th = (throttle - MIN_THROTTLE) / (1.0 - MIN_THROTTLE)
            action_0 = -0.5 + norm_th * 1.5
            action_0 = float(np.clip(action_0, -0.5, 1.0))

        # --- 3. Horizontal Outer-Loop Control (Position -> Target Tilt) ---
        theta_cmd = -(self.gains["kp_x"] * x + self.gains["kd_x"] * vx)

        # Smoothly taper allowed tilt near ground to ensure upright touchdown (|theta| < 5 deg)
        max_tilt_rad = float(np.radians(self.gains["max_tilt_deg"]))
        min_tilt_rad = float(np.radians(self.gains["min_tilt_ground_deg"]))
        align_alt = self.gains["align_alt"]

        if y < align_alt:
            fraction = max(0.0, y / align_alt)
            max_tilt_rad = min_tilt_rad + (max_tilt_rad - min_tilt_rad) * fraction

        theta_target = float(np.clip(theta_cmd, -max_tilt_rad, max_tilt_rad))

        # --- 4. Attitude Inner-Loop Control (Tilt -> Gimbal) ---
        theta_err = theta - theta_target

        self.i_theta_err += theta_err * self.dt
        self.i_theta_err = float(np.clip(
            self.i_theta_err,
            -self.gains["i_theta_limit"],
            self.gains["i_theta_limit"],
        ))

        delta = (
            self.gains["kp_theta"] * theta_err
            + self.gains["kd_theta"] * omega
            + self.gains["ki_theta"] * self.i_theta_err
        )
        delta_clamped = float(np.clip(delta, -GIMBAL_MAX, GIMBAL_MAX))

        # Linearly map gimbal angle [-GIMBAL_MAX, GIMBAL_MAX] to action_1 in [-1.0, 1.0]
        action_1 = float(np.clip(delta_clamped / GIMBAL_MAX, -1.0, 1.0))

        return np.array([action_0, action_1], dtype=np.float32)
