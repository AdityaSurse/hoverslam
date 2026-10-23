"""Gymnasium environment for 2D rocket landing simulation."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sim.rocket import (
    DIAMETER,
    DRY_MASS,
    DT,
    FUEL_MASS,
    GIMBAL_MAX,
    HEIGHT,
    IDX_M,
    IDX_OMEGA,
    IDX_THETA,
    IDX_VX,
    IDX_VY,
    IDX_X,
    IDX_Y,
    MIN_THROTTLE,
    START_MASS,
    is_touchdown,
    reset as sim_reset,
    step as sim_step,
)

# -----------------------------------------------------------------------------
# Reward Function Hyperparameters
# -----------------------------------------------------------------------------
DEFAULT_REWARD_WEIGHTS: dict[str, float] = {
    "dist_weight": 1.0,           # Weight for normalized distance error in potential
    "speed_weight": 0.5,          # Weight for normalized velocity error in potential
    "tilt_weight": 0.5,           # Weight for normalized tilt angle error in potential
    "progress_scale": 100.0,      # Scaling multiplier for potential difference (prev_err - new_err)
    "fuel_penalty_weight": 0.005, # Penalty multiplier proportional to commanded throttle
    "time_penalty": 0.01,         # Constant penalty per simulation step
    "soft_landing_bonus": 100.0,  # Terminal reward for achieving soft landing criteria
    "crash_penalty": 100.0,       # Base terminal penalty for ground collision outside limits
    "crash_speed_penalty": 1.0,   # Terminal penalty scaled per m/s of vertical impact velocity
    "out_of_bounds_penalty": 100.0, # Terminal penalty for exiting operational boundary
}

# -----------------------------------------------------------------------------
# Disturbance & Domain Randomization Constants
# -----------------------------------------------------------------------------
# Peak wind aerodynamic force in Newtons (1.0 = ~8 m/s equivalent push at peak)
MAX_WIND_FORCE: float = 800.0

# Base standard deviations for each of the 10 observation signals
DEFAULT_SENSOR_SIGMAS: np.ndarray = np.array([
    0.020,  # x: ~1.0m / 50m
    0.010,  # y: ~3.0m / 300m
    0.020,  # vx: ~0.4 m/s / 20 m/s
    0.010,  # vy: ~0.8 m/s / 80 m/s
    0.015,  # sin(theta): ~0.8 deg
    0.015,  # cos(theta): ~0.8 deg
    0.025,  # omega: ~0.025 rad/s
    0.010,  # fuel_fraction: ~1%
    0.010,  # prev_action[0]: ~1%
    0.010,  # prev_action[1]: ~1%
], dtype=np.float32)

# Start state dispersions for nominal, wide, and curriculum-easy configurations
DEFAULT_START_RANGES: dict[str, Any] = {
    "x_range": (-30.0, 30.0),
    "y_range": (290.0, 310.0),
    "vx_range": (-5.0, 5.0),
    "vy_range": (-70.0, -40.0),
    "theta_max_deg": 5.0,
    "omega_range": (-0.05, 0.05),
}

WIDE_START_RANGES: dict[str, Any] = {
    "x_range": (-60.0, 60.0),
    "y_range": (250.0, 350.0),
    "vx_range": (-10.0, 10.0),
    "vy_range": (-80.0, -30.0),
    "theta_max_deg": 10.0,
    "omega_range": (-0.10, 0.10),
}

EASY_START_RANGES: dict[str, Any] = {
    "x_range": (-10.0, 10.0),
    "y_range": (100.0, 150.0),
    "vx_range": (-1.0, 1.0),
    "vy_range": (-20.0, -10.0),
    "theta_max_deg": 1.0,
    "omega_range": (-0.01, 0.01),
}


class HoverSlamEnv(gym.Env):
    """2D Rocket Booster Landing Environment.

    Observation Space:
        Box(10,) float32:
        [0]: x_offset / 50.0   (normalized lateral distance from pad center)
        [1]: y / 300.0          (normalized altitude above ground)
        [2]: vx / 20.0          (normalized horizontal velocity)
        [3]: vy / 80.0          (normalized vertical velocity)
        [4]: sin(theta)         (sine of pitch angle)
        [5]: cos(theta)         (cosine of pitch angle)
        [6]: omega              (pitch angular velocity in rad/s)
        [7]: fuel_fraction      (remaining propellant fraction in [0, 1])
        [8]: prev_action[0]     (previous commanded throttle in [-1, 1])
        [9]: prev_action[1]     (previous commanded gimbal in [-1, 1])

    Action Space:
        Box(2,) float32 in [-1.0, 1.0]:
        action[0]: throttle command
                   if a[0] < -0.5: engine off (throttle = 0.0)
                   else: mapped linearly from [-0.5, 1.0] to [MIN_THROTTLE, 1.0]
        action[1]: gimbal angle command
                   mapped linearly from [-1.0, 1.0] to [-GIMBAL_MAX, GIMBAL_MAX]
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        vertical_only: bool = False,
        wind_strength: float = 0.0,
        sensor_noise: float = 0.0,
        mass_randomization: float = 0.0,
        randomize_start: bool = False,
        wide_start: bool = False,
        curriculum: bool = False,
        curriculum_level: float = 0.0,
        sample_random_params_each_episode: bool = False,
        render_mode: str | None = None,
        reward_weights: dict[str, float] | None = None,
        use_kalman: bool = False,
    ) -> None:
        super().__init__()

        self.vertical_only = bool(vertical_only)
        self.wind_strength = float(wind_strength)
        self.sensor_noise = float(sensor_noise)
        self.mass_randomization = float(mass_randomization)
        self.randomize_start = bool(randomize_start)
        self.wide_start = bool(wide_start)
        self.curriculum = bool(curriculum)
        self.curriculum_level = float(curriculum_level) if curriculum else 1.0
        self.sample_random_params_each_episode = bool(sample_random_params_each_episode)
        self.render_mode = render_mode
        self.use_kalman = bool(use_kalman)

        # Base configuration parameters
        self._base_wind_strength = self.wind_strength
        self._base_sensor_noise = self.sensor_noise
        self._base_mass_randomization = self.mass_randomization

        # Wind dynamics state (Ornstein-Uhlenbeck process)
        self.steady_wind: float = 0.0
        self.gust: float = 0.0
        self.ou_theta: float = 1.0  # Mean reversion rate
        self.ou_sigma: float = 0.5  # Volatility parameter
        self.last_wind_x: float = 0.0

        # Configure reward weights
        self.reward_weights = DEFAULT_REWARD_WEIGHTS.copy()
        if reward_weights is not None:
            self.reward_weights.update(reward_weights)

        # Simulation step duration
        self.dt = DT

        # Observation space: 10 continuous variables (roughly normalized to [-1, 1])
        obs_low = np.array(
            [-10.0, -0.5, -10.0, -10.0, -1.0, -1.0, -50.0, 0.0, -1.0, -1.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 50.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action space: 2 continuous commands normalized in [-1, 1]
        act_low = np.full(2, -1.0, dtype=np.float32)
        act_high = np.full(2, 1.0, dtype=np.float32)
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        # Internal state tracking
        self.state: np.ndarray | None = None
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.prev_error: float = 0.0
        self.step_count: int = 0
        self.last_throttle: float = 0.0
        self.last_gimbal: float = 0.0

        # Matplotlib rendering cache
        self.fig = None
        self.ax = None

        # Kalman state estimator
        self.kalman_filter = None
        if self.use_kalman:
            from estimation.kalman import HoverSlamKalmanFilter
            self.kalman_filter = HoverSlamKalmanFilter(
                dt=self.dt,
                sensor_noise=self.sensor_noise,
            )

    def set_curriculum_level(self, level: float) -> None:
        """Update curriculum difficulty level in [0.0, 1.0]."""
        self.curriculum = True
        self.curriculum_level = float(np.clip(level, 0.0, 1.0))

    def _map_action(self, action: np.ndarray | list[float]) -> tuple[float, float]:
        """Convert normalized action [-1, 1]^2 to physical throttle and gimbal angle.

        throttle:
            if a[0] < -0.5: engine off (0.0)
            else: linearly mapped from [-0.5, 1.0] to [MIN_THROTTLE, 1.0]
        gimbal:
            linearly mapped from [-1.0, 1.0] to [-GIMBAL_MAX, GIMBAL_MAX]
        """
        a0 = float(action[0])
        a1 = float(action[1])

        if a0 < -0.5:
            throttle = 0.0
        else:
            norm = (a0 - (-0.5)) / 1.5
            norm = float(np.clip(norm, 0.0, 1.0))
            throttle = MIN_THROTTLE + norm * (1.0 - MIN_THROTTLE)

        gimbal = float(np.clip(a1, -1.0, 1.0)) * GIMBAL_MAX

        if self.vertical_only:
            gimbal = 0.0

        return throttle, gimbal

    def _compute_error(self, state: np.ndarray) -> float:
        """Calculate weighted potential error of rocket state toward target landing."""
        x = float(state[IDX_X])
        y = max(0.0, float(state[IDX_Y]))
        vx = float(state[IDX_VX])
        vy = float(state[IDX_VY])
        theta = float(state[IDX_THETA])

        norm_dist = float(np.sqrt((x / 50.0) ** 2 + (y / 300.0) ** 2))
        norm_speed = float(np.sqrt((vx / 20.0) ** 2 + (vy / 80.0) ** 2))
        norm_tilt = float(abs(theta) / np.radians(5.0))

        error = (
            self.reward_weights["dist_weight"] * norm_dist
            + self.reward_weights["speed_weight"] * norm_speed
            + self.reward_weights["tilt_weight"] * norm_tilt
        )
        return float(error)

    def _get_obs(self) -> np.ndarray:
        """Construct normalized observation array with optional sensor noise."""
        assert self.state is not None, "Environment must be reset before getting observation."

        x = float(self.state[IDX_X])
        y = float(self.state[IDX_Y])
        vx = float(self.state[IDX_VX])
        vy = float(self.state[IDX_VY])
        theta = float(self.state[IDX_THETA])
        omega = float(self.state[IDX_OMEGA])
        m = float(self.state[IDX_M])

        fuel_remaining = max(0.0, m - DRY_MASS)
        fuel_fraction = float(np.clip(fuel_remaining / FUEL_MASS, 0.0, 1.0))

        obs = np.array([
            x / 50.0,
            y / 300.0,
            vx / 20.0,
            vy / 80.0,
            float(np.sin(theta)),
            float(np.cos(theta)),
            omega,
            fuel_fraction,
            float(self.prev_action[0]),
            float(self.prev_action[1]),
        ], dtype=np.float32)

        if self.sensor_noise > 0.0:
            sigmas = self.sensor_noise * DEFAULT_SENSOR_SIGMAS
            noise = self.np_random.normal(0.0, sigmas, size=obs.shape).astype(np.float32)
            obs = obs + noise

        return obs

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # 1. Parameter sampling for episode-level domain randomization
        if self.sample_random_params_each_episode:
            self.wind_strength = float(self.np_random.uniform(0.0, 1.0))
            self.sensor_noise = float(self.np_random.uniform(0.0, 1.0))
            self.mass_randomization = 0.10
            self.randomize_start = True
            self.wide_start = True

            # If curriculum is active, scale difficulty by curriculum_level
            if self.curriculum:
                self.wind_strength *= self.curriculum_level
                self.sensor_noise *= self.curriculum_level
                self.mass_randomization *= self.curriculum_level

        # 2. Wind dynamics initialization (steady component + zero initial gust)
        if self.wind_strength > 0.0:
            self.steady_wind = float(self.np_random.uniform(-0.6, 0.6))
        else:
            self.steady_wind = 0.0
        self.gust = 0.0
        self.last_wind_x = 0.0

        # 3. Mass randomization (±10% on start mass when mass_randomization = 0.10)
        if self.mass_randomization > 0.0:
            delta = float(self.np_random.uniform(-self.mass_randomization, self.mass_randomization))
            total_start_mass = START_MASS * (1.0 + delta)
            fuel_mass = max(10.0, total_start_mass - DRY_MASS)
        else:
            fuel_mass = FUEL_MASS

        # 4. Physical state initialization
        if self.curriculum:
            c = float(np.clip(self.curriculum_level, 0.0, 1.0))
            x_min = (1.0 - c) * EASY_START_RANGES["x_range"][0] + c * WIDE_START_RANGES["x_range"][0]
            x_max = (1.0 - c) * EASY_START_RANGES["x_range"][1] + c * WIDE_START_RANGES["x_range"][1]
            y_min = (1.0 - c) * EASY_START_RANGES["y_range"][0] + c * WIDE_START_RANGES["y_range"][0]
            y_max = (1.0 - c) * EASY_START_RANGES["y_range"][1] + c * WIDE_START_RANGES["y_range"][1]
            vx_min = (1.0 - c) * EASY_START_RANGES["vx_range"][0] + c * WIDE_START_RANGES["vx_range"][0]
            vx_max = (1.0 - c) * EASY_START_RANGES["vx_range"][1] + c * WIDE_START_RANGES["vx_range"][1]
            vy_min = (1.0 - c) * EASY_START_RANGES["vy_range"][0] + c * WIDE_START_RANGES["vy_range"][0]
            vy_max = (1.0 - c) * EASY_START_RANGES["vy_range"][1] + c * WIDE_START_RANGES["vy_range"][1]
            theta_max = (1.0 - c) * EASY_START_RANGES["theta_max_deg"] + c * WIDE_START_RANGES["theta_max_deg"]
            om_min = (1.0 - c) * EASY_START_RANGES["omega_range"][0] + c * WIDE_START_RANGES["omega_range"][0]
            om_max = (1.0 - c) * EASY_START_RANGES["omega_range"][1] + c * WIDE_START_RANGES["omega_range"][1]

            sim_seed = int(self.np_random.integers(0, 2**31 - 1))
            self.state = sim_reset(
                seed=sim_seed,
                randomize=True,
                x_range=(x_min, x_max),
                y_range=(y_min, y_max),
                vx_range=(vx_min, vx_max),
                vy_range=(vy_min, vy_max),
                theta_max_deg=theta_max,
                omega_range=(om_min, om_max),
                fuel_mass=fuel_mass,
            )
        elif self.randomize_start:
            ranges = WIDE_START_RANGES if self.wide_start else DEFAULT_START_RANGES
            sim_seed = int(self.np_random.integers(0, 2**31 - 1))
            self.state = sim_reset(
                seed=sim_seed,
                randomize=True,
                x_range=ranges["x_range"],
                y_range=ranges["y_range"],
                vx_range=ranges["vx_range"],
                vy_range=ranges["vy_range"],
                theta_max_deg=ranges["theta_max_deg"],
                omega_range=ranges["omega_range"],
                fuel_mass=fuel_mass,
            )
        else:
            self.state = sim_reset(randomize=False, fuel_mass=fuel_mass)

        # Enforce vertical constraints if enabled
        if self.vertical_only:
            self.state[IDX_X] = 0.0
            self.state[IDX_VX] = 0.0
            self.state[IDX_THETA] = 0.0
            self.state[IDX_OMEGA] = 0.0

        self.prev_action = np.zeros(2, dtype=np.float32)
        self.last_throttle = 0.0
        self.last_gimbal = 0.0
        self.step_count = 0
        self.prev_error = self._compute_error(self.state)

        obs = self._get_obs()
        if self.use_kalman:
            if self.kalman_filter is None:
                from estimation.kalman import HoverSlamKalmanFilter
                self.kalman_filter = HoverSlamKalmanFilter(
                    dt=self.dt,
                    sensor_noise=self.sensor_noise,
                )
            obs = self.kalman_filter.reset_from_obs(obs, sensor_noise=self.sensor_noise)

        info = {
            "landed": False,
            "touchdown_vy": None,
            "touchdown_vx": None,
            "touchdown_tilt": None,
            "distance_from_pad": float(np.hypot(self.state[IDX_X], self.state[IDX_Y])),
            "fuel_remaining": float(max(0.0, self.state[IDX_M] - DRY_MASS)),
            "step_count": 0,
            "wind_x": 0.0,
            "curriculum_level": self.curriculum_level,
        }

        return obs, info

    def step(
        self,
        action: np.ndarray | list[float],
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step the simulation forward by one time step."""
        assert self.state is not None, "Environment must be reset before stepping."

        self.step_count += 1
        action_arr = np.asarray(action, dtype=np.float32)
        self.prev_action = np.clip(action_arr, -1.0, 1.0)

        # Map normalized action to physical throttle and gimbal
        throttle, gimbal = self._map_action(self.prev_action)
        self.last_throttle = throttle
        self.last_gimbal = gimbal

        # Calculate wind disturbance if enabled (steady wind + Ornstein-Uhlenbeck gust)
        if self.wind_strength > 0.0:
            dw = float(self.np_random.normal(0.0, 1.0))
            self.gust += -self.ou_theta * self.gust * self.dt + self.ou_sigma * np.sqrt(self.dt) * dw
            raw_wind = float(np.clip(self.steady_wind + self.gust, -1.0, 1.0))

            if self.wind_strength <= 1.0:
                wind_x = raw_wind * MAX_WIND_FORCE * self.wind_strength
            else:
                wind_x = raw_wind * self.wind_strength
        else:
            wind_x = 0.0

        self.last_wind_x = wind_x

        # Step physics simulator
        self.state = sim_step(
            self.state,
            [throttle, gimbal],
            dt=self.dt,
            wind_x=wind_x,
            enable_drag=True,
            burn_fuel=True,
        )

        # Enforce vertical locking if enabled
        if self.vertical_only:
            self.state[IDX_X] = 0.0
            self.state[IDX_VX] = 0.0
            self.state[IDX_THETA] = 0.0
            self.state[IDX_OMEGA] = 0.0

        # Compute progress reward
        new_error = self._compute_error(self.state)
        progress_reward = self.reward_weights["progress_scale"] * (self.prev_error - new_error)
        self.prev_error = new_error

        # Compute per-step fuel and time penalties
        fuel_penalty = self.reward_weights["fuel_penalty_weight"] * throttle
        time_penalty = self.reward_weights["time_penalty"]
        reward = float(progress_reward - fuel_penalty - time_penalty)

        # Check termination conditions
        touchdown = is_touchdown(self.state)
        vy_abs = abs(float(self.state[IDX_VY]))
        vx_abs = abs(float(self.state[IDX_VX]))
        tilt_abs = abs(float(self.state[IDX_THETA]))
        x_abs = abs(float(self.state[IDX_X]))
        max_tilt_rad = float(np.radians(5.0))

        terminated = False
        truncated = False
        landed = False

        if touchdown:
            terminated = True
            # Soft landing criteria:
            # vy_abs < 2.0 m/s, vx_abs < 1.0 m/s, |tilt| < 5 deg, |x| < 5.0 m
            if vy_abs < 2.0 and vx_abs < 1.0 and tilt_abs < max_tilt_rad and x_abs < 5.0:
                landed = True
                reward += self.reward_weights["soft_landing_bonus"]
            else:
                landed = False
                reward -= (
                    self.reward_weights["crash_penalty"]
                    + self.reward_weights["crash_speed_penalty"] * vy_abs
                )
        elif abs(float(self.state[IDX_X])) > 200.0 or float(self.state[IDX_Y]) > 500.0:
            # Exceeded spatial boundaries
            terminated = True
            landed = False
            reward -= self.reward_weights["out_of_bounds_penalty"]

        # Truncation check
        if self.step_count >= 1500 and not terminated:
            truncated = True

        obs = self._get_obs()
        if self.use_kalman and self.kalman_filter is not None:
            obs = self.kalman_filter.filter_obs(obs, action=self.prev_action, dt=self.dt)

        info = {
            "landed": landed,
            "touchdown_vy": float(self.state[IDX_VY]) if touchdown else None,
            "touchdown_vx": float(self.state[IDX_VX]) if touchdown else None,
            "touchdown_tilt": float(self.state[IDX_THETA]) if touchdown else None,
            "distance_from_pad": float(np.hypot(self.state[IDX_X], max(0.0, self.state[IDX_Y]))),
            "fuel_remaining": float(max(0.0, self.state[IDX_M] - DRY_MASS)),
            "step_count": self.step_count,
            "wind_x": wind_x,
            "curriculum_level": self.curriculum_level,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render current state as RGB array."""
        if self.render_mode is None or self.state is None:
            return None

        if self.render_mode == "rgb_array":
            return self._render_frame()

        return None

    def _render_frame(self) -> np.ndarray:
        """Draw booster, landing pad, and trajectory status via matplotlib."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Rectangle

        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=100)
        else:
            self.ax.clear()

        ax = self.ax
        assert self.state is not None
        x = float(self.state[IDX_X])
        y = float(self.state[IDX_Y])
        theta = float(self.state[IDX_THETA])
        vx = float(self.state[IDX_VX])
        vy = float(self.state[IDX_VY])
        m = float(self.state[IDX_M])
        fuel = max(0.0, m - DRY_MASS)

        # Plot framing
        ax.set_xlim(-100, 100)
        ax.set_ylim(-20, 350)
        ax.set_aspect("equal")
        ax.set_facecolor("#0b0f19")

        # Ground level
        ax.axhline(0, color="#4a5568", linewidth=2)
        ax.fill_between([-150, 150], [-30, -30], [0, 0], color="#1a202c")

        # Landing pad (-15 m to 15 m) and target zone (-5 m to 5 m)
        pad = Rectangle((-15, -2), 30, 2, color="#718096", zorder=2)
        ax.add_patch(pad)
        target = Rectangle((-5, -2), 10, 2, color="#38a169", zorder=3)
        ax.add_patch(target)

        # Booster geometry: width 1.5 m, height 20.0 m
        hw = DIAMETER / 2.0
        hh = HEIGHT / 2.0
        corners_body = np.array([
            [-hw, -hh],
            [hw, -hh],
            [hw, hh - 2.0],
            [0.0, hh],
            [-hw, hh - 2.0],
        ])

        # Rotate corners according to pitch angle theta
        corners_world = np.zeros_like(corners_body)
        corners_world[:, 0] = x + corners_body[:, 0] * np.cos(theta) + corners_body[:, 1] * np.sin(theta)
        corners_world[:, 1] = y - corners_body[:, 0] * np.sin(theta) + corners_body[:, 1] * np.cos(theta)

        booster_patch = Polygon(corners_world, closed=True, facecolor="#e2e8f0", edgecolor="#94a3b8", zorder=4)
        ax.add_patch(booster_patch)

        # Engine plume when throttle > 0
        if self.last_throttle > 0.0:
            flame_len = 3.0 + 10.0 * self.last_throttle
            flame_angle = theta + self.last_gimbal
            nozzle_x = x - hh * np.sin(theta)
            nozzle_y = y - hh * np.cos(theta)

            flame_tip_x = nozzle_x - flame_len * np.sin(flame_angle)
            flame_tip_y = nozzle_y - flame_len * np.cos(flame_angle)

            flame_base_left_x = nozzle_x - hw * np.cos(flame_angle)
            flame_base_left_y = nozzle_y + hw * np.sin(flame_angle)
            flame_base_right_x = nozzle_x + hw * np.cos(flame_angle)
            flame_base_right_y = nozzle_y - hw * np.sin(flame_angle)

            flame_poly = np.array([
                [flame_base_left_x, flame_base_left_y],
                [flame_tip_x, flame_tip_y],
                [flame_base_right_x, flame_base_right_y],
            ])
            flame_patch = Polygon(flame_poly, closed=True, facecolor="#f97316", edgecolor="#facc15", zorder=3)
            ax.add_patch(flame_patch)

        # Text overlay
        info_text = (
            f"Alt: {y:5.1f} m | Vy: {vy:5.1f} m/s\n"
            f"X:   {x:5.1f} m | Vx: {vx:5.1f} m/s\n"
            f"Tilt:{np.degrees(theta):5.1f} deg | Thr: {self.last_throttle*100:3.0f}%\n"
            f"Fuel:{fuel:5.1f} kg | Step: {self.step_count}"
        )
        ax.text(
            0.03,
            0.95,
            info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            color="#f8fafc",
            bbox=dict(boxstyle="round", facecolor="#1e293b", alpha=0.8),
        )

        ax.set_title("HoverSlam Simulation", color="#f8fafc", fontsize=11)
        ax.tick_params(colors="#94a3b8")

        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        return rgba[:, :, :3].copy()

    def close(self) -> None:
        """Clean up rendering figure if initialized."""
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
            self.ax = None
