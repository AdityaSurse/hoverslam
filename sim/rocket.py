"""2D Rocket Landing Simulation Physics Model.

Sign Conventions:
    - Position (x, y):
        x = 0 is the center of the landing pad (positive x is downrange / right).
        y = 0 is ground level (positive y is altitude above ground).
    - Velocity (vx, vy):
        vx is horizontal velocity (positive towards +x).
        vy is vertical velocity (positive upwards, negative when descending).
    - Attitude (theta, omega):
        theta = 0 rad is perfectly upright (aligned with the vertical +y axis).
        Positive theta tilts the rocket top towards +x (clockwise).
        omega is the angular velocity (d(theta)/dt).
    - Engine Gimbal (delta):
        delta = 0 rad aligns the engine thrust along the rocket body centerline.
        Positive delta rotates the engine nozzle so the thrust direction is (theta + delta).
        Because the engine is located distance L below the center of mass,
        deflecting the nozzle with positive delta directs thrust towards +x,
        creating a negative (counter-clockwise / restoring) torque: tau = -L * T * sin(delta).
"""

from __future__ import annotations

import numpy as np

# -----------------------------------------------------------------------------
# Physical and Simulation Constants (SI units: m, kg, s, N, rad)
# -----------------------------------------------------------------------------
G: float = 9.81  # [m/s^2] Gravitational acceleration
G0: float = 9.81  # [m/s^2] Standard gravitational acceleration for Isp
DRY_MASS: float = 800.0  # [kg] Booster dry mass without propellant
FUEL_MASS: float = 150.0  # [kg] Initial usable landing fuel mass
START_MASS: float = DRY_MASS + FUEL_MASS  # [kg] Initial vehicle mass (950 kg)
T_MAX: float = 2.5 * START_MASS * G  # [N] Maximum engine thrust (T/W ratio = 2.5)
MIN_THROTTLE: float = 0.4  # [-] Minimum throttle limit when engine is active
GIMBAL_MAX: float = float(np.radians(10.0))  # [rad] Maximum gimbal deflection (10 deg)
L: float = 5.0  # [m] Engine distance below vehicle center of mass
HEIGHT: float = 20.0  # [m] Total vehicle height
ISP: float = 300.0  # [s] Engine specific impulse
DT: float = 0.02  # [s] Simulation discrete integration time step (50 Hz)

# Aerodynamic drag constants
RHO: float = 1.225  # [kg/m^3] Sea-level air density
CD: float = 0.8  # [-] Aerodynamic drag coefficient for cylinder body
DIAMETER: float = 1.5  # [m] Booster fuselage reference diameter
A: float = float(np.pi * (DIAMETER / 2.0) ** 2)  # [m^2] Frontal reference area

# State index constants
IDX_X: int = 0  # [m] Downrange position
IDX_Y: int = 1  # [m] Altitude
IDX_VX: int = 2  # [m/s] Downrange horizontal velocity
IDX_VY: int = 3  # [m/s] Vertical velocity
IDX_THETA: int = 4  # [rad] Pitch angle
IDX_OMEGA: int = 5  # [rad/s] Pitch angular rate
IDX_M: int = 6  # [kg] Total mass (dry mass + remaining fuel)
STATE_DIM: int = 7  # Dimension of state vector
ACTION_DIM: int = 2  # Dimension of action vector: [throttle, gimbal_delta]


def process_action(action: np.ndarray | list[float]) -> tuple[float, float]:
    """Validate and clamp throttle and gimbal action commands.

    Args:
        action: Sequence of [throttle, gimbal_delta].
            throttle in [0, 1]: 0 is off; if > 0, clamped to [MIN_THROTTLE, 1.0].
            gimbal_delta: Clamped to [-GIMBAL_MAX, GIMBAL_MAX].

    Returns:
        tuple[float, float]: Processed (throttle, delta).
    """
    throttle_cmd = float(action[0])
    delta_cmd = float(action[1])

    if throttle_cmd <= 0.0:
        throttle = 0.0
    else:
        throttle = float(np.clip(throttle_cmd, MIN_THROTTLE, 1.0))

    delta = float(np.clip(delta_cmd, -GIMBAL_MAX, GIMBAL_MAX))
    return throttle, delta


def is_touchdown(state: np.ndarray) -> bool:
    """Return True if the rocket has reached or passed ground level (y <= 0)."""
    return bool(state[IDX_Y] <= 0.0)


def reset(
    seed: int | None = None,
    randomize: bool = True,
    y_range: tuple[float, float] = (290.0, 310.0),
    vy_range: tuple[float, float] = (-70.0, -40.0),
    x_range: tuple[float, float] = (-30.0, 30.0),
    vx_range: tuple[float, float] = (-5.0, 5.0),
    theta_max_deg: float = 5.0,
    omega_range: tuple[float, float] = (-0.05, 0.05),
    fuel_mass: float = FUEL_MASS,
) -> np.ndarray:
    """Initialize rocket state for simulation.

    Default start:
        y ≈ 300 m, vy between -40 and -70 m/s, x within ±30 m, tilt under 5°.

    Args:
        seed: Random seed for reproducibility.
        randomize: If False, returns deterministic nominal start state.
        y_range: Initial altitude range [m].
        vy_range: Initial vertical velocity range [m/s].
        x_range: Initial horizontal displacement range [m].
        vx_range: Initial horizontal velocity range [m/s].
        theta_max_deg: Maximum initial tilt magnitude [degrees].
        omega_range: Initial angular velocity range [rad/s].
        fuel_mass: Initial propellant mass [kg].

    Returns:
        np.ndarray: Initial state vector [x, y, vx, vy, theta, omega, m].
    """
    if not randomize:
        return np.array([
            0.0,  # x [m]
            300.0,  # y [m]
            0.0,  # vx [m/s]
            -55.0,  # vy [m/s]
            0.0,  # theta [rad]
            0.0,  # omega [rad/s]
            DRY_MASS + fuel_mass,  # m [kg]
        ], dtype=np.float64)

    rng = np.random.default_rng(seed)
    x = float(rng.uniform(x_range[0], x_range[1]))
    y = float(rng.uniform(y_range[0], y_range[1]))
    vx = float(rng.uniform(vx_range[0], vx_range[1]))
    vy = float(rng.uniform(vy_range[0], vy_range[1]))
    max_theta_rad = float(np.radians(theta_max_deg))
    theta = float(rng.uniform(-max_theta_rad, max_theta_rad))
    omega = float(rng.uniform(omega_range[0], omega_range[1]))
    m = float(DRY_MASS + fuel_mass)

    return np.array([x, y, vx, vy, theta, omega, m], dtype=np.float64)


def step(
    state: np.ndarray,
    action: np.ndarray | list[float],
    dt: float = DT,
    wind_x: float = 0.0,
    enable_drag: bool = True,
    burn_fuel: bool = True,
) -> np.ndarray:
    """Step the 2D rocket physics forward by dt using semi-implicit Euler integration.

    Equations:
        T = throttle * T_max (0 if fuel is depleted)
        Fx = T * sin(theta + delta) - drag_x + wind_x
        Fy = T * cos(theta + delta) - m * g - drag_y
        tau = -L * T * sin(delta)
        ax = Fx / m, ay = Fy / m
        alpha = tau / I, with I = m * height^2 / 12
        m_dot = -T / (Isp * g0) (mass never drops below dry_mass)

    Args:
        state: Current state vector [x, y, vx, vy, theta, omega, m].
        action: Control commands [throttle, gimbal_delta].
        dt: Integration time step [s].
        wind_x: External aerodynamic wind force along +x [N].
        enable_drag: Whether aerodynamic drag force is modeled.
        burn_fuel: Whether propellant consumption reduces mass.

    Returns:
        np.ndarray: Updated state vector [x, y, vx, vy, theta, omega, m].
    """
    x = float(state[IDX_X])
    y = float(state[IDX_Y])
    vx = float(state[IDX_VX])
    vy = float(state[IDX_VY])
    theta = float(state[IDX_THETA])
    omega = float(state[IDX_OMEGA])
    m = float(state[IDX_M])

    # Check fuel availability
    usable_fuel = max(0.0, m - DRY_MASS)
    if usable_fuel <= 0.0:
        throttle = 0.0
        delta = 0.0
        T = 0.0
        m_dot = 0.0
    else:
        throttle, delta = process_action(action)
        T = throttle * T_MAX
        m_dot = -T / (ISP * G0) if burn_fuel else 0.0

    # Aerodynamic drag opposing velocity vector
    if enable_drag:
        speed = float(np.hypot(vx, vy))
        drag_mag = 0.5 * RHO * CD * A * speed
        drag_x = drag_mag * vx
        drag_y = drag_mag * vy
    else:
        drag_x = 0.0
        drag_y = 0.0

    # Planar force equations
    thrust_angle = theta + delta
    Fx = T * float(np.sin(thrust_angle)) - drag_x + wind_x
    Fy = T * float(np.cos(thrust_angle)) - m * G - drag_y

    # Torque about center of mass and moment of inertia
    tau = -L * T * float(np.sin(delta))
    I = m * (HEIGHT ** 2) / 12.0

    # Linear and angular accelerations
    ax = Fx / m
    ay = Fy / m
    alpha = tau / I

    # Semi-implicit Euler integration:
    # 1. Update velocities using accelerations
    new_vx = vx + ax * dt
    new_vy = vy + ay * dt
    new_omega = omega + alpha * dt

    # 2. Update positions using updated velocities
    new_x = x + new_vx * dt
    new_y = y + new_vy * dt
    new_theta = theta + new_omega * dt

    # 3. Update vehicle mass, preventing it from dropping below dry_mass
    if burn_fuel and m_dot < 0.0:
        new_m = max(DRY_MASS, m + m_dot * dt)
    else:
        new_m = m

    return np.array([new_x, new_y, new_vx, new_vy, new_theta, new_omega, new_m], dtype=np.float64)


class RocketSim:
    """Stateful wrapper for the pure-numpy rocket simulation."""

    def __init__(self, dt: float = DT):
        self.dt = dt
        self.state: np.ndarray | None = None

    def reset(self, seed: int | None = None, randomize: bool = True, **kwargs) -> np.ndarray:
        """Reset the simulation to an initial state."""
        self.state = reset(seed=seed, randomize=randomize, **kwargs)
        return self.state.copy()

    def step(
        self,
        action: np.ndarray | list[float],
        wind_x: float = 0.0,
        enable_drag: bool = True,
        burn_fuel: bool = True,
    ) -> np.ndarray:
        """Step the simulation forward by self.dt."""
        if self.state is None:
            raise RuntimeError("Simulation must be reset() before calling step().")
        self.state = step(
            self.state,
            action,
            dt=self.dt,
            wind_x=wind_x,
            enable_drag=enable_drag,
            burn_fuel=burn_fuel,
        )
        return self.state.copy()

    @property
    def is_touchdown(self) -> bool:
        """Check if current state has reached ground level."""
        if self.state is None:
            return False
        return is_touchdown(self.state)
