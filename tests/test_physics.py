"""Unit tests for pure-numpy rocket landing physics simulation."""

import numpy as np
import pytest

from sim.rocket import (
    A,
    CD,
    DRY_MASS,
    DT,
    FUEL_MASS,
    G,
    G0,
    GIMBAL_MAX,
    HEIGHT,
    IDX_M,
    IDX_OMEGA,
    IDX_THETA,
    IDX_VX,
    IDX_VY,
    IDX_X,
    IDX_Y,
    ISP,
    L,
    MIN_THROTTLE,
    RHO,
    START_MASS,
    T_MAX,
    is_touchdown,
    process_action,
    reset,
    step,
)


def test_free_fall():
    """1. Free fall (engine off, no drag) matches h = 0.5 * g * t^2 within tolerance.

    In semi-implicit Euler integration:
        v[k+1] = v[k] - g * dt
        y[k+1] = y[k] + v[k+1] * dt
    Over N steps of size dt (total time t = N * dt), the discrete displacement is:
        h_discrete = 0.5 * g * t^2 + 0.5 * g * t * dt
    We verify the simulated displacement matches the analytical continuous formula
    h = 0.5 * g * t^2 within the expected Euler discretization bound (0.5 * g * t * dt).
    """
    initial_y = 500.0
    state = np.array([0.0, initial_y, 0.0, 0.0, 0.0, 0.0, START_MASS], dtype=np.float64)
    action = [0.0, 0.0]  # engine completely off

    t_total = 2.0  # seconds
    steps = int(round(t_total / DT))

    for _ in range(steps):
        state = step(state, action, dt=DT, enable_drag=False, burn_fuel=False)

    distance_fallen = initial_y - state[IDX_Y]
    theoretical_h = 0.5 * G * (t_total ** 2)

    # Allow slight discretization error from semi-implicit Euler (0.5 * g * t * dt ≈ 0.20 m)
    assert distance_fallen == pytest.approx(theoretical_h, abs=0.25)
    # Final vertical velocity matches v = -g * t
    assert state[IDX_VY] == pytest.approx(-G * t_total, abs=1e-5)


def test_hover_thrust_holds_altitude():
    """2. A throttle giving thrust equal to weight holds altitude when fuel burn is ignored.

    Weight W = m * g.
    Max thrust T_max = 2.5 * start_mass * g.
    When m = START_MASS, required hover thrust T = START_MASS * g.
    Required throttle = T / T_max = 1 / 2.5 = 0.40, which matches MIN_THROTTLE.
    With vertical attitude (theta=0, delta=0), zero wind, no drag, and fuel burn ignored:
        Fy = T - m * g = 0
        ay = 0
    The rocket must maintain constant altitude and zero vertical velocity.
    """
    initial_y = 200.0
    mass = START_MASS
    state = np.array([0.0, initial_y, 0.0, 0.0, 0.0, 0.0, mass], dtype=np.float64)

    # Calculate exact throttle to balance weight: T = throttle * T_MAX = mass * G
    hover_throttle = (mass * G) / T_MAX  # 1 / 2.5 = 0.40
    action = [hover_throttle, 0.0]

    for _ in range(100):  # Simulate for 2.0 seconds
        state = step(state, action, dt=DT, enable_drag=False, burn_fuel=False)

    assert state[IDX_Y] == pytest.approx(initial_y, abs=1e-5)
    assert state[IDX_VY] == pytest.approx(0.0, abs=1e-5)


def test_positive_gimbal_angular_acceleration_sign():
    """3. Positive gimbal produces the expected angular acceleration sign.

    Physical Explanation:
        - theta = 0 rad represents an upright booster.
        - Positive theta tilts the rocket top towards +x (clockwise).
        - The engine gimbal pivot is located at the base, distance L below the
          vehicle center of mass.
        - A positive gimbal deflection (delta > 0) rotates the engine nozzle towards +x,
          directing thrust force at angle (theta + delta).
        - Pushing the bottom of the booster rightward towards +x induces a
          counter-clockwise (negative) torque about the center of mass:
              tau = -L * T * sin(delta) < 0
        - This negative torque produces negative angular acceleration (alpha = tau / I < 0),
          which drives omega negative and acts as a restoring force against positive tilt.
    """
    # Start upright with zero initial motion
    state = np.array([0.0, 300.0, 0.0, 0.0, 0.0, 0.0, START_MASS], dtype=np.float64)

    # Fire engine with positive gimbal deflection (delta = +5 deg)
    delta_pos = float(np.radians(5.0))
    action = [1.0, delta_pos]

    next_state = step(state, action, dt=DT, enable_drag=False, burn_fuel=False)

    # Angular rate omega must be negative (counter-clockwise restoring motion)
    assert next_state[IDX_OMEGA] < 0.0
    # Tilt angle theta after one integration step must also be negative
    assert next_state[IDX_THETA] < 0.0


def test_fuel_consumption_and_non_negativity():
    """4. Fuel never goes negative; mass decreases when thrust is on and stays constant when off.

    Sub-tests:
        a) Engine off: mass remains strictly constant.
        b) Engine on: mass decreases at theoretical rate m_dot = -T / (Isp * g0).
        c) Fuel depletion: mass stops decreasing at dry_mass; thrust cuts out to zero.
    """
    # a) Engine off: mass does not change
    state_off = np.array([0.0, 300.0, 0.0, 0.0, 0.0, 0.0, START_MASS], dtype=np.float64)
    for _ in range(50):
        state_off = step(state_off, [0.0, 0.0], dt=DT, burn_fuel=True)
    assert state_off[IDX_M] == pytest.approx(START_MASS, abs=1e-9)

    # b) Engine on: mass decreases at -T / (Isp * g0)
    state_on = np.array([0.0, 300.0, 0.0, 0.0, 0.0, 0.0, START_MASS], dtype=np.float64)
    throttle = 1.0
    state_on = step(state_on, [throttle, 0.0], dt=DT, burn_fuel=True)
    expected_m_dot = -(throttle * T_MAX) / (ISP * G0)
    expected_m = START_MASS + expected_m_dot * DT
    assert state_on[IDX_M] == pytest.approx(expected_m, rel=1e-6)

    # c) Depletion: start with only 0.1 kg fuel and step until tank is empty
    tiny_fuel = 0.1
    state_depleting = np.array([0.0, 300.0, 0.0, 0.0, 0.0, 0.0, DRY_MASS + tiny_fuel], dtype=np.float64)
    for _ in range(100):
        state_depleting = step(state_depleting, [1.0, 0.0], dt=DT, burn_fuel=True)

    # Mass must be clamped exactly at DRY_MASS (fuel cannot drop below 0)
    assert state_depleting[IDX_M] == pytest.approx(DRY_MASS, abs=1e-9)
    assert state_depleting[IDX_M] >= DRY_MASS

    # With empty fuel, commanded thrust must produce 0 force (pure gravity acceleration)
    state_empty = step(state_depleting, [1.0, 0.0], dt=DT, enable_drag=False, burn_fuel=True)
    assert state_empty[IDX_M] == DRY_MASS
    delta_vy = state_empty[IDX_VY] - state_depleting[IDX_VY]
    assert delta_vy == pytest.approx(-G * DT, abs=1e-5)


def test_throttle_clamping_and_engine_off():
    """5. Min throttle clamp works; engine-off works.

    Rules:
        - throttle <= 0: engine is off (thrust = 0).
        - 0 < throttle < MIN_THROTTLE: clamped up to MIN_THROTTLE (0.4).
        - throttle > 1.0: clamped down to 1.0.
        - throttle in [MIN_THROTTLE, 1.0]: preserved unchanged.
    """
    # Test action processing directly
    assert process_action([0.0, 0.0])[0] == 0.0
    assert process_action([-0.5, 0.0])[0] == 0.0
    assert process_action([0.1, 0.0])[0] == pytest.approx(MIN_THROTTLE, abs=1e-6)
    assert process_action([0.4, 0.0])[0] == pytest.approx(MIN_THROTTLE, abs=1e-6)
    assert process_action([0.75, 0.0])[0] == pytest.approx(0.75, abs=1e-6)
    assert process_action([1.5, 0.0])[0] == pytest.approx(1.0, abs=1e-6)

    # Test effective acceleration through step() with throttle = 0.1 (clamped to 0.4)
    state = np.array([0.0, 300.0, 0.0, 0.0, 0.0, 0.0, START_MASS], dtype=np.float64)
    state_clamped = step(state, [0.1, 0.0], dt=DT, enable_drag=False, burn_fuel=False)

    expected_thrust = MIN_THROTTLE * T_MAX
    expected_ay = (expected_thrust - START_MASS * G) / START_MASS
    actual_ay = (state_clamped[IDX_VY] - state[IDX_VY]) / DT
    assert actual_ay == pytest.approx(expected_ay, rel=1e-5)


def test_touchdown_detection():
    """Verify touchdown flag detection when y <= 0."""
    above_ground = np.array([0.0, 10.0, 0.0, -10.0, 0.0, 0.0, START_MASS])
    at_ground = np.array([0.0, 0.0, 0.0, -2.0, 0.0, 0.0, START_MASS])
    below_ground = np.array([0.0, -0.5, 0.0, -5.0, 0.0, 0.0, START_MASS])

    assert not is_touchdown(above_ground)
    assert is_touchdown(at_ground)
    assert is_touchdown(below_ground)


def test_reset_bounds():
    """Verify reset state falls within default specified ranges."""
    state = reset(seed=42, randomize=True)
    assert state.shape == (7,)
    assert 290.0 <= state[IDX_Y] <= 310.0
    assert -70.0 <= state[IDX_VY] <= -40.0
    assert -30.0 <= state[IDX_X] <= 30.0
    assert abs(state[IDX_THETA]) <= np.radians(5.0) + 1e-6
    assert state[IDX_M] == pytest.approx(START_MASS, abs=1e-6)

    # Deterministic reset
    state_det = reset(randomize=False)
    assert state_det[IDX_Y] == 300.0
    assert state_det[IDX_VY] == -55.0
    assert state_det[IDX_X] == 0.0
    assert state_det[IDX_THETA] == 0.0
