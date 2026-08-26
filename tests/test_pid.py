"""PID behaviour tests."""

import numpy as np
import pytest

from uav_sim.control.pid import PID


def _simulate_first_order(pid: PID, setpoint: float, steps: int = 3000) -> float:
    y = 0.0
    dt = 0.01
    for _step in range(steps):
        u = pid.update(setpoint, y, dt)
        y += dt * (-y + u)
    return y


def test_p_control_first_order_steady_state_error():
    pid = PID(kp=2.0, ki=0.0, kd=0.0)
    y = _simulate_first_order(pid, 1.0)

    assert y == pytest.approx(2.0 / 3.0, rel=1e-3)


def test_integral_control_removes_steady_state_error():
    pid = PID(kp=2.0, ki=1.0, kd=0.0)
    y = _simulate_first_order(pid, 1.0)

    assert y == pytest.approx(1.0, abs=2e-3)


def test_derivative_on_measurement_has_no_setpoint_kick():
    pid = PID(kp=0.0, ki=0.0, kd=0.5, N=20.0)
    before = pid.update(0.0, 0.0, 0.01)
    after = pid.update(10.0, 0.0, 0.01)
    wrong_derivative_on_error = 0.5 * (10.0 - 0.0) / 0.01

    assert before == pytest.approx(0.0)
    assert after == pytest.approx(0.0)
    assert abs(after) < wrong_derivative_on_error / 10.0


def test_anti_windup_reduces_release_overshoot():
    dt = 0.01
    with_aw = PID(0.5, 1.0, 0.0, limits=(-0.2, 0.2), Tt=0.1)
    no_aw = PID(0.5, 1.0, 0.0, limits=(-0.2, 0.2), Tt=np.inf)
    y_aw = 0.0
    y_no = 0.0
    for _step in range(1000):
        y_aw += dt * (-y_aw + with_aw.update(10.0, y_aw, dt))
        y_no += dt * (-y_no + no_aw.update(10.0, y_no, dt))

    with_aw.limits = (-10.0, 10.0)
    no_aw.limits = (-10.0, 10.0)
    aw_hist = []
    no_hist = []
    for _step in range(1000):
        y_aw += dt * (-y_aw + with_aw.update(0.0, y_aw, dt))
        y_no += dt * (-y_no + no_aw.update(0.0, y_no, dt))
        aw_hist.append(y_aw)
        no_hist.append(y_no)

    assert (
        max(abs(value) for value in aw_hist)
        < max(abs(value) for value in no_hist) / 5.0
    )


def test_reset_dt_zero_and_determinism():
    pid = PID(1.0, 0.5, 0.1, limits=(-1.0, 1.0))
    assert np.isfinite(pid.update(1.0, 0.0, 0.0))
    output = pid.update(1.0, 0.2, 0.01)
    pid.reset()
    assert pid.integral == 0.0
    assert pid.derivative == 0.0
    assert pid.previous_measurement is None
    assert pid.update(1.0, 0.2, 0.01) == output
