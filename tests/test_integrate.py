"""RK4 and simulation loop tests."""

import numpy as np

from uav_sim.dynamics import gravity_body, rigid_body_derivatives
from uav_sim.integrate import rk4_step, simulate
from uav_sim.state import IDX_QUAT, State, make_state


def test_rk4_error_reduces_by_fourth_order_factor():
    def deriv(_t, x):
        return -x

    def integrate(dt):
        x = np.array([1.0])
        t = 0.0
        for _ in range(int(round(1.0 / dt))):
            x = rk4_step(deriv, t, x, dt)
            t += dt
        return abs(x[0] - np.exp(-1.0))

    coarse = integrate(0.2)
    fine = integrate(0.1)
    assert coarse / fine > 12.0


def test_ballistic_step_convergence_between_half_steps(cfg):
    x0 = make_state(
        [0.0, 0.0, -1000.0],
        [25.0, 0.0, -5.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    )

    def gravity_only(_t, x, _u, local_cfg, _rng):
        return rigid_body_derivatives(
            x,
            gravity_body(x[IDX_QUAT], local_cfg.mass.m),
            np.zeros(3),
            local_cfg,
        )

    coarse = simulate(x0, None, 30.0, cfg, gravity_only, log_every=50)
    fine_cfg = cfg.model_copy(
        update={"integration": cfg.integration.model_copy(update={"dt": 0.001})}
    )
    fine = simulate(x0, None, 30.0, fine_cfg, gravity_only, log_every=100)

    np.testing.assert_allclose(coarse.x, fine.x, rtol=1e-3, atol=1e-9)


def test_quaternion_norm_is_maintained_for_constant_rate(cfg):
    omega = np.array([0.1, 0.2, 0.3])
    x0 = make_state(
        [0.0, 0.0, -1000.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        omega,
    )

    def constant_rate(_t, x, _u, _cfg, _rng):
        xdot = np.zeros_like(x)
        xdot[IDX_QUAT] = (
            0.5
            * np.array(
                [
                    [0.0, -omega[0], -omega[1], -omega[2]],
                    [omega[0], 0.0, omega[2], -omega[1]],
                    [omega[1], -omega[2], 0.0, omega[0]],
                    [omega[2], omega[1], -omega[0], 0.0],
                ]
            )
            @ x[IDX_QUAT]
        )
        return xdot

    result = simulate(x0, None, 100.0, cfg, constant_rate, log_every=50)
    norms = np.linalg.norm(result.x[:, IDX_QUAT], axis=1)

    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-10)


def test_simulate_is_deterministic(cfg):
    x0 = make_state(
        [0.0, 0.0, -1000.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    )

    def inertial(_t, x, _u, _cfg, _rng):
        return np.zeros_like(x)

    first = simulate(x0, None, 1.0, cfg, inertial, log_every=10)
    second = simulate(x0, None, 1.0, cfg, inertial, log_every=10)

    np.testing.assert_array_equal(first.t, second.t)
    np.testing.assert_array_equal(first.x, second.x)


def test_early_termination_returns_result_for_nan(cfg):
    x0 = make_state(
        [0.0, 0.0, -1000.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    )

    def bad_plant(_t, x, _u, _cfg, _rng):
        xdot = np.zeros_like(x)
        xdot[0] = np.nan
        return xdot

    result = simulate(x0, None, 1.0, cfg, bad_plant)

    assert result.terminated_early
    assert result.termination_reason is not None


def test_sim_result_to_dict(cfg):
    x0 = make_state(
        [0.0, 0.0, -100.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    )

    def inertial(_t, x, _u, _cfg, _rng):
        return np.zeros_like(x)

    result = simulate(x0, None, 0.0, cfg, inertial)
    data = result.to_dict()

    assert data["x"] == [State.from_array(x0).to_array().tolist()]
