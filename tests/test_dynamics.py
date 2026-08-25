"""Rigid-body derivative tests."""

import numpy as np
import pytest

from uav_sim.config import AircraftConfig
from uav_sim.constants import G0
from uav_sim.dynamics import gravity_body, rigid_body_derivatives
from uav_sim.rotations import euler_to_quat
from uav_sim.state import IDX_OMEGA, IDX_VEL, make_state


def test_zero_force_zero_moment_preserves_velocities(cfg):
    x = make_state(
        pos_n=[0.0, 0.0, 0.0],
        vel_b=[25.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        omega_b=[0.0, 0.0, 0.0],
    )
    xdot = rigid_body_derivatives(x, np.zeros(3), np.zeros(3), cfg)

    np.testing.assert_allclose(xdot[IDX_VEL], np.zeros(3))
    np.testing.assert_allclose(xdot[IDX_OMEGA], np.zeros(3))
    np.testing.assert_allclose(xdot[:3], [25.0, 0.0, 0.0])


def test_pure_body_x_force_produces_exact_forward_acceleration(cfg):
    x = make_state(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    )
    force = np.array([13.5, 0.0, 0.0])

    xdot = rigid_body_derivatives(x, force, np.zeros(3), cfg)

    assert xdot[IDX_VEL][0] == pytest.approx(force[0] / cfg.mass.m)


def test_coriolis_sign_for_y_acceleration(cfg):
    x = make_state(
        pos_n=[0.0, 0.0, 0.0],
        vel_b=[25.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        omega_b=[0.0, 0.0, 0.2],
    )
    xdot = rigid_body_derivatives(x, np.zeros(3), np.zeros(3), cfg)

    assert xdot[IDX_VEL][1] == pytest.approx(-0.2 * 25.0)


def test_gyroscopic_term_changes_z_rate_for_asymmetric_body(cfg):
    x = make_state(
        pos_n=[0.0, 0.0, 0.0],
        vel_b=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        omega_b=[1.0, 1.0, 0.0],
    )

    xdot = rigid_body_derivatives(x, np.zeros(3), np.zeros(3), cfg)

    assert abs(xdot[IDX_OMEGA][2]) > 0.0


def test_gravity_body_frame_signs(cfg):
    level = gravity_body(np.array([1.0, 0.0, 0.0, 0.0]), cfg.mass.m)
    pitch_up = gravity_body(euler_to_quat(0.0, np.deg2rad(90.0), 0.0), cfg.mass.m)

    np.testing.assert_allclose(level, [0.0, 0.0, cfg.mass.m * G0], atol=1e-12)
    np.testing.assert_allclose(pitch_up, [-cfg.mass.m * G0, 0.0, 0.0], atol=1e-12)


def test_derivative_does_not_modify_inputs(cfg):
    x = make_state(
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3],
    )
    force = np.array([1.0, 2.0, 3.0])
    moment = np.array([0.1, 0.2, 0.3])
    x_before = x.copy()
    force_before = force.copy()
    moment_before = moment.copy()

    xdot = rigid_body_derivatives(x, force, moment, cfg)

    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(force, force_before)
    np.testing.assert_array_equal(moment, moment_before)
    assert xdot.shape == (13,)
    assert xdot.dtype == np.float64


def test_config_type_is_aircraft_config(cfg):
    assert isinstance(cfg, AircraftConfig)
