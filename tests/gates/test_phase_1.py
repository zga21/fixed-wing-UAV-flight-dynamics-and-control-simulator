"""Phase 1 physics gate tests."""

import numpy as np
import pytest

from uav_sim.constants import G0
from uav_sim.dynamics import gravity_body, rigid_body_derivatives
from uav_sim.integrate import simulate
from uav_sim.rotations import euler_to_quat, quat_to_dcm
from uav_sim.state import IDX_OMEGA, IDX_QUAT, IDX_VEL, make_state


def _gravity_only(_t, x, _u, cfg, _rng):
    return rigid_body_derivatives(
        x,
        gravity_body(x[IDX_QUAT], cfg.mass.m),
        np.zeros(3),
        cfg,
    )


def _force_free(_t, x, _u, cfg, _rng):
    return rigid_body_derivatives(x, np.zeros(3), np.zeros(3), cfg)


@pytest.mark.gate
def test_free_fall_matches_analytic(cfg):
    """Catches gravity sign errors, integration bugs, and p_D sign confusion."""
    x0 = make_state(
        pos_n=[0.0, 0.0, -1000.0],
        vel_b=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        omega_b=[0.0, 0.0, 0.0],
    )
    result = simulate(x0, None, 10.0, cfg, _gravity_only, log_every=1)

    fallen = result.x[-1, 2] - result.x[0, 2]
    assert fallen == pytest.approx(0.5 * G0 * 10.0**2, abs=1e-9)


@pytest.mark.gate
def test_ballistic_trajectory_matches_analytic_range_and_apex(cfg):
    """Catches DCM transposition and body/NED frame confusion."""
    duration = 10.0
    launch_angle = np.deg2rad(45.0)
    v0 = G0 * duration / (2.0 * np.sin(launch_angle))
    x0 = make_state(
        pos_n=[0.0, 0.0, -1000.0],
        vel_b=[v0 * np.cos(launch_angle), 0.0, -v0 * np.sin(launch_angle)],
        quat=[1.0, 0.0, 0.0, 0.0],
        omega_b=[0.0, 0.0, 0.0],
    )

    result = simulate(x0, None, duration, cfg, _gravity_only, log_every=1)

    range_n = result.x[-1, 0] - result.x[0, 0]
    height_gain = result.x[0, 2] - result.x[:, 2].min()
    expected_range = v0**2 * np.sin(2.0 * launch_angle) / G0
    expected_apex = v0**2 * np.sin(launch_angle) ** 2 / (2.0 * G0)
    assert range_n == pytest.approx(expected_range, rel=1e-6)
    assert height_gain == pytest.approx(expected_apex, rel=1e-6)


@pytest.mark.gate
def test_quaternion_norm_constant_rate(cfg):
    """Catches missing renormalisation and a wrong Omega matrix."""
    x0 = make_state(
        pos_n=[0.0, 0.0, -1000.0],
        vel_b=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        omega_b=[0.1, 0.2, 0.3],
    )
    result = simulate(x0, None, 100.0, cfg, _force_free, log_every=10)

    norms = np.linalg.norm(result.x[:, IDX_QUAT], axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-10)


@pytest.mark.gate
def test_angular_momentum_conserved_torque_free(cfg):
    """Catches a missing omega x (I omega) gyroscopic term."""
    x0 = make_state(
        pos_n=[0.0, 0.0, -1000.0],
        vel_b=[0.0, 0.0, 0.0],
        quat=euler_to_quat(0.2, -0.1, 0.3),
        omega_b=[0.1, 2.0, 0.05],
    )
    result = simulate(x0, None, 30.0, cfg, _force_free, log_every=10)

    h_inertial = np.array(
        [
            quat_to_dcm(row[IDX_QUAT]) @ (cfg.mass.inertia_tensor @ row[IDX_OMEGA])
            for row in result.x
        ]
    )
    drift = np.abs(h_inertial - h_inertial[0]).max()
    assert drift < 1e-8
    assert np.ptp(result.x[:, IDX_OMEGA][:, 0]) > 0.05


@pytest.mark.gate
def test_torque_free_energy_conserved(cfg):
    """Catches sign errors that inject or dissipate rigid-body energy."""
    x0 = make_state(
        pos_n=[0.0, 0.0, -1000.0],
        vel_b=[20.0, -3.0, 4.0],
        quat=euler_to_quat(0.1, 0.2, -0.3),
        omega_b=[0.7, 1.1, -0.4],
    )
    result = simulate(x0, None, 60.0, cfg, _force_free, log_every=10)

    translational = 0.5 * cfg.mass.m * np.sum(result.x[:, IDX_VEL] ** 2, axis=1)
    rotational = np.array(
        [
            0.5 * row[IDX_OMEGA] @ cfg.mass.inertia_tensor @ row[IDX_OMEGA]
            for row in result.x
        ]
    )
    energy = translational + rotational
    relative_drift = np.abs(energy - energy[0]).max() / energy[0]
    assert relative_drift < 1e-8
