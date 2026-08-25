"""Quaternion and rotation utility tests."""

import numpy as np
import pytest

from uav_sim.integrate import rk4_step
from uav_sim.rotations import (
    euler_to_quat,
    quat_derivative,
    quat_multiply,
    quat_normalise,
    quat_to_dcm,
    quat_to_euler,
)


def test_identity_quaternion_maps_to_identity_matrix():
    np.testing.assert_array_equal(
        quat_to_dcm(np.array([1.0, 0.0, 0.0, 0.0])),
        np.eye(3),
    )


def test_random_rotation_matrices_are_orthonormal_with_positive_determinant():
    rng = np.random.default_rng(1)
    for _ in range(1000):
        quat = quat_normalise(rng.normal(size=4))
        rotation = quat_to_dcm(quat)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)


def test_euler_round_trip_away_from_singularity():
    rng = np.random.default_rng(2)
    for _ in range(1000):
        phi = rng.uniform(-np.pi, np.pi)
        theta = rng.uniform(np.deg2rad(-89.0), np.deg2rad(89.0))
        psi = rng.uniform(-np.pi, np.pi)
        recovered = quat_to_euler(euler_to_quat(phi, theta, psi))
        np.testing.assert_allclose(recovered, [phi, theta, psi], atol=1e-12)


def test_known_yaw_maps_body_x_to_ned_y():
    quat = euler_to_quat(0.0, 0.0, np.deg2rad(90.0))
    rotated = quat_to_dcm(quat) @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-12)


def test_constant_yaw_rate_returns_to_equivalent_quaternion():
    omega = np.array([0.0, 0.0, 0.1])
    dt = 0.01
    duration = 2.0 * np.pi / omega[2]
    n_steps = int(round(duration / dt))
    quat = np.array([1.0, 0.0, 0.0, 0.0])

    def deriv(_t, q):
        return quat_derivative(q, omega)

    for step in range(n_steps):
        quat = quat_normalise(rk4_step(deriv, step * dt, quat, dt))

    identity = np.array([1.0, 0.0, 0.0, 0.0])
    error = min(np.linalg.norm(quat - identity), np.linalg.norm(quat + identity))
    assert error < 5e-4


def test_quat_normalise_rejects_near_zero_quaternion():
    with pytest.raises(ValueError, match="near-zero"):
        quat_normalise(np.zeros(4))


def test_hamilton_product_identity():
    quat = euler_to_quat(0.2, -0.1, 0.3)
    np.testing.assert_allclose(
        quat_multiply(np.array([1.0, 0.0, 0.0, 0.0]), quat),
        quat,
        atol=1e-12,
    )
