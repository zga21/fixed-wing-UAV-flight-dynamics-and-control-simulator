"""Numerical linearisation tests."""

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from uav_sim.linearise import (
    coupling_norms,
    decouple,
    euler_derivative_from_full,
    euler_state_from_full,
    full_state_from_euler,
    linearise,
    linearise_euler,
)
from uav_sim.trim import trim


def test_linearise_shapes_and_finite_values(cfg, plant):
    point = trim(25.0, 0.0, 100.0, cfg, plant)
    A, B = linearise(point, plant, cfg)
    Ae, Be = linearise_euler(point, plant, cfg)

    assert A.shape == (13, 13)
    assert B.shape == (13, 4)
    assert Ae.shape == (12, 12)
    assert Be.shape == (12, 4)
    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(B))
    assert np.all(np.isfinite(Ae))
    assert np.all(np.isfinite(Be))


def test_step_size_plateau_is_stable(cfg, plant):
    point = trim(25.0, 0.0, 100.0, cfg, plant)
    A_mid, B_mid = linearise_euler(point, plant, cfg, eps=1e-5)
    A_small, B_small = linearise_euler(point, plant, cfg, eps=1e-6)
    A_large, B_large = linearise_euler(point, plant, cfg, eps=1e-4)

    np.testing.assert_allclose(A_mid, A_small, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(B_mid, B_small, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(A_mid, A_large, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(B_mid, B_large, rtol=1e-4, atol=1e-5)


def test_symmetric_trim_decouples_exactly(cfg, plant):
    point = trim(25.0, 0.0, 100.0, cfg, plant)
    A, B = linearise_euler(point, plant, cfg)
    lon, lat = decouple(A, B)

    assert lon.A.shape == (5, 5)
    assert lat.A.shape == (4, 4)
    assert coupling_norms(A) == pytest.approx((0.0, 0.0), abs=1e-8)


def test_hand_derived_linearisation_entries(cfg, plant):
    """Kinematics give theta_dot_q=1, p_D_dot_theta=-V, wdot_q=u0 at trim."""
    point = trim(25.0, 0.0, 100.0, cfg, plant)
    A, _B = linearise_euler(point, plant, cfg)
    y0 = euler_state_from_full(point.x)

    assert A[7, 10] == pytest.approx(1.0, abs=1e-10)
    assert A[2, 7] == pytest.approx(-point.V, rel=1e-8)
    assert A[5, 10] == pytest.approx(y0[3], rel=1e-8)


def test_independent_forward_difference_matches_central_difference(cfg, plant):
    point = trim(25.0, 0.0, 100.0, cfg, plant)
    y0 = euler_state_from_full(point.x)

    def f_y(y):
        x = full_state_from_euler(y)
        return euler_derivative_from_full(x, plant.derivatives(0.0, x, point.u))

    A, _B = linearise_euler(point, plant, cfg, eps=1e-5)
    A_forward = approx_fprime(y0, f_y, epsilon=1e-7)
    if A_forward.shape != A.shape:
        A_forward = A_forward.T

    np.testing.assert_allclose(A, A_forward, rtol=1e-5, atol=1e-5)


def test_nearby_trim_linearisation_is_continuous(cfg, plant):
    low = trim(24.5, 0.0, 100.0, cfg, plant)
    high = trim(25.0, 0.0, 100.0, cfg, plant)
    A_low, B_low = linearise_euler(low, plant, cfg)
    A_high, B_high = linearise_euler(high, plant, cfg)

    assert np.linalg.norm(A_high - A_low) / np.linalg.norm(A_high) < 0.04
    assert np.linalg.norm(B_high - B_low) / np.linalg.norm(B_high) < 0.04
