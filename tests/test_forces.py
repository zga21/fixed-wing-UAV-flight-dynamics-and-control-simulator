"""Aerodynamic force and moment assembly tests."""

import numpy as np
import pytest

from uav_sim.aero import AeroCoeffs, LinearAeroModel
from uav_sim.air_data import AirData
from uav_sim.forces import aero_forces_moments, total_forces_moments
from uav_sim.state import make_state


def _air(V=25.0, alpha=0.0, q_bar=None):
    q = 0.5 * 1.225 * V**2 if q_bar is None else q_bar
    return AirData(
        V=V,
        alpha=alpha,
        beta=0.0,
        q_bar=q,
        v_air_b=np.array([V, 0.0, 0.0]),
        rho=1.225,
    )


def test_zero_alpha_force_components(cfg):
    coeffs = AeroCoeffs(C_L=0.4, C_D=0.03, C_Y=0.0, C_l=0.0, C_m=0.0, C_n=0.0)
    F_b, _M_b = aero_forces_moments(_air(alpha=0.0), coeffs, cfg.geometry)
    q_s = _air().q_bar * cfg.geometry.S

    assert F_b[0] == pytest.approx(-q_s * coeffs.C_D)
    assert F_b[2] == pytest.approx(-q_s * coeffs.C_L)


def test_ninety_degree_alpha_swaps_lift_and_drag_axes(cfg):
    coeffs = AeroCoeffs(C_L=0.4, C_D=0.03, C_Y=0.0, C_l=0.0, C_m=0.0, C_n=0.0)
    F_b, _M_b = aero_forces_moments(_air(alpha=np.pi / 2.0), coeffs, cfg.geometry)
    q_s = _air().q_bar * cfg.geometry.S

    assert F_b[0] == pytest.approx(q_s * coeffs.C_L)
    assert F_b[2] == pytest.approx(-q_s * coeffs.C_D)


def test_lift_acts_upward_in_down_positive_body_frame(cfg):
    coeffs = AeroCoeffs(C_L=0.4, C_D=0.03, C_Y=0.0, C_l=0.0, C_m=0.0, C_n=0.0)
    F_b, _M_b = aero_forces_moments(_air(alpha=0.0), coeffs, cfg.geometry)

    assert F_b[2] < 0.0


def test_lift_magnitude_order_of_magnitude(cfg):
    """At 25 m/s, q*S*C_L ~= 0.5*1.225*25^2*0.55*0.4 = 84 N."""
    coeffs = AeroCoeffs(C_L=0.4, C_D=0.03, C_Y=0.0, C_l=0.0, C_m=0.0, C_n=0.0)
    F_b, _M_b = aero_forces_moments(_air(V=25.0), coeffs, cfg.geometry)

    assert abs(F_b[2]) == pytest.approx(84.2, rel=0.02)


def test_forces_scale_with_v_squared(cfg):
    coeffs = AeroCoeffs(C_L=0.4, C_D=0.03, C_Y=0.0, C_l=0.0, C_m=0.0, C_n=0.0)
    F_slow, _ = aero_forces_moments(_air(V=20.0), coeffs, cfg.geometry)
    F_fast, _ = aero_forces_moments(_air(V=40.0), coeffs, cfg.geometry)

    np.testing.assert_allclose(F_fast, 4.0 * F_slow)


def test_symmetric_flight_has_zero_lateral_force_and_moments(cfg):
    coeffs = AeroCoeffs(C_L=0.4, C_D=0.03, C_Y=0.0, C_l=0.0, C_m=0.2, C_n=0.0)
    F_b, M_b = aero_forces_moments(_air(), coeffs, cfg.geometry)

    assert F_b[1] == 0.0
    assert M_b[0] == 0.0
    assert M_b[2] == 0.0


def test_zero_speed_forces_are_zero(cfg):
    coeffs = AeroCoeffs(C_L=0.4, C_D=0.03, C_Y=0.1, C_l=0.1, C_m=0.2, C_n=0.3)
    F_b, M_b = aero_forces_moments(_air(V=0.0), coeffs, cfg.geometry)

    np.testing.assert_array_equal(F_b, np.zeros(3))
    np.testing.assert_array_equal(M_b, np.zeros(3))


def test_total_forces_moments_returns_air_data(cfg):
    x = make_state(
        [0.0, 0.0, -100.0],
        [25.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    )
    F_b, M_b, air = total_forces_moments(
        x,
        np.array([0.0, 0.0, 0.0, 0.5]),
        cfg,
        LinearAeroModel(cfg.aero),
    )

    assert F_b.shape == (3,)
    assert M_b.shape == (3,)
    assert air.V == pytest.approx(25.0)
