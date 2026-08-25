"""Aerodynamic coefficient model tests."""

import numpy as np

from uav_sim.aero import AeroModel, LinearAeroModel
from uav_sim.air_data import AirData


def _air(V=25.0, alpha=0.0, beta=0.0):
    return AirData(
        V=V,
        alpha=alpha,
        beta=beta,
        q_bar=0.5 * 1.225 * V**2,
        v_air_b=np.array([V, 0.0, 0.0]),
        rho=1.225,
    )


def test_lift_increases_monotonically_with_alpha(cfg):
    model = LinearAeroModel(cfg.aero)
    alphas = np.deg2rad(np.linspace(-5.0, 10.0, 50))
    values = [
        model.coefficients(
            _air(alpha=alpha),
            np.zeros(3),
            np.zeros(4),
            cfg.geometry,
        ).C_L
        for alpha in alphas
    ]

    assert np.all(np.diff(values) > 0.0)


def test_pitch_moment_decreases_with_alpha(cfg):
    model = LinearAeroModel(cfg.aero)
    alphas = np.deg2rad(np.linspace(-5.0, 10.0, 50))
    values = [
        model.coefficients(
            _air(alpha=alpha),
            np.zeros(3),
            np.zeros(4),
            cfg.geometry,
        ).C_m
        for alpha in alphas
    ]

    assert np.all(np.diff(values) < 0.0)


def test_polar_drag_positive_and_minimized_near_minimum_lift(cfg):
    model = LinearAeroModel(cfg.aero)
    alphas = np.deg2rad(np.linspace(-10.0, 12.0, 100))
    coeffs = [
        model.coefficients(_air(alpha=alpha), np.zeros(3), np.zeros(4), cfg.geometry)
        for alpha in alphas
    ]
    drags = np.array([coeff.C_D for coeff in coeffs])
    lifts = np.array([abs(coeff.C_L) for coeff in coeffs])

    assert np.all(drags > 0.0)
    assert int(np.argmin(drags)) == int(np.argmin(lifts))


def test_lateral_symmetry_is_exact(cfg):
    model = LinearAeroModel(cfg.aero)
    coeffs = model.coefficients(
        _air(alpha=0.05, beta=0.0),
        omega_b=np.array([0.0, 0.1, 0.0]),
        controls=np.array([0.0, -0.05, 0.0, 0.5]),
        geom=cfg.geometry,
    )

    assert coeffs.C_Y == 0.0
    assert coeffs.C_l == 0.0
    assert coeffs.C_n == 0.0


def test_control_signs_match_conventions(cfg):
    model = LinearAeroModel(cfg.aero)
    elevator = model.coefficients(
        _air(),
        np.zeros(3),
        np.array([0.0, 0.1, 0.0, 0.5]),
        cfg.geometry,
    )
    aileron = model.coefficients(
        _air(),
        np.zeros(3),
        np.array([0.1, 0.0, 0.0, 0.5]),
        cfg.geometry,
    )
    rudder = model.coefficients(
        _air(),
        np.zeros(3),
        np.array([0.0, 0.0, 0.1, 0.5]),
        cfg.geometry,
    )

    assert elevator.C_m < cfg.aero.C_m_0
    assert aileron.C_l > 0.0
    assert rudder.C_n > 0.0


def test_rate_nondimensionalisation_halves_when_speed_doubles(cfg):
    model = LinearAeroModel(cfg.aero)
    omega = np.array([0.3, 0.2, -0.1])
    slow = model.coefficients(_air(V=20.0), omega, np.zeros(4), cfg.geometry)
    fast = model.coefficients(_air(V=40.0), omega, np.zeros(4), cfg.geometry)
    no_rate = model.coefficients(_air(V=20.0), np.zeros(3), np.zeros(4), cfg.geometry)

    assert fast.C_l - no_rate.C_l == 0.5 * (slow.C_l - no_rate.C_l)


def test_zero_speed_produces_finite_coefficients(cfg):
    model = LinearAeroModel(cfg.aero)
    coeffs = model.coefficients(_air(V=0.0), np.ones(3), np.zeros(4), cfg.geometry)

    assert np.all(np.isfinite(list(coeffs.__dict__.values())))


def test_linear_aero_model_satisfies_protocol(cfg):
    assert isinstance(LinearAeroModel(cfg.aero), AeroModel)
