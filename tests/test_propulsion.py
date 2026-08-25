"""Propulsion model tests."""

import numpy as np
import pytest

from uav_sim.air_data import AirData
from uav_sim.constants import G0
from uav_sim.propulsion import propeller_thrust, propulsion_forces_moments


def test_zero_throttle_produces_zero_thrust(cfg):
    for V in [0.0, 10.0, 40.0]:
        assert propeller_thrust(0.0, V, 1.225, cfg.propulsion) == 0.0


def test_thrust_increases_with_throttle(cfg):
    values = [
        propeller_thrust(delta_t, 10.0, 1.225, cfg.propulsion)
        for delta_t in np.linspace(0.0, 1.0, 20)
    ]

    assert np.all(np.diff(values) >= 0.0)


def test_thrust_decreases_with_airspeed(cfg):
    values = [
        propeller_thrust(1.0, V, 1.225, cfg.propulsion)
        for V in np.linspace(0.0, 40.0, 20)
    ]

    assert np.all(np.diff(values) <= 0.0)


def test_thrust_is_clamped_nonnegative_and_throttle_is_clipped(cfg):
    assert propeller_thrust(-1.0, 0.0, 1.225, cfg.propulsion) == 0.0
    assert propeller_thrust(2.0, 0.0, 1.225, cfg.propulsion) == propeller_thrust(
        1.0,
        0.0,
        1.225,
        cfg.propulsion,
    )
    assert propeller_thrust(0.1, 100.0, 1.225, cfg.propulsion) == 0.0


def test_static_full_throttle_matches_frozen_source_model(cfg):
    thrust = propeller_thrust(1.0, 0.0, 1.225, cfg.propulsion)
    expected = (
        0.5
        * 1.225
        * cfg.propulsion.S_prop
        * cfg.propulsion.C_prop
        * cfg.propulsion.k_motor**2
    )
    thrust_to_weight = thrust / (cfg.mass.m * G0)

    assert thrust == pytest.approx(expected)
    assert thrust_to_weight > 1.5


def test_thrust_scales_with_density(cfg):
    full = propeller_thrust(1.0, 10.0, 1.0, cfg.propulsion)
    half = propeller_thrust(1.0, 10.0, 0.5, cfg.propulsion)

    assert half == pytest.approx(0.5 * full)


def test_propulsion_force_points_forward(cfg):
    air = AirData(
        V=10.0,
        alpha=0.0,
        beta=0.0,
        q_bar=61.25,
        v_air_b=np.array([10.0, 0.0, 0.0]),
        rho=1.225,
    )
    F_b, M_b = propulsion_forces_moments(1.0, air, cfg.propulsion)

    assert F_b[0] > 0.0
    np.testing.assert_array_equal(F_b[1:], [0.0, 0.0])
    np.testing.assert_array_equal(M_b, np.zeros(3))
