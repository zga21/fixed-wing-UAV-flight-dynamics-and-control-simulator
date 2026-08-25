"""Air-data tests."""

import numpy as np
import pytest

from uav_sim.air_data import air_relative_velocity, compute_air_data


def test_straight_level_air_data():
    air = compute_air_data([25.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 100.0)

    assert air.V == pytest.approx(25.0)
    assert air.alpha == pytest.approx(0.0)
    assert air.beta == pytest.approx(0.0)


def test_positive_body_w_gives_positive_alpha():
    air = compute_air_data([25.0, 0.0, 2.0], [1.0, 0.0, 0.0, 0.0], 100.0)

    assert air.alpha > 0.0


def test_positive_body_v_gives_positive_beta():
    air = compute_air_data([25.0, 2.0, 0.0], [1.0, 0.0, 0.0, 0.0], 100.0)

    assert air.beta > 0.0


def test_zero_airspeed_has_zero_angles_and_pressure():
    air = compute_air_data([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 100.0)

    assert air.V == 0.0
    assert air.alpha == 0.0
    assert air.beta == 0.0
    assert air.q_bar == 0.0


def test_headwind_increases_air_relative_speed():
    air = compute_air_data(
        [25.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        100.0,
        wind_n=[-10.0, 0.0, 0.0],
    )

    assert air.V == pytest.approx(35.0)


def test_air_relative_velocity_subtracts_wind_in_body_axes():
    v_air_b = air_relative_velocity(
        vel_b=np.array([25.0, 0.0, 0.0]),
        quat=np.array([1.0, 0.0, 0.0, 0.0]),
        wind_n=np.array([5.0, 0.0, 0.0]),
    )

    np.testing.assert_allclose(v_air_b, [20.0, 0.0, 0.0])


def test_air_data_round_trip_from_angles():
    V = 31.0
    alpha = 0.12
    beta = -0.08
    vel_b = np.array(
        [
            V * np.cos(beta) * np.cos(alpha),
            V * np.sin(beta),
            V * np.cos(beta) * np.sin(alpha),
        ]
    )

    air = compute_air_data(vel_b, [1.0, 0.0, 0.0, 0.0], 100.0)

    assert air.V == pytest.approx(V)
    assert air.alpha == pytest.approx(alpha, abs=1e-12)
    assert air.beta == pytest.approx(beta, abs=1e-12)
