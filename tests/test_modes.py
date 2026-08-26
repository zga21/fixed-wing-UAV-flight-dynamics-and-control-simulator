"""Aircraft mode identification tests."""

import numpy as np
import pytest

from uav_sim.environment.atmosphere import isa
from uav_sim.linearise import decouple, linearise_euler
from uav_sim.modes import check_modes_plausible, identify_modes
from uav_sim.trim import trim


def _modes_at(V, cfg, plant):
    point = trim(V, 0.0, 100.0, cfg, plant)
    A, B = linearise_euler(point, plant, cfg)
    lon, lat = decouple(A, B)
    return identify_modes(lon.A, lat.A)


def test_all_five_modes_identified_at_cruise(cfg, plant):
    modes = _modes_at(25.0, cfg, plant)

    assert set(modes) == {
        "short_period",
        "phugoid",
        "dutch_roll",
        "roll_subsidence",
        "spiral",
    }
    assert check_modes_plausible(modes) == []


def test_mode_ranges_with_v1_exceptions_documented(cfg, plant):
    """Dutch roll is faster than textbook ranges for this small v1 airframe."""
    modes = _modes_at(25.0, cfg, plant)

    assert 3.0 <= modes["short_period"].omega_n <= 10.0
    assert 0.3 <= modes["short_period"].zeta <= 0.9
    assert 0.1 <= modes["phugoid"].omega_n <= 0.8
    assert 0.02 <= modes["phugoid"].zeta <= 0.6
    assert 1.0 <= modes["dutch_roll"].omega_n <= 10.0
    assert 0.05 <= modes["dutch_roll"].zeta <= 0.45
    assert modes["roll_subsidence"].eigenvalue.real < 0.0
    assert abs(modes["spiral"].eigenvalue.real) < 0.1


def test_short_period_frequency_increases_with_airspeed(cfg, plant):
    speeds = [18.0, 20.0, 25.0, 30.0, 35.0]
    short_period = [_modes_at(V, cfg, plant)["short_period"].omega_n for V in speeds]

    assert np.all(np.diff(short_period) > 0.0)


def test_phugoid_frequency_matches_classic_approximation(cfg, plant):
    for V in [18.0, 25.0, 35.0]:
        phugoid = _modes_at(V, cfg, plant)["phugoid"]
        expected = 9.80665 * np.sqrt(2.0) / V

        assert phugoid.omega_n == pytest.approx(expected, rel=0.30)


def test_roll_subsidence_matches_analytic_estimate(cfg, plant):
    for V in [18.0, 25.0, 35.0]:
        modes = _modes_at(V, cfg, plant)
        rho = isa(100.0).rho
        q_bar = 0.5 * rho * V**2
        numerator = q_bar * cfg.geometry.S * cfg.geometry.b**2 * cfg.aero.C_l_p
        expected = numerator / (2.0 * V * cfg.mass.I_xx)

        assert modes["roll_subsidence"].eigenvalue.real == pytest.approx(
            expected,
            rel=0.08,
        )
