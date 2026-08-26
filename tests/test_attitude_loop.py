"""Attitude-loop tests."""

import numpy as np
import pytest

from uav_sim.control.attitude_loop import AttitudeLoop, coordinated_turn_rate


def test_coordinated_turn_rate_formula():
    rate = coordinated_turn_rate(np.deg2rad(30.0), 25.0)
    expected = 9.80665 * np.tan(np.deg2rad(30.0)) / 25.0

    assert rate == pytest.approx(expected)


def test_attitude_loop_clamps_large_roll_command(cfg):
    loop = AttitudeLoop(cfg=cfg)
    omega_cmd = loop.update(
        np.array([np.deg2rad(80.0), 0.0]),
        np.zeros(3),
        25.0,
        0.02,
    )

    assert omega_cmd[0] <= np.deg2rad(120.0)
    assert omega_cmd[2] <= coordinated_turn_rate(cfg.envelope.phi_max_rad, 25.0)


def test_attitude_loop_gain_vector_round_trips(cfg):
    loop = AttitudeLoop(cfg=cfg)
    gains = loop.gains.copy()
    gains[0] += 0.1
    loop.gains = gains

    np.testing.assert_allclose(loop.gains, gains)
    assert len(loop.gain_names) == gains.size
    assert len(loop.gain_bounds) == gains.size
