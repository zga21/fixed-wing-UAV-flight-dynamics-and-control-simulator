"""Actuator lag, rate, saturation, and reset tests."""

import numpy as np
import pytest

from uav_sim.hardware.actuator import Actuator, ActuatorBank


def test_small_step_matches_first_order_response():
    actuator = Actuator(tau=0.05, rate_limit=100.0, pos_limits=(-1.0, 1.0))
    dt = 0.0001
    command = 0.1
    for _ in range(int(0.15 / dt)):
        actual = actuator.step(command, dt)
    expected = command * (1.0 - np.exp(-0.15 / 0.05))
    assert actual == pytest.approx(expected, rel=0.01)


def test_large_step_has_exact_rate_limited_traversal_time():
    actuator = Actuator(tau=0.05, rate_limit=2.0, pos_limits=(-0.4, 0.4))
    dt = 0.001
    elapsed = 0.0
    while actuator.position < 0.4:
        actuator.step(10.0, dt)
        elapsed += dt
    assert elapsed == pytest.approx(0.4 / 2.0, abs=1e-12)
    assert actuator.is_saturated


def test_large_step_is_a_straight_rate_limited_ramp():
    actuator = Actuator(tau=0.05, rate_limit=1.0, pos_limits=(-1.0, 1.0))
    values = np.array([actuator.step(10.0, 0.01) for _ in range(20)])
    np.testing.assert_allclose(np.diff(values), np.full(19, 0.01), atol=1e-15)
    assert actuator.is_rate_limited


def test_reset_restores_position_and_clears_flags():
    actuator = Actuator(tau=0.05, rate_limit=1.0, pos_limits=(-0.2, 0.2))
    actuator.step(10.0, 1.0)
    actuator.reset(0.05)
    assert actuator.position == 0.05
    assert not actuator.is_saturated
    assert not actuator.is_rate_limited


def test_actuator_bank_steps_all_four_channels(cfg):
    bank = ActuatorBank.from_config(cfg)
    command = np.array([0.2, -0.2, 0.1, 0.8])
    actual = bank.step(command, cfg.integration.dt)
    assert actual.shape == (4,)
    assert np.all(np.abs(actual[0:3]) < np.abs(command[0:3]))
    assert actual[3] < command[3]


def test_ideal_actuator_bank_returns_commands_exactly(cfg):
    bank = ActuatorBank.from_config(cfg, ideal=True)
    command = np.array([0.1, -0.1, 0.02, 0.7])
    np.testing.assert_array_equal(bank.step(command, 0.002), command)
