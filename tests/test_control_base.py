"""Controller protocol contract tests."""

import numpy as np

from uav_sim.control.base import Controller, Reference, ZeroController
from uav_sim.control.baseline import CascadedPIDAutopilot
from uav_sim.integrate import simulate
from uav_sim.state import initial_state


def test_zero_controller_implements_protocol_and_runs_in_simulate(cfg, plant):
    controller = ZeroController()
    result = simulate(
        initial_state(25.0, alpha=0.05, h=100.0),
        controller,
        0.02,
        cfg,
        plant,
    )

    assert isinstance(controller, Controller)
    assert not result.terminated_early
    np.testing.assert_array_equal(controller.gains, np.zeros(0))


def test_baseline_gains_round_trip_and_metadata_lengths(cfg):
    controller = CascadedPIDAutopilot(cfg)
    original = controller.gains.copy()
    updated = original.copy()
    updated[0] += 0.01
    controller.gains = updated

    np.testing.assert_allclose(controller.gains, updated)
    assert len(controller.gain_names) == controller.gains.size
    assert len(controller.gain_bounds) == controller.gains.size


def test_baseline_reset_reproduces_identical_outputs(cfg):
    controller = CascadedPIDAutopilot(cfg)
    x = initial_state(25.0, alpha=0.05, h=100.0)
    ref = Reference(altitude=120.0, airspeed=25.0, heading=0.2)
    first = [
        controller.update(
            x, ref, idx * cfg.integration.dt, cfg.integration.dt
        ).to_array()
        for idx in range(50)
    ]
    controller.reset()
    second = [
        controller.update(
            x, ref, idx * cfg.integration.dt, cfg.integration.dt
        ).to_array()
        for idx in range(50)
    ]

    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
