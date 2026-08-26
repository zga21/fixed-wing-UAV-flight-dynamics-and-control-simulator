"""Outer-loop and baseline autopilot tests."""

import numpy as np
import pytest

from uav_sim.control.baseline import CascadedPIDAutopilot
from uav_sim.control.outer_loop import HeadingLoop, TECSLite, wrap_angle
from uav_sim.scenario import load_scenario, run_scenario


def test_tecs_commands_throttle_up_for_energy_deficit(cfg):
    tecs = TECSLite(cfg)
    theta_cmd, throttle = tecs.update(150.0, 25.0, 100.0, 25.0, 0.1, 0.3, 0.1)

    assert theta_cmd > 0.1
    assert throttle > 0.3


def test_heading_loop_wraps_angle_and_commands_bounded_roll(cfg):
    loop = HeadingLoop(cfg)
    phi = loop.update(np.deg2rad(-179.0), np.deg2rad(179.0), 25.0, 0.1)

    assert wrap_angle(np.deg2rad(-179.0) - np.deg2rad(179.0)) > 0.0
    assert 0.0 < phi <= cfg.envelope.phi_max_rad


@pytest.mark.slow
def test_baseline_runs_all_loops_at_specified_rates(cfg, plant):
    controller = CascadedPIDAutopilot(cfg)
    scenario = load_scenario("config/scenarios/disturbance_rejection.yaml")
    run_scenario(scenario, controller, plant, cfg, seed=0)

    assert controller.loop_counts == {"outer": 600, "attitude": 3000, "rate": 12000}


@pytest.mark.slow
def test_baseline_default_scenarios_succeed(cfg, plant):
    controller = CascadedPIDAutopilot(cfg)
    for path in [
        "config/scenarios/mission_a.yaml",
        "config/scenarios/step_responses.yaml",
        "config/scenarios/disturbance_rejection.yaml",
    ]:
        scenario = load_scenario(path)
        _result, metrics = run_scenario(scenario, controller, plant, cfg, seed=0)

        assert metrics.success
        assert metrics.envelope_violations == 0
