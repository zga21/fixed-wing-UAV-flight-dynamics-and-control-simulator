"""Fair paired-comparison harness tests."""

from dataclasses import replace

import pandas as pd
import pytest

from uav_sim.compare import (
    assert_fair_comparison,
    compare_controllers,
    paired_differences,
)
from uav_sim.control.baseline import CascadedPIDAutopilot
from uav_sim.plant import AircraftPlant
from uav_sim.scenario import load_scenario


def _short_scenario():
    return replace(
        load_scenario("config/scenarios/disturbance_rejection.yaml"), duration=0.1
    )


def test_comparison_uses_identical_pairs_and_supports_differences(cfg):
    controllers = {
        "PID": CascadedPIDAutopilot(cfg),
        "PID copy": CascadedPIDAutopilot(cfg),
    }
    frame = compare_controllers(
        controllers,
        [_short_scenario()],
        [3, 7],
        lambda: AircraftPlant(cfg),
        cfg,
    )

    assert len(frame) == 4
    assert frame["git_commit"].nunique() == 1
    assert frame["config_hash"].nunique() == 1
    differences = paired_differences(frame, reference="PID")
    assert (differences["PID copy"] == 0.0).all()


def test_fairness_rejects_different_seed_sets():
    frame = pd.DataFrame(
        [
            {
                "controller": "A",
                "scenario": "s",
                "seed": 1,
                "git_commit": "x",
                "config_hash": "c",
            },
            {
                "controller": "B",
                "scenario": "s",
                "seed": 2,
                "git_commit": "x",
                "config_hash": "c",
            },
        ]
    )
    with pytest.raises(ValueError, match="different scenario/seed pairs"):
        assert_fair_comparison(frame)


def test_parallel_and_serial_comparisons_are_identical(cfg):
    controllers = {
        "PID": CascadedPIDAutopilot(cfg),
        "PID copy": CascadedPIDAutopilot(cfg),
    }
    arguments = (
        controllers,
        [_short_scenario()],
        [3, 7],
        lambda: AircraftPlant(cfg),
        cfg,
    )
    serial = compare_controllers(*arguments, workers=1)
    parallel = compare_controllers(*arguments, workers=2)
    pd.testing.assert_frame_equal(serial, parallel, check_exact=True)
