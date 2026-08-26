"""Phase 4 baseline autopilot gate."""

import json
from pathlib import Path

import numpy as np
import pytest
from scripts.record_baseline import build_record

from uav_sim.control.base import Reference
from uav_sim.control.baseline import CascadedPIDAutopilot
from uav_sim.scenario import (
    Scenario,
    ScenarioInitial,
    ScenarioSegment,
    load_scenario,
    run_scenario,
)


@pytest.fixture
def baseline(cfg):
    return CascadedPIDAutopilot(cfg)


@pytest.mark.gate
@pytest.mark.slow
def test_step_responses_meet_spec(baseline, plant, cfg):
    """Baseline step scenario succeeds without envelope violations."""
    scenario = load_scenario("config/scenarios/step_responses.yaml")
    _result, metrics = run_scenario(scenario, baseline, plant, cfg, seed=0)

    assert metrics.success
    assert metrics.envelope_violations == 0
    assert metrics.rmse_h < 12.0
    assert metrics.rmse_V < 3.0


@pytest.mark.gate
@pytest.mark.slow
def test_full_mission_succeeds(baseline, plant, cfg):
    """mission_a completes with success=True and no envelope violations."""
    scenario = load_scenario("config/scenarios/mission_a.yaml")
    _result, metrics = run_scenario(scenario, baseline, plant, cfg, seed=0)

    assert metrics.success
    assert metrics.envelope_violations == 0


@pytest.mark.gate
def test_loop_frequency_separation(baseline):
    """Outer loops are separated from inner loops by the 3-5x rule."""
    assert baseline.rates.rate_hz / baseline.rates.attitude_hz == pytest.approx(4.0)
    assert baseline.rates.attitude_hz / baseline.rates.outer_hz == pytest.approx(5.0)


@pytest.mark.gate
@pytest.mark.slow
def test_envelope_coverage(baseline, plant, cfg):
    """Same gains run at 18, 25, and 35 m/s; degradation is recorded."""
    outcomes = {}
    for speed in [18.0, 25.0, 35.0]:
        scenario = Scenario(
            name=f"hold_{speed:g}",
            duration=20.0,
            initial=ScenarioInitial(V=speed, altitude=100.0),
            segments=(ScenarioSegment(0.0, 100.0, speed, 0.0),),
        )
        _result, metrics = run_scenario(scenario, baseline, plant, cfg, seed=0)
        outcomes[speed] = metrics

    assert not outcomes[18.0].terminated_early
    assert outcomes[25.0].success
    assert outcomes[35.0].success


@pytest.mark.gate
@pytest.mark.slow
def test_baseline_record_is_current():
    """Re-running reproduces docs/baseline_record.json within 1%."""
    record_path = Path("docs/baseline_record.json")
    assert record_path.exists()
    recorded = json.loads(record_path.read_text(encoding="utf-8"))
    current = build_record()

    for scenario_name, recorded_metrics in recorded["scenarios"].items():
        current_metrics = current["scenarios"][scenario_name]
        for key, value in recorded_metrics.items():
            if isinstance(value, bool):
                assert current_metrics[key] is value
            elif isinstance(value, int):
                assert current_metrics[key] == value
            else:
                assert current_metrics[key] == pytest.approx(value, rel=0.01, abs=1e-9)


def test_baseline_direct_reference_update_returns_finite_controls(baseline, cfg):
    x = np.zeros(13, dtype=np.float64)
    from uav_sim.state import initial_state

    x = initial_state(25.0, alpha=0.05, h=100.0)
    controls = baseline.update(
        x,
        Reference(altitude=120.0, airspeed=25.0, heading=0.0),
        0.0,
        cfg.integration.dt,
    )

    assert np.all(np.isfinite(controls.to_array()))
