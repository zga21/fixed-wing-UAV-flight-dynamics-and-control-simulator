"""Mission metric tests."""

import numpy as np

from uav_sim.metrics import SUCCESS_CRITERIA, Metrics, compute_metrics
from uav_sim.result import SimResult
from uav_sim.scenario import Scenario, ScenarioInitial, ScenarioSegment
from uav_sim.state import IDX_VEL, initial_state


def _simple_scenario() -> Scenario:
    return Scenario(
        name="synthetic",
        duration=2.0,
        initial=ScenarioInitial(V=25.0, altitude=100.0),
        segments=(
            ScenarioSegment(0.0, 100.0, 25.0, 0.0),
            ScenarioSegment(1.0, 100.0, 25.0, 0.0),
        ),
    )


def test_metrics_are_zero_for_perfect_tracking(cfg):
    t = np.linspace(0.0, 2.0, 11)
    x = np.asarray([initial_state(25.0, h=100.0) for _ in t])
    u = np.zeros((t.size, 4), dtype=np.float64)
    result = SimResult(t=t, x=x, u=u, u_actual=u)
    metrics = compute_metrics(result, _simple_scenario(), cfg)

    assert metrics.rmse_h == 0.0
    assert metrics.rmse_V == 0.0
    assert metrics.envelope_violations == 0
    assert metrics.success


def test_metrics_serialize_to_flat_dict(cfg):
    t = np.array([0.0])
    x = np.asarray([initial_state(25.0, h=100.0)])
    u = np.zeros((1, 4), dtype=np.float64)
    metrics = compute_metrics(
        SimResult(t=t, x=x, u=u, u_actual=u), _simple_scenario(), cfg
    )
    payload = metrics.to_dict()

    assert set(payload) == set(Metrics.__dataclass_fields__)
    assert all(not isinstance(value, dict) for value in payload.values())


def test_success_criteria_are_frozen_with_expected_keys():
    assert set(SUCCESS_CRITERIA) == {
        "rmse_h",
        "rmse_V",
        "max_phi_deg",
        "saturation_fraction",
        "envelope_violations",
    }


def test_envelope_violation_counts_bad_airspeed(cfg):
    t = np.array([0.0])
    x = initial_state(25.0, h=100.0)
    x[IDX_VEL] = np.array([100.0, 0.0, 0.0])
    u = np.zeros((1, 4), dtype=np.float64)
    metrics = compute_metrics(
        SimResult(t=t, x=np.asarray([x]), u=u, u_actual=u),
        _simple_scenario(),
        cfg,
    )

    assert metrics.envelope_violations > 0
