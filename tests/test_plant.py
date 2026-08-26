"""Aircraft plant composition tests."""

import numpy as np
import pytest

from uav_sim.integrate import simulate
from uav_sim.plant import AircraftPlant
from uav_sim.rotations import quat_to_euler
from uav_sim.state import IDX_OMEGA, IDX_QUAT, IDX_VEL, initial_state


def _near_trim_state():
    return initial_state(25.0, alpha=0.091, h=100.0)


def _near_trim_controls():
    return np.array([0.0, -0.116, 0.0, 0.329], dtype=np.float64)


def _constant_controller(controls):
    def controller(_t, _x, _cfg, _rng):
        return controls

    return controller


@pytest.mark.slow
def test_level_flight_near_trim_persists(cfg):
    plant = AircraftPlant(cfg)
    result = simulate(
        _near_trim_state(),
        _constant_controller(_near_trim_controls()),
        30.0,
        cfg,
        plant,
        log_every=100,
    )
    altitude = -result.x[:, 2]

    assert not result.terminated_early
    assert abs(altitude[-1] - altitude[0]) < 10.0


@pytest.mark.slow
def test_pull_up_exchanges_speed_for_altitude(cfg):
    plant = AircraftPlant(cfg)
    controls = _near_trim_controls()
    controls[1] -= 0.08
    result = simulate(
        _near_trim_state(),
        _constant_controller(controls),
        8.0,
        cfg,
        plant,
        log_every=20,
    )
    airspeeds = np.linalg.norm(result.x[:, IDX_VEL], axis=1)
    altitude = -result.x[:, 2]

    assert altitude[-1] > altitude[0] + 20.0
    assert airspeeds[-1] < airspeeds[0]


@pytest.mark.slow
def test_aileron_step_produces_positive_roll_rate(cfg):
    plant = AircraftPlant(cfg)
    controls = _near_trim_controls()
    controls[0] = 0.05
    result = simulate(
        _near_trim_state(),
        _constant_controller(controls),
        5.0,
        cfg,
        plant,
        log_every=10,
    )

    assert result.x[:, IDX_OMEGA][:, 0].max() > 0.1


@pytest.mark.slow
def test_symmetric_flight_stays_symmetric(cfg):
    plant = AircraftPlant(cfg)
    result = simulate(
        _near_trim_state(),
        _constant_controller(_near_trim_controls()),
        60.0,
        cfg,
        plant,
        log_every=100,
    )
    euler = np.array([quat_to_euler(row[IDX_QUAT]) for row in result.x])

    np.testing.assert_allclose(result.x[:, IDX_VEL][:, 1], 0.0, atol=0.0)
    np.testing.assert_allclose(result.x[:, IDX_OMEGA][:, [0, 2]], 0.0, atol=0.0)
    np.testing.assert_allclose(euler[:, [0, 2]], 0.0, atol=0.0)


@pytest.mark.slow
def test_no_nan_over_120_second_near_trim_simulation(cfg):
    plant = AircraftPlant(cfg)
    result = simulate(
        _near_trim_state(),
        _constant_controller(_near_trim_controls()),
        120.0,
        cfg,
        plant,
        log_every=500,
    )

    assert not result.terminated_early
    assert np.all(np.isfinite(result.x))


def test_plant_diagnostics_include_force_breakdown(cfg):
    plant = AircraftPlant(cfg)
    diagnostics = plant.diagnostics(0.0, _near_trim_state(), _near_trim_controls())

    assert diagnostics["V"] == pytest.approx(25.0)
    assert diagnostics["F_total_b"].shape == (3,)
    assert diagnostics["M_total_b"].shape == (3,)
    assert diagnostics["F_aero_b"].shape == (3,)
    assert diagnostics["F_prop_b"].shape == (3,)
