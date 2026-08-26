"""Trim solver tests."""

import numpy as np
import pytest

from uav_sim.air_data import compute_air_data
from uav_sim.integrate import simulate
from uav_sim.rotations import quat_to_euler
from uav_sim.state import IDX_OMEGA, IDX_QUAT, IDX_VEL
from uav_sim.trim import TrimNotConverged, trim, trim_grid


def _constant_controller(controls):
    def controller(_t, _x, _cfg, _rng):
        return controls

    return controller


def test_trim_grid_reachable_points_converge(cfg, plant):
    points = trim_grid(
        np.array([18.0, 25.0, 35.0]),
        np.deg2rad(np.array([-2.0, 0.0, 5.0])),
        100.0,
        cfg,
        plant,
    )

    assert len(points) == 9
    assert max(point.residual for point in points) < 1e-10


def test_trim_is_symmetric_and_zero_sideslip(cfg, plant):
    point = trim(25.0, 0.0, 100.0, cfg, plant)
    phi, _theta, _psi = quat_to_euler(point.x[IDX_QUAT])
    air = compute_air_data(point.x[IDX_VEL], point.x[IDX_QUAT], point.altitude)

    assert point.x[IDX_OMEGA] == pytest.approx(np.zeros(3), abs=0.0)
    assert phi == pytest.approx(0.0, abs=1e-14)
    assert air.beta == pytest.approx(0.0, abs=1e-14)


def test_trim_physical_trends(cfg, plant):
    level = [trim(V, 0.0, 100.0, cfg, plant) for V in [18.0, 25.0, 35.0]]
    climbs = [
        trim(25.0, np.deg2rad(gamma_deg), 100.0, cfg, plant)
        for gamma_deg in [-2.0, 0.0, 5.0]
    ]

    assert level[0].alpha > level[1].alpha > level[2].alpha
    assert climbs[0].u[3] < climbs[1].u[3] < climbs[2].u[3]


@pytest.mark.slow
def test_trim_holds_level_flight_for_60_seconds(cfg, plant):
    point = trim(25.0, 0.0, 100.0, cfg, plant)
    result = simulate(
        point.x,
        _constant_controller(point.u),
        60.0,
        cfg,
        plant,
        log_every=200,
    )
    altitude = -result.x[:, 2]
    airspeed = np.linalg.norm(result.x[:, IDX_VEL], axis=1)

    assert not result.terminated_early
    assert abs(altitude[-1] - altitude[0]) < 1.0
    assert abs(airspeed[-1] - airspeed[0]) < 0.1


def test_trim_rejects_outside_envelope(cfg, plant):
    with pytest.raises(TrimNotConverged, match="outside"):
        trim(5.0, 0.0, 100.0, cfg, plant)


def test_low_speed_steep_descent_is_rejected_by_v1_physics(cfg, plant):
    """Frozen v1 has thrust-only propulsion and no windmilling drag device."""
    with pytest.raises(TrimNotConverged):
        trim(18.0, np.deg2rad(-5.0), 100.0, cfg, plant)
