"""Phase 3 trim, linearisation, and modes gate."""

import numpy as np
import pytest

from uav_sim.integrate import rk4_step, simulate
from uav_sim.linearise import decouple, euler_state_from_full, linearise_euler
from uav_sim.modes import check_modes_plausible, identify_modes
from uav_sim.state import IDX_VEL
from uav_sim.trim import TrimNotConverged, trim, trim_grid


def _constant_controller(controls):
    def controller(_t, _x, _cfg, _rng):
        return controls

    return controller


def _doublet_controls(trim_u, amplitude, t):
    controls = trim_u.copy()
    if t < 1.0:
        controls[1] += amplitude
    elif t < 2.0:
        controls[1] -= amplitude
    return controls


def _linear_doublet(A, B, trim_u, amplitude, cfg, duration):
    dt = cfg.integration.dt
    n_steps = int(round(duration / dt))
    z = np.zeros(12, dtype=np.float64)
    history = [z.copy()]

    def derivative(t, state):
        du = _doublet_controls(trim_u, amplitude, t) - trim_u
        return A @ state + B @ du

    t = 0.0
    for _step in range(n_steps):
        z = rk4_step(lambda t_stage, z_stage: derivative(t_stage, z_stage), t, z, dt)
        t += dt
        history.append(z.copy())
    return np.asarray(history)


def _nonlinear_doublet_delta(point, amplitude, cfg, plant, duration):
    def controller(t, _x, _cfg, _rng):
        return _doublet_controls(point.u, amplitude, t)

    disturbed = simulate(point.x, controller, duration, cfg, plant, log_every=1)
    baseline = simulate(
        point.x,
        _constant_controller(point.u),
        duration,
        cfg,
        plant,
        log_every=1,
    )
    assert not disturbed.terminated_early
    assert not baseline.terminated_early
    return np.asarray(
        [
            euler_state_from_full(row) - euler_state_from_full(base)
            for row, base in zip(disturbed.x, baseline.x, strict=True)
        ],
        dtype=np.float64,
    )


@pytest.mark.gate
def test_trim_converges_across_envelope(cfg, plant):
    """9 reachable grid points, residual < 1e-10 at every one."""
    points = trim_grid(
        [18.0, 25.0, 35.0],
        np.deg2rad([-2.0, 0.0, 5.0]),
        100.0,
        cfg,
        plant,
    )

    assert max(point.residual for point in points) < 1e-10


@pytest.mark.gate
def test_v1_rejects_unreachable_low_speed_steep_descent(cfg, plant):
    """The frozen thrust-only v1 model cannot trim 18 m/s at -5 deg."""
    with pytest.raises(TrimNotConverged):
        trim(18.0, np.deg2rad(-5.0), 100.0, cfg, plant)


@pytest.mark.gate
def test_trim_is_physically_sensible(cfg, plant):
    """alpha rises as V falls; delta_t rises with climb angle."""
    level = [trim(V, 0.0, 100.0, cfg, plant) for V in [18.0, 25.0, 35.0]]
    climbs = [
        trim(25.0, np.deg2rad(gamma_deg), 100.0, cfg, plant)
        for gamma_deg in [-2.0, 0.0, 5.0]
    ]

    assert level[0].alpha > level[1].alpha > level[2].alpha
    assert climbs[0].u[3] < climbs[1].u[3] < climbs[2].u[3]


@pytest.mark.gate
@pytest.mark.slow
def test_trim_actually_holds(cfg, plant):
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


@pytest.mark.gate
def test_all_five_modes_present_and_plausible(cfg, plant):
    """All five aircraft modes are present in the v1 cruise linearisation."""
    point = trim(25.0, 0.0, 100.0, cfg, plant)
    A, B = linearise_euler(point, plant, cfg)
    lon, lat = decouple(A, B)
    modes = identify_modes(lon.A, lat.A)

    assert set(modes) == {
        "short_period",
        "phugoid",
        "dutch_roll",
        "roll_subsidence",
        "spiral",
    }
    assert check_modes_plausible(modes) == []


@pytest.mark.gate
@pytest.mark.slow
def test_linear_matches_nonlinear_small_signal(cfg, plant):
    """0.5 deg elevator doublet, 5 s, linear and nonlinear trajectories agree."""
    point = trim(25.0, 0.0, 100.0, cfg, plant)
    A, B = linearise_euler(point, plant, cfg)
    linear = _linear_doublet(A, B, point.u, np.deg2rad(0.5), cfg, 5.0)
    nonlinear = _nonlinear_doublet_delta(point, np.deg2rad(0.5), cfg, plant, 5.0)
    idx = np.array([2, 3, 5, 7, 10])

    error = np.linalg.norm(nonlinear[:, idx] - linear[:, idx])
    magnitude = np.linalg.norm(nonlinear[:, idx])
    assert error / magnitude < 0.02


@pytest.mark.gate
@pytest.mark.slow
def test_linear_diverges_from_nonlinear_large_signal(cfg, plant):
    """15 deg doublet must diverge, proving the linear model is local."""
    point = trim(25.0, 0.0, 100.0, cfg, plant)
    A, B = linearise_euler(point, plant, cfg)
    linear = _linear_doublet(A, B, point.u, np.deg2rad(15.0), cfg, 5.0)
    nonlinear = _nonlinear_doublet_delta(point, np.deg2rad(15.0), cfg, plant, 5.0)
    idx = np.array([2, 3, 5, 7, 10])

    error = np.linalg.norm(nonlinear[:, idx] - linear[:, idx])
    magnitude = np.linalg.norm(nonlinear[:, idx])
    assert error / magnitude > 0.10
