"""Rate-loop tests."""

import numpy as np

from uav_sim.control.rate_loop import (
    DEFAULT_RATE_GAINS,
    ActuatorLimits,
    RateGains,
    RateLoop,
)


def test_rate_loop_tracks_simple_roll_rate_step(cfg):
    gains = RateGains(
        p=(1.5, 2.0, 0.0),
        q=DEFAULT_RATE_GAINS.q,
        r=DEFAULT_RATE_GAINS.r,
    )
    loop = RateLoop(gains=gains, limits=ActuatorLimits.from_config(cfg))
    dt = 0.005
    p = 0.0
    command = np.array([np.deg2rad(30.0), 0.0, 0.0])
    history = []
    for _step in range(400):
        surfaces = loop.update(command, np.array([p, 0.0, 0.0]), dt)
        p += dt * (-20.0 * p + 80.0 * surfaces[0])
        history.append(p)

    values = np.asarray(history)
    target = command[0]
    rise = np.nonzero(values >= 0.9 * target)[0][0] * dt
    overshoot = (np.max(values) - target) / target
    steady_error = abs(values[-1] - target) / target

    assert rise < 0.3
    assert overshoot < 0.2
    assert steady_error < 0.02


def test_rate_loop_outputs_stay_inside_limits(cfg):
    limits = ActuatorLimits.from_config(cfg)
    loop = RateLoop(limits=limits)
    surfaces = loop.update(np.full(3, 100.0), np.zeros(3), 0.005)

    assert limits.delta_a[0] <= surfaces[0] <= limits.delta_a[1]
    assert limits.delta_e[0] <= surfaces[1] <= limits.delta_e[1]
    assert limits.delta_r[0] <= surfaces[2] <= limits.delta_r[1]


def test_rate_loop_gain_vector_round_trips(cfg):
    loop = RateLoop(limits=ActuatorLimits.from_config(cfg))
    gains = loop.gains.copy()
    gains[1] += 0.1
    loop.gains = gains

    np.testing.assert_allclose(loop.gains, gains)
    assert len(loop.gain_names) == gains.size
    assert len(loop.gain_bounds) == gains.size
