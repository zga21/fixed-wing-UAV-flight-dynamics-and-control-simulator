"""Wind-field and Dryden turbulence validation tests."""

import numpy as np
import pytest
from scipy.signal import welch

from uav_sim.air_data import compute_air_data
from uav_sim.environment.wind import DiscreteGust, DrydenTurbulence, SteadyWind


def test_north_headwind_increases_air_relative_speed():
    wind = SteadyWind(10.0, direction_rad=0.0)
    air = compute_air_data(
        np.array([25.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        100.0,
        wind.velocity(100.0),
    )
    assert air.V == pytest.approx(35.0)


def test_zero_turbulence_is_exactly_zero():
    turbulence = DrydenTurbulence("zero", 100.0, np.random.default_rng(1))
    for _ in range(100):
        np.testing.assert_array_equal(turbulence.step(25.0, 0.002), np.zeros(3))


def test_turbulence_is_bit_reproducible_and_seed_sensitive():
    first = DrydenTurbulence("moderate", 100.0, np.random.default_rng(123))
    second = DrydenTurbulence("moderate", 100.0, np.random.default_rng(123))
    third = DrydenTurbulence("moderate", 100.0, np.random.default_rng(124))
    a = np.array([first.step(25.0, 0.01) for _ in range(2_000)])
    b = np.array([second.step(25.0, 0.01) for _ in range(2_000)])
    c = np.array([third.step(25.0, 0.01) for _ in range(2_000)])
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.slow
def test_dryden_standard_deviation_and_longitudinal_spectrum():
    dt = 0.01
    V = 25.0
    turbulence = DrydenTurbulence("light", 100.0, np.random.default_rng(1))
    samples = np.array([turbulence.step(V, dt) for _ in range(60_000)])
    settled = samples[10_000:]
    np.testing.assert_allclose(np.std(settled, axis=0), turbulence.sigmas, rtol=0.1)

    frequency_hz, psd = welch(settled[:, 0], fs=1.0 / dt, nperseg=8192)
    omega = 2.0 * np.pi * frequency_hz
    tau = turbulence.length_scales[0] / V
    analytic_shape = 1.0 / (1.0 + (tau * omega) ** 2)
    band = (omega >= 0.1) & (omega <= 10.0)
    measured = psd[band] / psd[band][0]
    expected = analytic_shape[band] / analytic_shape[band][0]
    ratio = measured / expected
    assert np.median(ratio[(ratio > 0.5) & (ratio < 2.0)]) > 0.8
    assert np.mean((ratio > 0.5) & (ratio < 2.0)) > 0.55


def test_dryden_filter_stays_finite_over_airspeed_envelope():
    turbulence = DrydenTurbulence("severe", 100.0, np.random.default_rng(4))
    values = []
    for speed in np.linspace(15.0, 40.0, 5_000):
        values.append(turbulence.step(float(speed), 0.002))
    assert np.all(np.isfinite(values))
    assert np.max(np.abs(values)) < 50.0


def test_one_minus_cosine_gust_peak_and_duration():
    gust = DiscreteGust(2.0, 4.0, 6.0, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(gust.velocity(1.99), np.zeros(3))
    assert gust.velocity(4.0)[2] == pytest.approx(6.0)
    np.testing.assert_allclose(gust.velocity(2.0), np.zeros(3), atol=1e-15)
    np.testing.assert_allclose(gust.velocity(6.0), np.zeros(3), atol=1e-15)
    np.testing.assert_array_equal(gust.velocity(6.01), np.zeros(3))
