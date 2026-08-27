"""Sensor statistics, sampling, independence, and reproducibility tests."""

from dataclasses import replace

import numpy as np

from uav_sim.hardware.sensors import Sensor, SensorSuite, load_sensor_spec
from uav_sim.state import initial_state


def test_sensor_noise_matches_requested_standard_deviation():
    sensor = Sensor(500.0, 0.7, 0.0, np.random.default_rng(123))
    samples = np.array([sensor.measure(0.0, index / 500.0) for index in range(100_000)])
    assert abs(np.std(samples, ddof=1) / 0.7 - 1.0) < 0.05


def test_five_hz_sensor_holds_for_one_hundred_simulation_steps():
    sensor = Sensor(5.0, 1.0, 0.0, np.random.default_rng(12))
    samples = np.array([sensor.measure(0.0, index / 500.0) for index in range(101)])
    np.testing.assert_array_equal(samples[:100], np.full(100, samples[0]))
    assert samples[100] != samples[99]


def test_bias_walk_ensemble_variance_grows_linearly():
    final_biases = []
    for seed in range(500):
        sensor = Sensor(10.0, 0.0, 0.2, np.random.default_rng(seed))
        for index in range(101):
            sensor.measure(0.0, index / 10.0)
        final_biases.append(sensor.bias)
    expected_variance = 0.2**2 * 10.0
    np.testing.assert_allclose(
        np.var(final_biases, ddof=1), expected_variance, rtol=0.15
    )


def test_sensor_suites_are_seed_reproducible_and_independent():
    spec = load_sensor_spec("config/sensors_v1.yaml")
    first = SensorSuite(spec, np.random.default_rng(44))
    second = SensorSuite(spec, np.random.default_rng(44))
    different = SensorSuite(spec, np.random.default_rng(45))
    x = initial_state(25.0, h=100.0)
    xdot = np.zeros(13)
    sequence_a = []
    sequence_b = []
    sequence_c = []
    for index in range(500):
        t = index / 500.0
        sequence_a.append(first.measure(x, xdot, t).imu.gyro)
        sequence_b.append(second.measure(x, xdot, t).imu.gyro)
        sequence_c.append(different.measure(x, xdot, t).imu.gyro)
    np.testing.assert_array_equal(sequence_a, sequence_b)
    assert not np.array_equal(sequence_a, sequence_c)
    correlation = np.corrcoef(np.asarray(sequence_a).T)
    assert np.max(np.abs(correlation - np.eye(3))) < 0.2


def test_ideal_suite_returns_truth_without_modifying_state():
    spec = load_sensor_spec("config/sensors_v1.yaml")
    ideal_spec = replace(spec, ideal=True)
    suite = SensorSuite(ideal_spec, np.random.default_rng(9))
    x = initial_state(25.0, h=100.0)
    original = x.copy()
    measurement = suite.measure(x, np.zeros(13), 0.0)
    np.testing.assert_array_equal(x, original)
    np.testing.assert_array_equal(measurement.imu.gyro, x[10:13])
    np.testing.assert_array_equal(measurement.gps.position_n, x[0:3])
    assert measurement.pitot_airspeed == 25.0
