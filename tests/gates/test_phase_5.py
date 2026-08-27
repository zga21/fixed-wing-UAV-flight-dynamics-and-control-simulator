"""Phase 5 realism, estimation, and reproducibility gate."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import welch
from scipy.stats import chi2

from uav_sim.config import load_aircraft
from uav_sim.control.baseline import CascadedPIDAutopilot
from uav_sim.environment.wind import DrydenTurbulence
from uav_sim.estimation.ekf import NavigationEKF
from uav_sim.hardware.actuator import Actuator
from uav_sim.hardware.sensors import Sensor, SensorSuite, load_sensor_spec
from uav_sim.realism import build_plant, degradation_configurations
from uav_sim.scenario import load_scenario, run_scenario
from uav_sim.state import initial_state


def _short_pipeline_run(seed: int) -> tuple[np.ndarray, np.ndarray]:
    cfg = load_aircraft("config/aircraft_v1.yaml")
    scenario = replace(
        load_scenario("config/scenarios/disturbance_rejection.yaml"), duration=2.0
    )
    plant = build_plant(cfg, degradation_configurations()["C5 realistic"])
    result, _metrics = run_scenario(
        scenario, CascadedPIDAutopilot(cfg), plant, cfg, seed=seed
    )
    return result.x, result.u_actual


@pytest.mark.gate
def test_actuator_rate_limit_timing():
    actuator = Actuator(0.05, 2.0, (-0.4, 0.4))
    dt = 0.001
    steps = 0
    while actuator.position < 0.4:
        actuator.step(10.0, dt)
        steps += 1
    assert steps * dt == pytest.approx(0.2, abs=1e-12)


@pytest.mark.gate
def test_sensor_noise_statistics():
    sensor = Sensor(500.0, 0.5, 0.0, np.random.default_rng(10))
    values = np.array([sensor.measure(0.0, index / 500.0) for index in range(100_000)])
    assert np.std(values, ddof=1) == pytest.approx(0.5, rel=0.05)


@pytest.mark.gate
@pytest.mark.slow
def test_turbulence_spectrum_matches_dryden():
    dt = 0.01
    speed = 25.0
    turbulence = DrydenTurbulence("light", 100.0, np.random.default_rng(1))
    samples = np.array([turbulence.step(speed, dt) for _ in range(60_000)])
    settled = samples[10_000:]
    np.testing.assert_allclose(np.std(settled, axis=0), turbulence.sigmas, rtol=0.1)
    frequency_hz, psd = welch(settled[:, 0], fs=1.0 / dt, nperseg=8192)
    omega = 2.0 * np.pi * frequency_hz
    tau = turbulence.length_scales[0] / speed
    analytic = 1.0 / (1.0 + (tau * omega) ** 2)
    band = (omega >= 0.1) & (omega <= 10.0)
    ratio = (psd[band] / psd[band][0]) / (analytic[band] / analytic[band][0])
    assert np.mean((ratio > 0.5) & (ratio < 2.0)) > 0.9


@pytest.mark.gate
@pytest.mark.slow
def test_ekf_consistency_nees(cfg):
    truth = initial_state(0.0, h=100.0)
    suite = SensorSuite(
        load_sensor_spec("config/sensors_v1.yaml"), np.random.default_rng(3)
    )
    ekf = NavigationEKF(cfg, x0=truth)
    nees = []
    for index in range(4_000):
        measurement = suite.measure(truth, np.zeros(13), index * 0.005, 0.0)
        if index:
            ekf.predict(measurement.imu, 0.005)
        if index and "imu" in measurement.updated:
            ekf.update_accelerometer(measurement.imu)
        if "gps" in measurement.updated:
            ekf.update_gps(measurement.gps)
        if "barometer" in measurement.updated:
            ekf.update_baro(measurement.barometric_altitude)
        if "pitot" in measurement.updated:
            ekf.update_airspeed(measurement.pitot_airspeed)
        if "magnetometer" in measurement.updated:
            ekf.update_magnetometer(measurement.magnetic_heading)
        error = ekf.error_state(truth, suite.gyro_bias, suite.accelerometer_bias)
        nees.append(float(error @ np.linalg.solve(ekf.P, error)))
    settled = np.asarray(nees[1_000:])
    lower, upper = chi2.ppf([0.025, 0.975], 15)
    in_bounds = np.mean((settled >= lower) & (settled <= upper))
    assert 0.5 * 15 <= np.mean(settled) <= 2.0 * 15
    assert in_bounds > 0.6
    assert np.linalg.eigvalsh(ekf.P).min() > 0.0


@pytest.mark.gate
def test_full_pipeline_bit_reproducible():
    first_x, first_u = _short_pipeline_run(1234)
    second_x, second_u = _short_pipeline_run(1234)
    np.testing.assert_array_equal(first_x, second_x)
    np.testing.assert_array_equal(first_u, second_u)


@pytest.mark.gate
def test_different_seeds_differ():
    first_x, _ = _short_pipeline_run(1234)
    second_x, _ = _short_pipeline_run(5678)
    assert not np.array_equal(first_x, second_x)


@pytest.mark.gate
@pytest.mark.slow
def test_full_pipeline_reproducible_under_spawn():
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_short_pipeline_run, 1234) for _ in range(2)]
        first, second = [future.result() for future in futures]
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


@pytest.mark.gate
def test_degradation_recorded():
    path = Path("docs/degradation_table.md")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "C0 perfect" in text
    assert "C5 realistic" in text
    assert "Dominant mechanism" in text
