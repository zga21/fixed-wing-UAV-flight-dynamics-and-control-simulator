"""Navigation EKF convergence and covariance tests."""

import numpy as np

from uav_sim.constants import G0
from uav_sim.estimation.ekf import NavigationEKF
from uav_sim.hardware.sensors import GPSMeasurement, IMUMeasurement
from uav_sim.rotations import euler_to_quat, quat_to_euler
from uav_sim.state import IDX_QUAT, initial_state


def _stationary_imu(gyro=None):
    value = np.zeros(3) if gyro is None else np.asarray(gyro)
    return IMUMeasurement(value, np.array([0.0, 0.0, -G0]), True)


def test_ekf_recovers_from_thirty_degree_attitude_error(cfg):
    x0 = initial_state(0.0, h=100.0)
    x0[IDX_QUAT] = euler_to_quat(np.deg2rad(30.0), np.deg2rad(-30.0), np.deg2rad(30.0))
    ekf = NavigationEKF(cfg, x0=x0)
    imu = _stationary_imu()
    gps = GPSMeasurement(np.array([0.0, 0.0, -100.0]), np.zeros(3), True)
    dt = 0.005
    errors = []
    for index in range(int(10.0 / dt)):
        ekf.predict(imu, dt)
        ekf.update_accelerometer(imu)
        if index % 4 == 0:
            ekf.update_magnetometer(0.0)
        if index % 40 == 0:
            ekf.update_gps(gps)
        errors.append(np.rad2deg(quat_to_euler(ekf.state[IDX_QUAT])))
    assert np.sqrt(np.mean(np.square(errors[-200:]))) < 2.0


def test_ekf_estimates_constant_gyro_bias(cfg):
    x0 = initial_state(0.0, h=100.0)
    ekf = NavigationEKF(cfg, x0=x0)
    true_bias = np.array([0.02, -0.01, 0.015])
    imu = _stationary_imu(true_bias)
    for index in range(4_000):
        ekf.predict(imu, 0.005)
        ekf.update_accelerometer(imu)
        if index % 4 == 0:
            ekf.update_magnetometer(0.0)
    np.testing.assert_allclose(ekf.gyro_bias, true_bias, rtol=0.2, atol=0.003)


def test_ekf_covariance_stays_symmetric_positive_definite(cfg):
    ekf = NavigationEKF(cfg, x0=initial_state(0.0, h=100.0))
    imu = _stationary_imu()
    for index in range(1_000):
        ekf.predict(imu, 0.01)
        if index % 2 == 0:
            ekf.update_accelerometer(imu)
        np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-13)
        assert np.linalg.eigvalsh(ekf.P).min() > 0.0


def test_ideal_stationary_measurements_keep_near_zero_error(cfg):
    truth = initial_state(0.0, h=100.0)
    ekf = NavigationEKF(cfg, x0=truth, P0=np.eye(15) * 1e-6)
    imu = _stationary_imu()
    gps = GPSMeasurement(truth[0:3], np.zeros(3), True)
    for index in range(500):
        ekf.predict(imu, 0.01)
        ekf.update_accelerometer(imu)
        if index % 20 == 0:
            ekf.update_gps(gps)
            ekf.update_baro(100.0)
            ekf.update_airspeed(0.0)
            ekf.update_magnetometer(0.0)
    np.testing.assert_allclose(ekf.state, truth, atol=1e-5)
