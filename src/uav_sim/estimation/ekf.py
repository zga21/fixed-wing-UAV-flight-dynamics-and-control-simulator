"""Error-state navigation EKF for the Phase 5 realism pipeline."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from uav_sim.config import AircraftConfig
from uav_sim.constants import G0
from uav_sim.hardware.sensors import GPSMeasurement, IMUMeasurement
from uav_sim.rotations import (
    quat_multiply,
    quat_normalise,
    quat_to_dcm,
    quat_to_euler,
)
from uav_sim.state import IDX_OMEGA, IDX_POS, IDX_QUAT, IDX_VEL, N_STATES

N_ERROR_STATES = 15


def _skew(value: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(value, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _small_angle_quat(angle: np.ndarray) -> np.ndarray:
    vector = np.asarray(angle, dtype=np.float64)
    magnitude = float(np.linalg.norm(vector))
    if magnitude < 1e-12:
        return quat_normalise(np.concatenate(([1.0], 0.5 * vector)))
    half = 0.5 * magnitude
    return np.concatenate(([np.cos(half)], np.sin(half) * vector / magnitude))


def _wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


class NavigationEKF:
    """Fifteen-error-state inertial/GPS navigation filter.

    The nominal state stores NED position and velocity, a body-to-NED
    quaternion, and IMU biases. The covariance uses additive position,
    velocity, small-angle attitude, gyro-bias, and accelerometer-bias errors.
    """

    def __init__(
        self,
        cfg: AircraftConfig,
        Q: np.ndarray | None = None,
        R_dict: Mapping[str, float | np.ndarray] | None = None,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ):
        self.cfg = cfg
        self.Q = self._process_covariance(Q)
        self.R = {
            "gps_position": 4.0,
            "gps_velocity": 0.04,
            "barometer": 2.0**2,
            "airspeed": 1.5**2,
            "magnetometer": np.deg2rad(5.0) ** 2,
            "accelerometer": 0.15**2,
        }
        if R_dict:
            self.R.update(R_dict)
        initial = np.zeros(N_STATES, dtype=np.float64) if x0 is None else x0
        if x0 is None:
            initial[IDX_QUAT] = np.array([1.0, 0.0, 0.0, 0.0])
        self._P0 = self._initial_covariance(P0)
        self.reset(initial, self._P0)

    @staticmethod
    def _process_covariance(value: np.ndarray | None) -> np.ndarray:
        if value is None:
            # Process-noise floors (per-axis std, applied as Q*dt). The down
            # position (index 2) and yaw attitude (index 8) carry ELEVATED
            # floors on purpose: the barometer and magnetometer both have a
            # random-walk bias (config/sensors_v1.yaml), and this 15-state
            # filter does not estimate those biases. Without a floor the filter
            # averages the correlated bias away and P collapses below the true
            # error (overconfident yaw), failing the NEES consistency gate.
            # The floor keeps P at the unobservable-bias level instead.
            standard_deviations = np.array(
                [
                    1e-4,
                    1e-4,
                    0.03,  # p_D: covers unmodelled barometer bias walk
                    0.08,
                    0.08,
                    0.08,
                    0.02,
                    0.02,
                    0.09,  # yaw: covers unmodelled magnetometer bias walk
                    0.001,
                    0.001,
                    0.001,
                    0.005,
                    0.005,
                    0.005,
                ]
            )
            return np.diag(standard_deviations**2)
        array = np.asarray(value, dtype=np.float64)
        if array.shape == (N_ERROR_STATES,):
            return np.diag(array)
        if array.shape != (N_ERROR_STATES, N_ERROR_STATES):
            raise ValueError("Q must have shape (15,) or (15, 15)")
        return array.copy()

    @staticmethod
    def _initial_covariance(value: np.ndarray | None) -> np.ndarray:
        if value is None:
            deviations = np.concatenate(
                [
                    np.full(3, 5.0),
                    np.full(3, 1.0),
                    np.full(3, np.deg2rad(15.0)),
                    np.full(3, 0.03),
                    np.full(3, 0.2),
                ]
            )
            return np.diag(deviations**2)
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (N_ERROR_STATES, N_ERROR_STATES):
            raise ValueError("P0 must have shape (15, 15)")
        return array.copy()

    def reset(self, x0: np.ndarray, P0: np.ndarray | None = None) -> None:
        state = np.asarray(x0, dtype=np.float64)
        if state.shape != (N_STATES,):
            raise ValueError("x0 must have shape (13,)")
        self.position_n = state[IDX_POS].copy()
        rotation = quat_to_dcm(state[IDX_QUAT])
        self.velocity_n = rotation @ state[IDX_VEL]
        self.quat = quat_normalise(state[IDX_QUAT])
        self.gyro_bias = np.zeros(3, dtype=np.float64)
        self.accelerometer_bias = np.zeros(3, dtype=np.float64)
        self.omega_b = state[IDX_OMEGA].copy()
        self._pitot_airspeed: float | None = None
        covariance = self._P0 if P0 is None else np.asarray(P0, dtype=np.float64)
        self.P = covariance.copy()
        self._stabilise_covariance(ensure_positive=True)

    def predict(self, imu: IMUMeasurement, dt: float) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        omega = np.asarray(imu.gyro, dtype=np.float64) - self.gyro_bias
        specific_force = (
            np.asarray(imu.accelerometer, dtype=np.float64) - self.accelerometer_bias
        )
        rotation = quat_to_dcm(self.quat)
        acceleration_n = rotation @ specific_force + np.array([0.0, 0.0, G0])
        self.position_n += self.velocity_n * dt + 0.5 * acceleration_n * dt**2
        self.velocity_n += acceleration_n * dt
        self.quat = quat_normalise(
            quat_multiply(self.quat, _small_angle_quat(omega * dt))
        )
        self.omega_b = omega.copy()

        F = np.zeros((N_ERROR_STATES, N_ERROR_STATES), dtype=np.float64)
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -rotation @ _skew(specific_force)
        F[3:6, 12:15] = -rotation
        F[6:9, 6:9] = -_skew(omega)
        F[6:9, 9:12] = -np.eye(3)
        transition = np.eye(N_ERROR_STATES) + F * dt
        self.P = transition @ self.P @ transition.T + self.Q * dt
        self.P *= 1.0 + dt
        self._stabilise_covariance(ensure_positive=False)

    def update_accelerometer(self, imu: IMUMeasurement) -> None:
        measured = np.asarray(imu.accelerometer, dtype=np.float64)
        if not 0.5 * G0 <= np.linalg.norm(measured) <= 1.5 * G0:
            return

        def model(quat: np.ndarray) -> np.ndarray:
            return (
                -quat_to_dcm(quat).T @ np.array([0.0, 0.0, G0])
                + self.accelerometer_bias
            )

        predicted = model(self.quat)
        if (
            np.linalg.norm(self.velocity_n) > 5.0
            and np.linalg.norm(measured - predicted) > 1.0
        ):
            return
        H = np.zeros((3, N_ERROR_STATES), dtype=np.float64)
        epsilon = 1e-6
        for axis in range(3):
            perturbation = np.zeros(3)
            perturbation[axis] = epsilon
            perturbed_quat = quat_multiply(self.quat, _small_angle_quat(perturbation))
            H[:, 6 + axis] = (model(perturbed_quat) - predicted) / epsilon
        if np.linalg.norm(measured - predicted) < 0.2:
            H[:, 12:15] = np.eye(3)
        self._update(measured - predicted, H, self.R["accelerometer"])

    def update_gps(self, gps: GPSMeasurement) -> None:
        measurement = np.concatenate((gps.position_n, gps.velocity_n))
        predicted = np.concatenate((self.position_n, self.velocity_n))
        H = np.zeros((6, N_ERROR_STATES), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)
        covariance = np.diag(
            np.concatenate(
                (
                    np.full(3, float(self.R["gps_position"])),
                    np.full(3, float(self.R["gps_velocity"])),
                )
            )
        )
        self._update(measurement - predicted, H, covariance)

    def update_baro(self, altitude: float) -> None:
        H = np.zeros((1, N_ERROR_STATES), dtype=np.float64)
        H[0, 2] = -1.0
        innovation = np.array([float(altitude) + self.position_n[2]])
        self._update(innovation, H, self.R["barometer"])

    def update_airspeed(self, airspeed: float) -> None:
        speed = float(np.linalg.norm(self.velocity_n))
        H = np.zeros((1, N_ERROR_STATES), dtype=np.float64)
        if speed > 1e-6:
            H[0, 3:6] = self.velocity_n / speed
        self._update(np.array([float(airspeed) - speed]), H, self.R["airspeed"])
        self._pitot_airspeed = float(airspeed)

    def update_magnetometer(self, heading: float) -> None:
        predicted = float(quat_to_euler(self.quat)[2])
        H = np.zeros((1, N_ERROR_STATES), dtype=np.float64)
        epsilon = 1e-6
        for axis in range(3):
            perturbation = np.zeros(3)
            perturbation[axis] = epsilon
            perturbed = quat_multiply(self.quat, _small_angle_quat(perturbation))
            delta = _wrap_angle(float(quat_to_euler(perturbed)[2]) - predicted)
            H[0, 6 + axis] = delta / epsilon
        self._update(
            np.array([_wrap_angle(float(heading) - predicted)]),
            H,
            self.R["magnetometer"],
        )

    def _update(
        self,
        innovation: np.ndarray,
        H: np.ndarray,
        covariance: float | np.ndarray,
    ) -> None:
        residual = np.asarray(innovation, dtype=np.float64)
        R = np.asarray(covariance, dtype=np.float64)
        if R.ndim == 0:
            R = np.eye(residual.size) * float(R)
        innovation_covariance = H @ self.P @ H.T + R
        gain = np.linalg.solve(innovation_covariance, H @ self.P).T
        error = gain @ residual
        self._inject(error)
        identity = np.eye(N_ERROR_STATES)
        correction = identity - gain @ H
        self.P = correction @ self.P @ correction.T + gain @ R @ gain.T
        self._stabilise_covariance(ensure_positive=True)

    def _inject(self, error: np.ndarray) -> None:
        self.position_n += error[0:3]
        self.velocity_n += error[3:6]
        self.quat = quat_normalise(
            quat_multiply(self.quat, _small_angle_quat(error[6:9]))
        )
        self.gyro_bias += error[9:12]
        self.accelerometer_bias += error[12:15]

    def _stabilise_covariance(self, ensure_positive: bool) -> None:
        self.P = 0.5 * (self.P + self.P.T)
        if ensure_positive:
            minimum = float(np.linalg.eigvalsh(self.P).min())
            if minimum <= 1e-12:
                self.P += np.eye(N_ERROR_STATES) * (1e-12 - minimum + 1e-15)

    @property
    def state(self) -> np.ndarray:
        state = np.empty(N_STATES, dtype=np.float64)
        state[IDX_POS] = self.position_n
        velocity_b = quat_to_dcm(self.quat).T @ self.velocity_n
        if self._pitot_airspeed is not None:
            speed = np.linalg.norm(velocity_b)
            if speed > 1e-9:
                velocity_b = velocity_b * (self._pitot_airspeed / speed)
        state[IDX_VEL] = velocity_b
        state[IDX_QUAT] = self.quat
        state[IDX_OMEGA] = self.omega_b
        return state

    @property
    def covariance_diagonal(self) -> np.ndarray:
        return np.diag(self.P).copy()

    def error_state(
        self,
        x_true: np.ndarray,
        gyro_bias_true: np.ndarray | None = None,
        accelerometer_bias_true: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the 15-state truth-minus-estimate error for NEES checks."""
        truth = np.asarray(x_true, dtype=np.float64)
        if truth.shape != (N_STATES,):
            raise ValueError("x_true must have shape (13,)")
        true_rotation = quat_to_dcm(truth[IDX_QUAT])
        true_velocity_n = true_rotation @ truth[IDX_VEL]
        conjugate = self.quat.copy()
        conjugate[1:4] *= -1.0
        attitude_error_quat = quat_multiply(conjugate, truth[IDX_QUAT])
        if attitude_error_quat[0] < 0.0:
            attitude_error_quat *= -1.0
        gyro_truth = (
            np.zeros(3) if gyro_bias_true is None else np.asarray(gyro_bias_true)
        )
        accel_truth = (
            np.zeros(3)
            if accelerometer_bias_true is None
            else np.asarray(accelerometer_bias_true)
        )
        return np.concatenate(
            [
                truth[IDX_POS] - self.position_n,
                true_velocity_n - self.velocity_n,
                2.0 * attitude_error_quat[1:4],
                gyro_truth - self.gyro_bias,
                accel_truth - self.accelerometer_bias,
            ]
        )
