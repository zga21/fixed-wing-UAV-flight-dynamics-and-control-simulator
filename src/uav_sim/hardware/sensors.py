"""Sampled, biased, noisy sensor models with explicit random streams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from uav_sim.constants import G0
from uav_sim.rotations import quat_to_dcm, quat_to_euler
from uav_sim.state import IDX_OMEGA, IDX_POS, IDX_QUAT, IDX_VEL


@dataclass(frozen=True)
class ChannelSpec:
    rate_hz: float
    sigma_noise: float
    sigma_bias_walk: float = 0.0
    initial_bias: float = 0.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ChannelSpec:
        return cls(
            rate_hz=float(data["rate_hz"]),
            sigma_noise=float(data.get("sigma_noise", 0.0)),
            sigma_bias_walk=float(data.get("sigma_bias_walk", 0.0)),
            initial_bias=float(data.get("initial_bias", 0.0)),
        )


@dataclass(frozen=True)
class SensorSpec:
    ideal: bool
    gyro: ChannelSpec
    accelerometer: ChannelSpec
    gps_position: ChannelSpec
    gps_velocity: ChannelSpec
    pitot: ChannelSpec
    barometer: ChannelSpec
    magnetometer: ChannelSpec


def load_sensor_spec(path: str | Path) -> SensorSpec:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return SensorSpec(
        ideal=bool(raw.get("ideal", False)),
        gyro=ChannelSpec.from_mapping(raw["gyro"]),
        accelerometer=ChannelSpec.from_mapping(raw["accelerometer"]),
        gps_position=ChannelSpec.from_mapping(raw["gps_position"]),
        gps_velocity=ChannelSpec.from_mapping(raw["gps_velocity"]),
        pitot=ChannelSpec.from_mapping(raw["pitot"]),
        barometer=ChannelSpec.from_mapping(raw["barometer"]),
        magnetometer=ChannelSpec.from_mapping(raw["magnetometer"]),
    )


class Sensor:
    """Scalar sampled sensor with random-walk bias and zero-order hold."""

    def __init__(
        self,
        rate_hz: float,
        sigma_noise: float,
        sigma_bias_walk: float,
        rng: np.random.Generator,
        initial_bias: float = 0.0,
    ):
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")
        if sigma_noise < 0.0 or sigma_bias_walk < 0.0:
            raise ValueError("sensor standard deviations cannot be negative")
        self.rate_hz = float(rate_hz)
        self.sigma_noise = float(sigma_noise)
        self.sigma_bias_walk = float(sigma_bias_walk)
        self._initial_bias = float(initial_bias)
        self._rng = rng
        self.bias = self._initial_bias
        self._held: float | None = None
        self._next_sample = 0.0
        self.updated = False
        self.reset()

    def measure(self, true_value: float, t: float) -> float:
        if t < -1e-12:
            raise ValueError("sample time cannot be negative")
        self.updated = False
        if self._held is None or t + 1e-12 >= self._next_sample:
            bias_dt = 0.0 if self._held is None else self.sample_period
            if bias_dt:
                self.bias += (
                    self.sigma_bias_walk * np.sqrt(bias_dt) * self._rng.normal()
                )
            noise = self.sigma_noise * self._rng.normal()
            self._held = float(true_value) + self.bias + noise
            self.updated = True
            while self._next_sample <= t + 1e-12:
                self._next_sample += self.sample_period
        return float(self._held)

    def reset(self, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self._rng = rng
        self.bias = self._initial_bias
        self._held = None
        self._next_sample = 0.0
        self.updated = False

    def sample_due(self, t: float) -> bool:
        return self._held is None or t + 1e-12 >= self._next_sample

    @property
    def sample_period(self) -> float:
        return 1.0 / self.rate_hz


@dataclass(frozen=True)
class IMUMeasurement:
    gyro: np.ndarray
    accelerometer: np.ndarray
    updated: bool


@dataclass(frozen=True)
class GPSMeasurement:
    position_n: np.ndarray
    velocity_n: np.ndarray
    updated: bool


@dataclass(frozen=True)
class Measurements:
    imu: IMUMeasurement
    gps: GPSMeasurement
    pitot_airspeed: float
    barometric_altitude: float
    magnetic_heading: float
    updated: frozenset[str]


class SensorSuite:
    """MEMS IMU, GPS, pitot, barometer, and magnetometer suite."""

    _CHANNEL_COUNT = 15

    def __init__(self, spec: SensorSpec, rng: np.random.Generator):
        self.spec = spec
        self._build_channels(rng)

    def _build_channels(self, rng: np.random.Generator) -> None:
        streams = rng.spawn(self._CHANNEL_COUNT)
        cursor = iter(streams)

        def vector(spec: ChannelSpec) -> tuple[Sensor, Sensor, Sensor]:
            return tuple(self._make_sensor(spec, next(cursor)) for _ in range(3))  # type: ignore[return-value]

        self._gyro = vector(self.spec.gyro)
        self._accel = vector(self.spec.accelerometer)
        self._gps_pos = vector(self.spec.gps_position)
        self._gps_vel = vector(self.spec.gps_velocity)
        self._pitot = self._make_sensor(self.spec.pitot, next(cursor))
        self._baro = self._make_sensor(self.spec.barometer, next(cursor))
        self._mag = self._make_sensor(self.spec.magnetometer, next(cursor))

    @staticmethod
    def _make_sensor(spec: ChannelSpec, rng: np.random.Generator) -> Sensor:
        return Sensor(
            spec.rate_hz,
            spec.sigma_noise,
            spec.sigma_bias_walk,
            rng,
            spec.initial_bias,
        )

    def reset(self, rng: np.random.Generator) -> None:
        self._build_channels(rng)

    def sample_due(self, t: float) -> bool:
        """Return whether any channel needs new truth at time ``t``."""
        if self.spec.ideal:
            return True
        representatives = (
            self._gyro[0],
            self._accel[0],
            self._gps_pos[0],
            self._gps_vel[0],
            self._pitot,
            self._baro,
            self._mag,
        )
        return any(sensor.sample_due(t) for sensor in representatives)

    def measure(
        self,
        x_true: np.ndarray,
        xdot: np.ndarray,
        t: float,
        airspeed_true: float | None = None,
    ) -> Measurements:
        state = np.asarray(x_true, dtype=np.float64)
        derivative = np.asarray(xdot, dtype=np.float64)
        if state.shape != (13,) or derivative.shape != (13,):
            raise ValueError("x_true and xdot must both have shape (13,)")

        rotation = quat_to_dcm(state[IDX_QUAT])
        velocity_n = rotation @ state[IDX_VEL]
        gravity_b = rotation.T @ np.array([0.0, 0.0, G0], dtype=np.float64)
        omega = state[IDX_OMEGA]
        velocity_body = state[IDX_VEL]
        # Inline cross product: np.cross on 3-vectors is ~10x slower (moveaxis
        # overhead). Bit-identical for (3,) float64. See dynamics._cross.
        coriolis = np.array(
            [
                omega[1] * velocity_body[2] - omega[2] * velocity_body[1],
                omega[2] * velocity_body[0] - omega[0] * velocity_body[2],
                omega[0] * velocity_body[1] - omega[1] * velocity_body[0],
            ],
            dtype=np.float64,
        )
        specific_force_b = derivative[IDX_VEL] + coriolis - gravity_b
        heading = float(quat_to_euler(state[IDX_QUAT])[2])
        airspeed = (
            float(np.linalg.norm(state[IDX_VEL]))
            if airspeed_true is None
            else float(airspeed_true)
        )
        altitude = float(-state[2])

        if self.spec.ideal:
            return Measurements(
                imu=IMUMeasurement(state[IDX_OMEGA].copy(), specific_force_b, True),
                gps=GPSMeasurement(state[IDX_POS].copy(), velocity_n, True),
                pitot_airspeed=airspeed,
                barometric_altitude=altitude,
                magnetic_heading=heading,
                updated=frozenset({"imu", "gps", "pitot", "barometer", "magnetometer"}),
            )

        gyro = self._measure_vector(self._gyro, state[IDX_OMEGA], t)
        accel = self._measure_vector(self._accel, specific_force_b, t)
        gps_pos = self._measure_vector(self._gps_pos, state[IDX_POS], t)
        gps_vel = self._measure_vector(self._gps_vel, velocity_n, t)
        pitot = self._pitot.measure(airspeed, t)
        baro = self._baro.measure(altitude, t)
        mag = self._mag.measure(heading, t)
        updated = set()
        if self._gyro[0].updated:
            updated.add("imu")
        if self._gps_pos[0].updated:
            updated.add("gps")
        if self._pitot.updated:
            updated.add("pitot")
        if self._baro.updated:
            updated.add("barometer")
        if self._mag.updated:
            updated.add("magnetometer")
        return Measurements(
            imu=IMUMeasurement(gyro, accel, self._gyro[0].updated),
            gps=GPSMeasurement(gps_pos, gps_vel, self._gps_pos[0].updated),
            pitot_airspeed=pitot,
            barometric_altitude=baro,
            magnetic_heading=mag,
            updated=frozenset(updated),
        )

    @staticmethod
    def _measure_vector(
        sensors: tuple[Sensor, Sensor, Sensor], truth: np.ndarray, t: float
    ) -> np.ndarray:
        return np.array(
            [
                sensor.measure(value, t)
                for sensor, value in zip(sensors, truth, strict=True)
            ],
            dtype=np.float64,
        )

    @property
    def gyro_bias(self) -> np.ndarray:
        return np.array([sensor.bias for sensor in self._gyro], dtype=np.float64)

    @property
    def accelerometer_bias(self) -> np.ndarray:
        return np.array([sensor.bias for sensor in self._accel], dtype=np.float64)
