"""Steady wind, Dryden turbulence, and deterministic discrete gusts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.signal import cont2discrete, tf2ss

from uav_sim.rotations import quat_to_dcm


class SteadyWind:
    """Constant horizontal wind with optional power-law altitude shear.

    ``direction_rad`` is the meteorological direction the wind comes from,
    measured clockwise from north. A direction of zero therefore gives a
    negative north NED velocity.
    """

    def __init__(
        self,
        speed: float,
        direction_rad: float,
        shear_exponent: float = 0.0,
        reference_altitude: float = 100.0,
    ):
        if speed < 0.0:
            raise ValueError("wind speed cannot be negative")
        if shear_exponent < 0.0:
            raise ValueError("shear exponent cannot be negative")
        if reference_altitude <= 0.0:
            raise ValueError("reference altitude must be positive")
        self.speed = float(speed)
        self.direction_rad = float(direction_rad)
        self.shear_exponent = float(shear_exponent)
        self.reference_altitude = float(reference_altitude)

    def velocity(self, altitude: float) -> np.ndarray:
        factor = 1.0
        if self.shear_exponent:
            factor = (
                max(float(altitude), 0.1) / self.reference_altitude
            ) ** self.shear_exponent
        speed = self.speed * factor
        return np.array(
            [
                -speed * np.cos(self.direction_rad),
                -speed * np.sin(self.direction_rad),
                0.0,
            ],
            dtype=np.float64,
        )


class _DrydenAxis:
    def __init__(
        self,
        sigma: float,
        length_scale: float,
        second_order: bool,
        rng: np.random.Generator,
    ):
        self.sigma = float(sigma)
        self.length_scale = float(length_scale)
        self.second_order = bool(second_order)
        self.rng = rng
        self.state = np.zeros(2 if second_order else 1, dtype=np.float64)
        self._V = -1.0
        self._dt = -1.0
        self._Ad = np.eye(self.state.size)
        self._Bd = np.zeros(self.state.size)
        self._Cd = np.zeros(self.state.size)

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.state.fill(0.0)
        self._V = -1.0
        self._dt = -1.0

    def step(self, V: float, dt: float) -> float:
        if self.sigma == 0.0:
            return 0.0
        if V <= 0.0 or dt <= 0.0:
            raise ValueError("airspeed and time step must be positive")
        if (
            self._V <= 0.0
            or abs(V - self._V) / self._V > 0.02
            or not np.isclose(dt, self._dt)
        ):
            self._configure(V, dt)
        white = self.rng.normal() / np.sqrt(dt)
        self.state = self._Ad @ self.state + self._Bd * white
        return float(self._Cd @ self.state)

    def _configure(self, V: float, dt: float) -> None:
        tau = self.length_scale / float(V)
        if self.second_order:
            numerator = [np.sqrt(3.0) * tau, 1.0]
            denominator = [tau * tau, 2.0 * tau, 1.0]
        else:
            numerator = [1.0]
            denominator = [tau, 1.0]
        A, B, C, D = tf2ss(numerator, denominator)
        Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), dt, method="zoh")
        input_covariance = (Bd @ Bd.T) / dt
        stationary = solve_discrete_lyapunov(Ad, input_covariance)
        raw_variance = float((Cd @ stationary @ Cd.T + Dd @ Dd.T / dt).item())
        scale = self.sigma / np.sqrt(max(raw_variance, np.finfo(float).tiny))
        self._Ad = np.asarray(Ad, dtype=np.float64)
        self._Bd = np.asarray(Bd[:, 0], dtype=np.float64)
        self._Cd = np.asarray(Cd[0] * scale, dtype=np.float64)
        self._V = float(V)
        self._dt = float(dt)


class DrydenTurbulence:
    """Three-axis Dryden shaping filters updated at the simulation rate."""

    _SIGMA = {"zero": 0.0, "light": 1.0, "moderate": 2.0, "severe": 4.0}

    def __init__(
        self,
        intensity: str,
        altitude: float,
        rng: np.random.Generator,
    ):
        if intensity not in self._SIGMA:
            raise ValueError(f"unknown turbulence intensity: {intensity}")
        self.intensity = intensity
        self.altitude = float(altitude)
        sigma_w = self._SIGMA[intensity]
        altitude_scale = float(np.clip(max(altitude, 1.0) / 100.0, 0.5, 2.0))
        self.sigmas = np.array([1.15 * sigma_w, 1.15 * sigma_w, sigma_w])
        self.length_scales = np.array(
            [200.0 * altitude_scale, 200.0 * altitude_scale, 50.0 * altitude_scale]
        )
        streams = rng.spawn(3)
        self._axes = (
            _DrydenAxis(self.sigmas[0], self.length_scales[0], False, streams[0]),
            _DrydenAxis(self.sigmas[1], self.length_scales[1], True, streams[1]),
            _DrydenAxis(self.sigmas[2], self.length_scales[2], True, streams[2]),
        )

    def step(self, V: float, dt: float) -> np.ndarray:
        return np.array([axis.step(V, dt) for axis in self._axes], dtype=np.float64)

    def reset(self, rng: np.random.Generator) -> None:
        streams = rng.spawn(3)
        for axis, stream in zip(self._axes, streams, strict=True):
            axis.reset(stream)


@dataclass(frozen=True)
class DiscreteGust:
    """One-minus-cosine gust in a fixed NED direction."""

    start: float
    duration: float
    magnitude: float
    direction_n: np.ndarray

    def __post_init__(self) -> None:
        direction = np.asarray(self.direction_n, dtype=np.float64)
        if self.duration <= 0.0:
            raise ValueError("gust duration must be positive")
        norm = np.linalg.norm(direction)
        if norm <= 0.0:
            raise ValueError("gust direction cannot be zero")
        object.__setattr__(self, "direction_n", direction / norm)

    def velocity(self, t: float) -> np.ndarray:
        elapsed = float(t) - self.start
        if elapsed < 0.0 or elapsed > self.duration:
            return np.zeros(3, dtype=np.float64)
        amplitude = (
            0.5 * self.magnitude * (1.0 - np.cos(2.0 * np.pi * elapsed / self.duration))
        )
        return amplitude * self.direction_n


class WindField:
    """Compose steady wind, body-axis turbulence, and discrete gusts."""

    def __init__(
        self,
        steady: SteadyWind | None = None,
        turbulence: DrydenTurbulence | None = None,
        gusts: tuple[DiscreteGust, ...] = (),
    ):
        self.steady = steady
        self.turbulence = turbulence
        self.gusts = tuple(gusts)

    def velocity_n(
        self,
        t: float,
        pos_n: np.ndarray,
        V: float,
        dt: float,
        quat: np.ndarray | None = None,
    ) -> np.ndarray:
        altitude = float(-np.asarray(pos_n, dtype=np.float64)[2])
        velocity = (
            np.zeros(3, dtype=np.float64)
            if self.steady is None
            else self.steady.velocity(altitude)
        )
        if self.turbulence is not None:
            turbulence_b = self.turbulence.step(max(float(V), 0.1), dt)
            velocity = velocity + (
                turbulence_b
                if quat is None
                else quat_to_dcm(np.asarray(quat, dtype=np.float64)) @ turbulence_b
            )
        for gust in self.gusts:
            velocity = velocity + gust.velocity(t)
        return velocity

    def reset(self, rng: np.random.Generator) -> None:
        if self.turbulence is not None:
            self.turbulence.reset(rng)
