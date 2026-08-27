"""Deterministic servo and throttle actuator dynamics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from uav_sim.config import AircraftConfig


class Actuator:
    """First-order lag, followed by rate and position limits."""

    def __init__(
        self,
        tau: float,
        rate_limit: float,
        pos_limits: tuple[float, float],
        initial: float = 0.0,
    ):
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        if rate_limit <= 0.0:
            raise ValueError("rate_limit must be positive")
        if pos_limits[0] >= pos_limits[1]:
            raise ValueError("position limits must be ordered")
        self.tau = float(tau)
        self.rate_limit = float(rate_limit)
        self.pos_limits = (float(pos_limits[0]), float(pos_limits[1]))
        self._initial = float(initial)
        self._position = 0.0
        self._is_saturated = False
        self._is_rate_limited = False
        self.reset(initial)

    def step(self, command: float, dt: float) -> float:
        """Advance one simulation step using lag, rate, then position limits."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        raw_rate = (float(command) - self._position) / self.tau
        rate = float(np.clip(raw_rate, -self.rate_limit, self.rate_limit))
        candidate = self._position + rate * dt
        position = float(np.clip(candidate, *self.pos_limits))
        self._is_rate_limited = not np.isclose(rate, raw_rate, rtol=0.0, atol=1e-15)
        self._is_saturated = not np.isclose(
            position, candidate, rtol=0.0, atol=1e-15
        ) or (
            (position <= self.pos_limits[0] and command < self.pos_limits[0])
            or (position >= self.pos_limits[1] and command > self.pos_limits[1])
        )
        self._position = position
        return position

    def reset(self, initial: float = 0.0) -> None:
        """Reset position and clear limit flags."""
        if not self.pos_limits[0] <= initial <= self.pos_limits[1]:
            raise ValueError("initial position must lie inside position limits")
        self._position = float(initial)
        self._is_saturated = False
        self._is_rate_limited = False

    @property
    def position(self) -> float:
        return self._position

    @property
    def is_saturated(self) -> bool:
        return self._is_saturated

    @property
    def is_rate_limited(self) -> bool:
        return self._is_rate_limited


class ActuatorBank:
    """The three control surfaces and throttle, advanced as one unit."""

    def __init__(self, actuators: Sequence[Actuator], ideal: bool = False):
        if len(actuators) != 4:
            raise ValueError("actuator bank requires exactly four actuators")
        self.actuators = tuple(actuators)
        self.ideal = bool(ideal)
        self._last = np.zeros(4, dtype=np.float64)
        self._saturated_steps = 0
        self._rate_limited_steps = 0
        self._steps = 0

    @classmethod
    def from_config(
        cls, cfg: AircraftConfig, ideal: bool = False, initial: np.ndarray | None = None
    ) -> ActuatorBank:
        values = (
            np.zeros(4) if initial is None else np.asarray(initial, dtype=np.float64)
        )
        if values.shape != (4,):
            raise ValueError("initial controls must have shape (4,)")
        specs = (
            cfg.actuators.aileron,
            cfg.actuators.elevator,
            cfg.actuators.rudder,
        )
        actuators = [
            Actuator(
                spec.tau,
                spec.rate_rad_s,
                (spec.min_rad, spec.max_rad),
                float(values[index]),
            )
            for index, spec in enumerate(specs)
        ]
        throttle = cfg.actuators.throttle
        actuators.append(
            Actuator(
                throttle.tau,
                throttle.rate,
                (throttle.min, throttle.max),
                float(values[3]),
            )
        )
        return cls(actuators, ideal=ideal)

    def step(self, commands: np.ndarray, dt: float) -> np.ndarray:
        command = np.asarray(commands, dtype=np.float64)
        if command.shape != (4,):
            raise ValueError(f"commands must have shape (4,), got {command.shape}")
        if self.ideal:
            self._last = command.copy()
        else:
            self._last = np.array(
                [
                    actuator.step(value, dt)
                    for actuator, value in zip(self.actuators, command, strict=True)
                ],
                dtype=np.float64,
            )
        self._steps += 1
        self._saturated_steps += int(any(a.is_saturated for a in self.actuators[:3]))
        self._rate_limited_steps += int(
            any(a.is_rate_limited for a in self.actuators[:3])
        )
        return self._last.copy()

    def reset(self, initial: np.ndarray | None = None) -> None:
        values = (
            np.zeros(4) if initial is None else np.asarray(initial, dtype=np.float64)
        )
        if values.shape != (4,):
            raise ValueError("initial controls must have shape (4,)")
        for actuator, value in zip(self.actuators, values, strict=True):
            actuator.reset(float(value))
        self._last = values.copy()
        self._saturated_steps = 0
        self._rate_limited_steps = 0
        self._steps = 0

    @property
    def values(self) -> np.ndarray:
        return self._last.copy()

    @property
    def is_saturated(self) -> bool:
        return any(actuator.is_saturated for actuator in self.actuators)

    @property
    def is_rate_limited(self) -> bool:
        return any(actuator.is_rate_limited for actuator in self.actuators)

    @property
    def saturation_fraction(self) -> float:
        return self._saturated_steps / self._steps if self._steps else 0.0

    @property
    def rate_limited_fraction(self) -> float:
        return self._rate_limited_steps / self._steps if self._steps else 0.0
