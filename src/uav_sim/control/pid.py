"""Discrete PID controller with derivative filtering and anti-windup."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PID:
    """Discrete PID with derivative-on-measurement and back-calculation."""

    kp: float
    ki: float
    kd: float
    N: float = 20.0
    limits: tuple[float, float] = (-np.inf, np.inf)
    Tt: float | None = None
    name: str = ""

    def __post_init__(self) -> None:
        self.integral = 0.0
        self.derivative = 0.0
        self.previous_measurement: float | None = None

    def update(self, setpoint: float, measurement: float, dt: float) -> float:
        """Update the PID and return a saturated command."""
        error = float(setpoint - measurement)
        if dt <= 0.0:
            proportional = self.kp * error
            unsaturated = proportional + self.integral + self.derivative
            return float(np.clip(unsaturated, self.limits[0], self.limits[1]))

        measurement_delta = (
            0.0
            if self.previous_measurement is None
            else float(measurement - self.previous_measurement)
        )
        denom = self.kd + self.N * dt
        if denom > 0.0:
            self.derivative *= self.kd / denom
            self.derivative -= (self.kd * self.N / denom) * measurement_delta
        else:
            self.derivative = 0.0
        self.previous_measurement = float(measurement)

        proportional = self.kp * error
        unsaturated = proportional + self.integral + self.derivative
        saturated = float(np.clip(unsaturated, self.limits[0], self.limits[1]))
        tracking_time = self.Tt if self.Tt is not None else _default_tracking_time(self)
        self.integral += self.ki * dt * error
        if np.isfinite(tracking_time) and tracking_time > 0.0:
            self.integral += (dt / tracking_time) * (saturated - unsaturated)
        return saturated

    def reset(self) -> None:
        """Clear integral and derivative history."""
        self.integral = 0.0
        self.derivative = 0.0
        self.previous_measurement = None


def _default_tracking_time(pid: PID) -> float:
    if pid.ki == 0.0:
        return 1.0
    if pid.kp == 0.0:
        return 1.0
    return max(abs(pid.kp / pid.ki), 1e-3)
