"""Outer-loop heading and TECS-lite controllers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_sim.config import AircraftConfig
from uav_sim.constants import G0
from uav_sim.control.pid import PID


@dataclass(frozen=True)
class TECSGains:
    """Total-energy and energy-balance gains."""

    total: tuple[float, float, float]
    balance: tuple[float, float, float]


@dataclass(frozen=True)
class HeadingGains:
    """Heading-loop PID gains."""

    heading: tuple[float, float, float]


DEFAULT_TECS_GAINS = TECSGains(
    total=(0.030, 0.006, 0.0),
    balance=(0.006, 0.001, 0.0),
)
DEFAULT_HEADING_GAINS = HeadingGains(heading=(0.9, 0.02, 0.0))


class TECSLite:
    """Small-aircraft total-energy controller."""

    def __init__(
        self,
        cfg: AircraftConfig,
        gains: TECSGains = DEFAULT_TECS_GAINS,
    ):
        self.cfg = cfg
        self.total_pid = PID(
            *gains.total,
            limits=(-0.45, 0.45),
            Tt=2.0,
            name="tecs_total",
        )
        self.balance_pid = PID(
            *gains.balance,
            limits=(np.deg2rad(-12.0), np.deg2rad(5.0)),
            Tt=2.0,
            name="tecs_balance",
        )

    def update(
        self,
        h_cmd: float,
        V_cmd: float,
        h_meas: float,
        V_meas: float,
        theta_trim: float,
        throttle_trim: float,
        dt: float,
    ) -> tuple[float, float]:
        """Return ``(theta_cmd, delta_t)``."""
        speed_height = (V_cmd**2 - V_meas**2) / (2.0 * G0)
        height_error = h_cmd - h_meas
        total_error = height_error + speed_height
        balance_error = height_error - speed_height
        throttle_delta = self.total_pid.update(total_error, 0.0, dt)
        theta_delta = self.balance_pid.update(balance_error, 0.0, dt)
        theta_cmd = float(
            np.clip(theta_trim + theta_delta, np.deg2rad(-20.0), np.deg2rad(25.0))
        )
        throttle = float(
            np.clip(
                throttle_trim + throttle_delta,
                self.cfg.actuators.throttle.min,
                self.cfg.actuators.throttle.max,
            )
        )
        return theta_cmd, throttle

    def reset(self) -> None:
        self.total_pid.reset()
        self.balance_pid.reset()

    @property
    def gains(self) -> np.ndarray:
        return np.array(
            [
                self.total_pid.kp,
                self.total_pid.ki,
                self.total_pid.kd,
                self.balance_pid.kp,
                self.balance_pid.ki,
                self.balance_pid.kd,
            ],
            dtype=np.float64,
        )

    @gains.setter
    def gains(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (6,):
            raise ValueError(f"TECSLite gains must have shape (6,), got {arr.shape}")
        self.total_pid.kp, self.total_pid.ki, self.total_pid.kd = arr[0:3]
        self.balance_pid.kp, self.balance_pid.ki, self.balance_pid.kd = arr[3:6]

    @property
    def gain_names(self) -> list[str]:
        return [
            "tecs_total_kp",
            "tecs_total_ki",
            "tecs_total_kd",
            "tecs_balance_kp",
            "tecs_balance_ki",
            "tecs_balance_kd",
        ]

    @property
    def gain_bounds(self) -> list[tuple[float, float]]:
        return [
            (0.0, 0.12),
            (0.0, 0.04),
            (0.0, 0.02),
            (0.0, 0.12),
            (0.0, 0.04),
            (0.0, 0.02),
        ]


class HeadingLoop:
    """Map heading command to roll command."""

    def __init__(
        self,
        cfg: AircraftConfig,
        gains: HeadingGains = DEFAULT_HEADING_GAINS,
    ):
        self.cfg = cfg
        self.pid = PID(*gains.heading, limits=(-0.8, 0.8), Tt=2.0, name="heading")

    def update(self, psi_cmd: float, psi_meas: float, V: float, dt: float) -> float:
        """Return roll command from wrapped heading error."""
        error = wrap_angle(psi_cmd - psi_meas)
        psi_rate_cmd = self.pid.update(error, 0.0, dt)
        phi_cmd = np.arctan2(max(V, 1.0) * psi_rate_cmd, G0)
        return float(
            np.clip(
                phi_cmd, -self.cfg.envelope.phi_max_rad, self.cfg.envelope.phi_max_rad
            )
        )

    def reset(self) -> None:
        self.pid.reset()

    @property
    def gains(self) -> np.ndarray:
        return np.array([self.pid.kp, self.pid.ki, self.pid.kd], dtype=np.float64)

    @gains.setter
    def gains(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"HeadingLoop gains must have shape (3,), got {arr.shape}")
        self.pid.kp, self.pid.ki, self.pid.kd = arr

    @property
    def gain_names(self) -> list[str]:
        return ["heading_kp", "heading_ki", "heading_kd"]

    @property
    def gain_bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 4.0), (0.0, 1.0), (0.0, 0.2)]


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
