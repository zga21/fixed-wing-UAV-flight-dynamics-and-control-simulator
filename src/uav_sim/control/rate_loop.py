"""Inner angular-rate loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_sim.config import AircraftConfig
from uav_sim.control.pid import PID


@dataclass(frozen=True)
class ActuatorLimits:
    """Control surface limits."""

    delta_a: tuple[float, float]
    delta_e: tuple[float, float]
    delta_r: tuple[float, float]

    @classmethod
    def from_config(cls, cfg: AircraftConfig) -> ActuatorLimits:
        return cls(
            (cfg.actuators.aileron.min_rad, cfg.actuators.aileron.max_rad),
            (cfg.actuators.elevator.min_rad, cfg.actuators.elevator.max_rad),
            (cfg.actuators.rudder.min_rad, cfg.actuators.rudder.max_rad),
        )


@dataclass(frozen=True)
class RateGains:
    """PID gains for ``p``, ``q``, and ``r`` rate loops."""

    p: tuple[float, float, float]
    q: tuple[float, float, float]
    r: tuple[float, float, float]


DEFAULT_RATE_GAINS = RateGains(
    p=(0.42, 0.55, 0.012),
    q=(-0.34, -0.70, -0.010),
    r=(0.30, 0.25, 0.006),
)


class RateLoop:
    """Map body-rate commands to aileron, elevator, and rudder increments."""

    def __init__(
        self,
        gains: RateGains = DEFAULT_RATE_GAINS,
        limits: ActuatorLimits | None = None,
    ):
        if limits is None:
            inf = float("inf")
            limits = ActuatorLimits((-inf, inf), (-inf, inf), (-inf, inf))
        self.limits = limits
        self.p_pid = PID(*gains.p, limits=limits.delta_a, name="p_rate")
        self.q_pid = PID(*gains.q, limits=limits.delta_e, name="q_rate")
        self.r_pid = PID(*gains.r, limits=limits.delta_r, name="r_rate")

    def update(
        self,
        omega_cmd: np.ndarray,
        omega_meas: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Return surface increments ``[delta_a, delta_e, delta_r]``."""
        cmd = np.asarray(omega_cmd, dtype=np.float64)
        meas = np.asarray(omega_meas, dtype=np.float64)
        if cmd.shape != (3,) or meas.shape != (3,):
            raise ValueError("omega_cmd and omega_meas must both have shape (3,)")
        return np.array(
            [
                self.p_pid.update(cmd[0], meas[0], dt),
                self.q_pid.update(cmd[1], meas[1], dt),
                self.r_pid.update(cmd[2], meas[2], dt),
            ],
            dtype=np.float64,
        )

    def reset(self) -> None:
        self.p_pid.reset()
        self.q_pid.reset()
        self.r_pid.reset()

    @property
    def gains(self) -> np.ndarray:
        return np.array(
            [
                self.p_pid.kp,
                self.p_pid.ki,
                self.p_pid.kd,
                self.q_pid.kp,
                self.q_pid.ki,
                self.q_pid.kd,
                self.r_pid.kp,
                self.r_pid.ki,
                self.r_pid.kd,
            ],
            dtype=np.float64,
        )

    @gains.setter
    def gains(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (9,):
            raise ValueError(f"RateLoop gains must have shape (9,), got {arr.shape}")
        pids = (self.p_pid, self.q_pid, self.r_pid)
        for idx, pid in enumerate(pids):
            pid.kp, pid.ki, pid.kd = arr[3 * idx : 3 * idx + 3]

    @property
    def gain_names(self) -> list[str]:
        return [
            "rate_p_kp",
            "rate_p_ki",
            "rate_p_kd",
            "rate_q_kp",
            "rate_q_ki",
            "rate_q_kd",
            "rate_r_kp",
            "rate_r_ki",
            "rate_r_kd",
        ]

    @property
    def gain_bounds(self) -> list[tuple[float, float]]:
        return [
            (-2.0, 2.0),
            (-5.0, 5.0),
            (-0.2, 0.2),
            (-2.0, 2.0),
            (-5.0, 5.0),
            (-0.2, 0.2),
            (-2.0, 2.0),
            (-5.0, 5.0),
            (-0.2, 0.2),
        ]
