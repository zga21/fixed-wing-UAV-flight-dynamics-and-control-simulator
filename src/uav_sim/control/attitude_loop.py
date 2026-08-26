"""Middle attitude loop."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_sim.config import AircraftConfig
from uav_sim.constants import G0
from uav_sim.control.pid import PID


@dataclass(frozen=True)
class RateLimits:
    """Body-rate command limits."""

    p: tuple[float, float]
    q: tuple[float, float]
    r: tuple[float, float]

    @classmethod
    def default(cls) -> RateLimits:
        limit = np.deg2rad(120.0)
        yaw = np.deg2rad(80.0)
        return cls((-limit, limit), (-limit, limit), (-yaw, yaw))


@dataclass(frozen=True)
class AttitudeGains:
    """P/PI gains for roll and pitch attitude loops."""

    phi: tuple[float, float, float]
    theta: tuple[float, float, float]


DEFAULT_ATTITUDE_GAINS = AttitudeGains(
    phi=(5.0, 0.15, 0.0),
    theta=(4.2, 0.20, 0.0),
)


class AttitudeLoop:
    """Map roll/pitch commands to body-rate commands."""

    def __init__(
        self,
        gains: AttitudeGains = DEFAULT_ATTITUDE_GAINS,
        limits: RateLimits | None = None,
        cfg: AircraftConfig | None = None,
    ):
        self.limits = RateLimits.default() if limits is None else limits
        self.cfg = cfg
        self.phi_pid = PID(*gains.phi, limits=self.limits.p, name="phi_attitude")
        self.theta_pid = PID(
            *gains.theta,
            limits=self.limits.q,
            name="theta_attitude",
        )

    def update(
        self,
        ref_attitude: np.ndarray,
        euler_meas: np.ndarray,
        V: float,
        dt: float,
    ) -> np.ndarray:
        """Return ``[p_cmd, q_cmd, r_cmd]`` for the rate loop."""
        ref = np.asarray(ref_attitude, dtype=np.float64)
        euler = np.asarray(euler_meas, dtype=np.float64)
        if ref.shape != (2,) or euler.shape != (3,):
            raise ValueError("ref_attitude must be (2,) and euler_meas must be (3,)")
        phi_cmd = self._clamp_phi(ref[0])
        theta_cmd = self._clamp_theta(ref[1])
        p_cmd = self.phi_pid.update(phi_cmd, euler[0], dt)
        q_cmd = self.theta_pid.update(theta_cmd, euler[1], dt)
        r_cmd = np.clip(
            coordinated_turn_rate(phi_cmd, max(V, 1.0), euler[1]),
            self.limits.r[0],
            self.limits.r[1],
        )
        return np.array([p_cmd, q_cmd, r_cmd], dtype=np.float64)

    def reset(self) -> None:
        self.phi_pid.reset()
        self.theta_pid.reset()

    def _clamp_phi(self, value: float) -> float:
        if self.cfg is None:
            return float(value)
        return float(
            np.clip(
                value, -self.cfg.envelope.phi_max_rad, self.cfg.envelope.phi_max_rad
            )
        )

    def _clamp_theta(self, value: float) -> float:
        return float(np.clip(value, np.deg2rad(-20.0), np.deg2rad(25.0)))

    @property
    def gains(self) -> np.ndarray:
        return np.array(
            [
                self.phi_pid.kp,
                self.phi_pid.ki,
                self.phi_pid.kd,
                self.theta_pid.kp,
                self.theta_pid.ki,
                self.theta_pid.kd,
            ],
            dtype=np.float64,
        )

    @gains.setter
    def gains(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (6,):
            raise ValueError(
                f"AttitudeLoop gains must have shape (6,), got {arr.shape}"
            )
        self.phi_pid.kp, self.phi_pid.ki, self.phi_pid.kd = arr[0:3]
        self.theta_pid.kp, self.theta_pid.ki, self.theta_pid.kd = arr[3:6]

    @property
    def gain_names(self) -> list[str]:
        return [
            "att_phi_kp",
            "att_phi_ki",
            "att_phi_kd",
            "att_theta_kp",
            "att_theta_ki",
            "att_theta_kd",
        ]

    @property
    def gain_bounds(self) -> list[tuple[float, float]]:
        return [
            (0.0, 12.0),
            (0.0, 3.0),
            (0.0, 0.5),
            (0.0, 12.0),
            (0.0, 3.0),
            (0.0, 0.5),
        ]


def coordinated_turn_rate(phi: float, V: float, theta: float = 0.0) -> float:
    """Yaw rate for a coordinated banked turn."""
    return float(G0 * np.tan(phi) * np.cos(theta) / max(V, 1.0))
