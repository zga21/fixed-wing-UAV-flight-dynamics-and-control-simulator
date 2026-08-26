"""Baseline cascaded PID autopilot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_sim.air_data import compute_air_data
from uav_sim.config import AircraftConfig
from uav_sim.control.attitude_loop import AttitudeLoop
from uav_sim.control.base import Controls, Reference
from uav_sim.control.outer_loop import HeadingLoop, TECSLite
from uav_sim.control.rate_loop import ActuatorLimits, RateLoop
from uav_sim.rotations import quat_to_euler
from uav_sim.state import IDX_OMEGA, IDX_QUAT, State
from uav_sim.trim import TrimNotConverged, trim


@dataclass(frozen=True)
class LoopRates:
    """Nominal cascaded loop update rates."""

    outer_hz: float = 10.0
    attitude_hz: float = 50.0
    rate_hz: float = 200.0


class CascadedPIDAutopilot:
    """TECS + heading + attitude + body-rate baseline controller."""

    def __init__(self, cfg: AircraftConfig, rates: LoopRates | None = None):
        self.cfg = cfg
        self.rates = LoopRates() if rates is None else rates
        self.rate_loop = RateLoop(limits=ActuatorLimits.from_config(cfg))
        self.attitude_loop = AttitudeLoop(cfg=cfg)
        self.tecs = TECSLite(cfg)
        self.heading_loop = HeadingLoop(cfg)
        self.reset()

    def reset(self) -> None:
        self.rate_loop.reset()
        self.attitude_loop.reset()
        self.tecs.reset()
        self.heading_loop.reset()
        self._next_outer = 0.0
        self._next_attitude = 0.0
        self._next_rate = 0.0
        self._last_attitude_cmd = np.zeros(2, dtype=np.float64)
        self._last_omega_cmd = np.zeros(3, dtype=np.float64)
        self._last_surfaces = np.zeros(3, dtype=np.float64)
        self._last_throttle = 0.0
        self._last_trim_controls = np.zeros(4, dtype=np.float64)
        self._trim_cache: dict[tuple[float, float], tuple[float, np.ndarray]] = {}
        self.loop_counts = {"outer": 0, "attitude": 0, "rate": 0}

    def update(
        self,
        x_hat: np.ndarray,
        ref: Reference,
        t: float,
        dt: float,
    ) -> Controls:
        state = State.from_array(x_hat)
        euler = quat_to_euler(x_hat[IDX_QUAT])
        air = compute_air_data(state.vel_b, state.quat, state.altitude)
        h_cmd = state.altitude if ref.altitude is None else ref.altitude
        V_cmd = air.V if ref.airspeed is None else ref.airspeed
        psi_cmd = euler[2] if ref.heading is None else ref.heading

        if t + 1e-12 >= self._next_outer:
            theta_trim, trim_controls = self._trim_feedforward(V_cmd, h_cmd)
            theta_cmd, throttle = self.tecs.update(
                h_cmd,
                V_cmd,
                state.altitude,
                air.V,
                theta_trim,
                trim_controls[3],
                1.0 / self.rates.outer_hz,
            )
            phi_cmd = (
                self.heading_loop.update(
                    psi_cmd, euler[2], max(air.V, 1.0), 1.0 / self.rates.outer_hz
                )
                if ref.roll is None
                else ref.roll
            )
            if ref.pitch is not None:
                theta_cmd = ref.pitch
            self._last_attitude_cmd = np.array([phi_cmd, theta_cmd], dtype=np.float64)
            self._last_throttle = throttle
            self._last_trim_controls = trim_controls
            self._next_outer += 1.0 / self.rates.outer_hz
            self.loop_counts["outer"] += 1

        if t + 1e-12 >= self._next_attitude:
            self._last_omega_cmd = self.attitude_loop.update(
                self._last_attitude_cmd,
                euler,
                max(air.V, 1.0),
                1.0 / self.rates.attitude_hz,
            )
            self._next_attitude += 1.0 / self.rates.attitude_hz
            self.loop_counts["attitude"] += 1

        if t + 1e-12 >= self._next_rate:
            self._last_surfaces = self.rate_loop.update(
                self._last_omega_cmd,
                x_hat[IDX_OMEGA],
                1.0 / self.rates.rate_hz,
            )
            self._next_rate += 1.0 / self.rates.rate_hz
            self.loop_counts["rate"] += 1

        controls = self._last_trim_controls.copy()
        controls[0:3] += self._last_surfaces
        controls[3] = self._last_throttle
        controls = self._saturate_controls(controls)
        return Controls(
            float(controls[0]),
            float(controls[1]),
            float(controls[2]),
            float(controls[3]),
        )

    def __call__(
        self, t: float, x_hat: np.ndarray, cfg: AircraftConfig, _rng
    ) -> np.ndarray:
        return self.update(x_hat, Reference(), t, cfg.integration.dt).to_array()

    def _trim_feedforward(self, V_cmd: float, h_cmd: float) -> tuple[float, np.ndarray]:
        speed = float(np.clip(V_cmd, self.cfg.envelope.V_min, self.cfg.envelope.V_max))
        altitude = float(
            np.clip(h_cmd, self.cfg.envelope.h_min, self.cfg.envelope.h_max)
        )
        key = (round(speed, 1), round(altitude, 0))
        cached = self._trim_cache.get(key)
        if cached is not None:
            return cached
        try:
            from uav_sim.plant import AircraftPlant

            point = trim(speed, 0.0, altitude, self.cfg, AircraftPlant(self.cfg))
            value = (point.theta, point.u.copy())
        except TrimNotConverged:
            value = (0.0, np.array([0.0, -0.1, 0.0, 0.35], dtype=np.float64))
        self._trim_cache[key] = value
        return value

    def _saturate_controls(self, controls: np.ndarray) -> np.ndarray:
        out = controls.copy()
        out[0] = np.clip(
            out[0],
            self.cfg.actuators.aileron.min_rad,
            self.cfg.actuators.aileron.max_rad,
        )
        out[1] = np.clip(
            out[1],
            self.cfg.actuators.elevator.min_rad,
            self.cfg.actuators.elevator.max_rad,
        )
        out[2] = np.clip(
            out[2], self.cfg.actuators.rudder.min_rad, self.cfg.actuators.rudder.max_rad
        )
        out[3] = np.clip(
            out[3], self.cfg.actuators.throttle.min, self.cfg.actuators.throttle.max
        )
        return out

    @property
    def gains(self) -> np.ndarray:
        return np.concatenate(
            [
                self.rate_loop.gains,
                self.attitude_loop.gains,
                self.tecs.gains,
                self.heading_loop.gains,
            ]
        )

    @gains.setter
    def gains(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (24,):
            raise ValueError(f"baseline gains must have shape (24,), got {arr.shape}")
        self.rate_loop.gains = arr[0:9]
        self.attitude_loop.gains = arr[9:15]
        self.tecs.gains = arr[15:21]
        self.heading_loop.gains = arr[21:24]

    @property
    def gain_names(self) -> list[str]:
        return (
            self.rate_loop.gain_names
            + self.attitude_loop.gain_names
            + self.tecs.gain_names
            + self.heading_loop.gain_names
        )

    @property
    def gain_bounds(self) -> list[tuple[float, float]]:
        return (
            self.rate_loop.gain_bounds
            + self.attitude_loop.gain_bounds
            + self.tecs.gain_bounds
            + self.heading_loop.gain_bounds
        )
