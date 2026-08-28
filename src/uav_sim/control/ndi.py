"""Guarded nonlinear dynamic-inversion autopilot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_sim.air_data import AirData
from uav_sim.config import AircraftConfig
from uav_sim.control.affine_model import ControlAffineModel
from uav_sim.control.allocation import AllocationResult, redistributed_allocation
from uav_sim.control.baseline import CascadedPIDAutopilot, LoopRates
from uav_sim.state import IDX_OMEGA


@dataclass(frozen=True)
class NDIDiagnostics:
    """Latest inversion-health sample."""

    condition_number: float
    minimum_singular_value: float
    q_bar_clamped: bool
    damped_inverse: bool
    low_speed_blend: float
    allocation_residual_norm: float
    saturated: np.ndarray


class NDIController(CascadedPIDAutopilot):
    """Phase 4 outer loops with a model-based angular-rate inner loop."""

    def __init__(
        self,
        model: ControlAffineModel,
        K_omega: np.ndarray | tuple[float, float, float] = (15.0, 15.0, 12.0),
        K_I: np.ndarray | tuple[float, float, float] = (1.0, 1.0, 0.5),
        q_bar_min: float = 5.0,
        cond_max: float = 1.0e4,
        blend_V: float = 14.0,
        blend_width: float = 2.0,
        anti_windup_gain: float = 2.0,
        nu_limits: np.ndarray | tuple[float, float, float] = (4.0, 2.0, 3.0),
        static_cancellation: float = 1.0,
        ndi_authority: float = 1.0,
        ndi_filter_tau: float = 0.0,
        pid_retention: float | None = None,
        rates: LoopRates | None = None,
        cfg: AircraftConfig | None = None,
    ):
        self.model = model
        self.K_omega = self._gain_vector(K_omega, "K_omega", strictly_positive=True)
        self.K_I = self._gain_vector(K_I, "K_I", strictly_positive=False)
        if q_bar_min <= 0.0 or cond_max <= 1.0 or blend_V <= 0.0:
            raise ValueError("NDI guard thresholds must be positive")
        if blend_width <= 0.0 or anti_windup_gain < 0.0:
            raise ValueError(
                "blend_width must be positive and anti-windup non-negative"
            )
        self.q_bar_min = float(q_bar_min)
        self.cond_max = float(cond_max)
        self.blend_V = float(blend_V)
        self.blend_width = float(blend_width)
        self.anti_windup_gain = float(anti_windup_gain)
        self.nu_limits = self._gain_vector(
            nu_limits, "nu_limits", strictly_positive=True
        )
        if not 0.0 <= static_cancellation <= 1.0:
            raise ValueError("static_cancellation must be between zero and one")
        if not 0.0 <= ndi_authority <= 1.0:
            raise ValueError("ndi_authority must be between zero and one")
        if ndi_filter_tau < 0.0:
            raise ValueError("ndi_filter_tau must be non-negative")
        if pid_retention is not None and not 0.0 <= pid_retention <= 1.0:
            raise ValueError("pid_retention must be between zero and one")
        self.static_cancellation = float(static_cancellation)
        self.ndi_authority = float(ndi_authority)
        self.ndi_filter_tau = float(ndi_filter_tau)
        self.pid_retention = (
            1.0 - self.ndi_authority if pid_retention is None else float(pid_retention)
        )
        super().__init__(model.cfg if cfg is None else cfg, rates=rates)

    def reset(self) -> None:
        super().reset()
        self._rate_error_integral = np.zeros(3, dtype=np.float64)
        self._last_ndi_diagnostics = NDIDiagnostics(
            condition_number=float("inf"),
            minimum_singular_value=0.0,
            q_bar_clamped=False,
            damped_inverse=False,
            low_speed_blend=0.0,
            allocation_residual_norm=0.0,
            saturated=np.zeros(3, dtype=bool),
        )
        self._diagnostic_history: list[NDIDiagnostics] = []
        self._filtered_ndi_increment = np.zeros(3, dtype=np.float64)

    def _virtual_control(
        self,
        omega_cmd: np.ndarray,
        omega_hat: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Return desired angular acceleration and update the error integral."""
        command = np.asarray(omega_cmd, dtype=np.float64)
        measured = np.asarray(omega_hat, dtype=np.float64)
        if command.shape != (3,) or measured.shape != (3,):
            raise ValueError("omega_cmd and omega_hat must both have shape (3,)")
        error = command - measured
        self._rate_error_integral += error * dt
        integral_limit = np.divide(
            10.0,
            np.maximum(self.K_I, 1.0e-9),
        )
        self._rate_error_integral = np.clip(
            self._rate_error_integral, -integral_limit, integral_limit
        )
        unconstrained = self.K_omega * error + self.K_I * self._rate_error_integral
        constrained = np.clip(unconstrained, -self.nu_limits, self.nu_limits)
        correction = unconstrained - constrained
        self._rate_error_integral -= (
            self.anti_windup_gain * correction * dt / np.maximum(self.K_I, 1.0e-6)
        )
        return constrained

    def _invert(
        self,
        nu: np.ndarray,
        x_hat: np.ndarray,
        air: AirData,
    ) -> AllocationResult:
        """Invert guarded angular dynamics and allocate physical surfaces."""
        q_bar = max(float(air.q_bar), self.q_bar_min)
        matrix = self.model.g(air, q_bar=q_bar)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        minimum = float(singular_values[-1])
        condition = float("inf") if minimum == 0.0 else float(np.linalg.cond(matrix))
        trim_surfaces = self._last_trim_controls[:3]
        target = (
            np.asarray(nu, dtype=np.float64)
            - self.model.f(x_hat, air, static_scale=self.static_cancellation)
            - matrix @ trim_surfaces
        )
        damped = condition > self.cond_max or not np.isfinite(condition)

        limits = self._surface_limits() - trim_surfaces[:, None]
        if damped:
            damping = max(float(singular_values[0]) / self.cond_max, 1.0e-8)
            delta = matrix.T @ np.linalg.solve(
                matrix @ matrix.T + damping**2 * np.eye(3), target
            )
            if np.all(delta >= limits[:, 0]) and np.all(delta <= limits[:, 1]):
                incremental_result = AllocationResult(
                    delta=delta,
                    residual=target - matrix @ delta,
                    saturated=np.zeros(3, dtype=bool),
                    iterations=1,
                )
            else:
                incremental_result = redistributed_allocation(matrix, target, limits)
        else:
            incremental_result = redistributed_allocation(matrix, target, limits)

        result = AllocationResult(
            delta=trim_surfaces + incremental_result.delta,
            residual=incremental_result.residual,
            saturated=incremental_result.saturated,
            iterations=incremental_result.iterations,
        )

        blend = self._ndi_blend(air.V)
        self._last_ndi_diagnostics = NDIDiagnostics(
            condition_number=condition,
            minimum_singular_value=minimum,
            q_bar_clamped=air.q_bar < self.q_bar_min,
            damped_inverse=damped,
            low_speed_blend=blend,
            allocation_residual_norm=float(np.linalg.norm(result.residual)),
            saturated=result.saturated.copy(),
        )
        return result

    def _rate_surfaces(
        self,
        omega_cmd: np.ndarray,
        x_hat: np.ndarray,
        air: AirData,
        dt: float,
    ) -> np.ndarray:
        omega_hat = np.asarray(x_hat[IDX_OMEGA], dtype=np.float64)
        nu = self._virtual_control(omega_cmd, omega_hat, dt)
        allocation = self._invert(nu, x_hat, air)
        low_speed_blend = self._last_ndi_diagnostics.low_speed_blend
        ndi_weight = self.ndi_authority * low_speed_blend
        pid_weight = (1.0 - low_speed_blend) + low_speed_blend * self.pid_retention
        ndi_increment = allocation.delta - self._last_trim_controls[:3]
        if self.ndi_filter_tau > 0.0:
            fraction = dt / (self.ndi_filter_tau + dt)
            self._filtered_ndi_increment += fraction * (
                ndi_increment - self._filtered_ndi_increment
            )
            ndi_increment = self._filtered_ndi_increment.copy()
        pid_increment = self.rate_loop.update(omega_cmd, omega_hat, dt)

        if np.any(np.abs(allocation.residual) > 1.0e-10):
            self._rate_error_integral -= (
                self.anti_windup_gain
                * allocation.residual
                * dt
                / np.maximum(self.K_I, 1.0e-6)
            )
        self._diagnostic_history.append(self._last_ndi_diagnostics)
        return ndi_weight * ndi_increment + pid_weight * pid_increment

    def _ndi_blend(self, airspeed: float) -> float:
        low = self.blend_V - self.blend_width
        fraction = float(np.clip((airspeed - low) / self.blend_width, 0.0, 1.0))
        return fraction * fraction * (3.0 - 2.0 * fraction)

    def _surface_limits(self) -> np.ndarray:
        return np.array(
            [
                [
                    self.cfg.actuators.aileron.min_rad,
                    self.cfg.actuators.aileron.max_rad,
                ],
                [
                    self.cfg.actuators.elevator.min_rad,
                    self.cfg.actuators.elevator.max_rad,
                ],
                [self.cfg.actuators.rudder.min_rad, self.cfg.actuators.rudder.max_rad],
            ],
            dtype=np.float64,
        )

    @property
    def diagnostics(self) -> NDIDiagnostics:
        return self._last_ndi_diagnostics

    @property
    def diagnostic_history(self) -> tuple[NDIDiagnostics, ...]:
        return tuple(self._diagnostic_history)

    @property
    def gains(self) -> np.ndarray:
        return np.concatenate(
            [
                self.K_omega,
                self.K_I,
                self.attitude_loop.gains,
                self.tecs.gains,
                self.heading_loop.gains,
            ]
        )

    @gains.setter
    def gains(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (21,):
            raise ValueError(f"NDI gains must have shape (21,), got {arr.shape}")
        if np.any(arr[:3] <= 0.0) or np.any(arr[3:6] < 0.0):
            raise ValueError("K_omega must be positive and K_I non-negative")
        self.K_omega = arr[:3].copy()
        self.K_I = arr[3:6].copy()
        self.attitude_loop.gains = arr[6:12]
        self.tecs.gains = arr[12:18]
        self.heading_loop.gains = arr[18:21]

    @property
    def gain_names(self) -> list[str]:
        return [
            "ndi_K_omega_p",
            "ndi_K_omega_q",
            "ndi_K_omega_r",
            "ndi_K_I_p",
            "ndi_K_I_q",
            "ndi_K_I_r",
            *self.attitude_loop.gain_names,
            *self.tecs.gain_names,
            *self.heading_loop.gain_names,
        ]

    @property
    def gain_bounds(self) -> list[tuple[float, float]]:
        return (
            [(0.1, 20.0)] * 3
            + [(0.0, 10.0)] * 3
            + self.attitude_loop.gain_bounds
            + self.tecs.gain_bounds
            + self.heading_loop.gain_bounds
        )

    @staticmethod
    def _gain_vector(
        values: np.ndarray | tuple[float, float, float],
        name: str,
        strictly_positive: bool,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (3,):
            raise ValueError(f"{name} must have shape (3,)")
        invalid = array <= 0.0 if strictly_positive else array < 0.0
        if np.any(invalid):
            qualifier = "positive" if strictly_positive else "non-negative"
            raise ValueError(f"{name} must be {qualifier}")
        return array.copy()
