"""Composed fixed-wing aircraft plant."""

from __future__ import annotations

from typing import Any

import numpy as np

from uav_sim.aero import AeroModel, LinearAeroModel
from uav_sim.air_data import compute_air_data
from uav_sim.config import AircraftConfig
from uav_sim.dynamics import rigid_body_derivatives
from uav_sim.forces import aero_forces_moments, total_forces_moments
from uav_sim.propulsion import propulsion_forces_moments
from uav_sim.rotations import quat_to_dcm
from uav_sim.state import IDX_OMEGA, IDX_QUAT, IDX_VEL, State


class AircraftPlant:
    """Compose dynamics, aerodynamics, propulsion, gravity, and wind."""

    def __init__(
        self,
        cfg: AircraftConfig,
        aero_model: AeroModel | None = None,
        wind_model: Any | None = None,
        actuators: Any | None = None,
        sensors: Any | None = None,
        estimator: Any | None = None,
    ):
        self.cfg = cfg
        self.aero_model = aero_model or LinearAeroModel(cfg.aero)
        self.wind_model = wind_model
        self.actuators = actuators
        self.sensors = sensors
        self.estimator = estimator
        self._wind_cache = np.zeros(3, dtype=np.float64)
        self._actual_cache = np.zeros(4, dtype=np.float64)
        self._last_measurements: Any | None = None
        self._estimator_started = False
        self._actuator_primed = False
        self._last_imu_time = 0.0

    def derivatives(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Full state derivative for the integrator."""
        controls = self._validate_controls(u)
        wind_n = self._wind_cache
        # Build the body->NED matrix once and reuse it in the force assembly and
        # the rigid-body derivative (it was previously rebuilt ~3x per call).
        dcm = quat_to_dcm(x[IDX_QUAT])
        F_b, M_b, _air = total_forces_moments(
            x,
            controls,
            self.cfg,
            self.aero_model,
            wind_n=wind_n,
            dcm=dcm,
        )
        return rigid_body_derivatives(x, F_b, M_b, self.cfg, dcm=dcm)

    def diagnostics(self, t: float, x: np.ndarray, u: np.ndarray) -> dict[str, Any]:
        """Return force, moment, and air-data diagnostics for logging/debugging."""
        controls = self._validate_controls(u)
        wind_n = self._wind_cache
        state = State.from_array(x)
        F_total, M_total, air = total_forces_moments(
            x,
            controls,
            self.cfg,
            self.aero_model,
            wind_n=wind_n,
        )
        coeffs = self.aero_model.coefficients(
            air,
            x[IDX_OMEGA],
            controls,
            self.cfg.geometry,
        )
        F_aero, M_aero = aero_forces_moments(air, coeffs, self.cfg.geometry)
        F_prop, M_prop = propulsion_forces_moments(
            controls[3],
            air,
            self.cfg.propulsion,
        )
        return {
            "altitude": state.altitude,
            "V": air.V,
            "alpha": air.alpha,
            "beta": air.beta,
            "q_bar": air.q_bar,
            "wind_n": wind_n.copy(),
            "F_total_b": F_total,
            "M_total_b": M_total,
            "F_aero_b": F_aero,
            "M_aero_b": M_aero,
            "F_prop_b": F_prop,
            "M_prop_b": M_prop,
            "coeffs": coeffs,
        }

    def reset(self, rng: np.random.Generator, x0: np.ndarray | None = None) -> None:
        """Reset every stateful realism component from one parent stream."""
        streams = rng.spawn(2)
        self._wind_cache = np.zeros(3, dtype=np.float64)
        self._actual_cache = np.zeros(4, dtype=np.float64)
        self._last_measurements = None
        self._estimator_started = False
        self._actuator_primed = False
        self._last_imu_time = 0.0
        if self.actuators is not None:
            self.actuators.reset()
        if self.wind_model is not None and hasattr(self.wind_model, "reset"):
            self.wind_model.reset(streams[0])
        if self.sensors is not None:
            self.sensors.reset(streams[1])
        if self.estimator is not None:
            if x0 is None:
                raise ValueError("x0 is required when resetting an estimator")
            self.estimator.reset(x0)

    def feedback_state(self, t: float, x_true: np.ndarray, dt: float) -> np.ndarray:
        """Return truth or the sensor/EKF estimate presented to the controller."""
        if self.sensors is None or self.estimator is None:
            feedback = np.asarray(x_true, dtype=np.float64).copy()
            if self.wind_model is not None:
                state = State.from_array(x_true)
                feedback[IDX_VEL] = compute_air_data(
                    state.vel_b, state.quat, state.altitude, self._wind_cache
                ).v_air_b
            return feedback
        if self._last_measurements is not None and not self.sensors.sample_due(t):
            return self.estimator.state
        xdot = self.derivatives(t, x_true, self._actual_cache)
        state = State.from_array(x_true)
        air = compute_air_data(
            state.vel_b, state.quat, state.altitude, self._wind_cache
        )
        measurements = self.sensors.measure(x_true, xdot, t, air.V)
        self._last_measurements = measurements
        estimator_was_started = self._estimator_started
        if estimator_was_started and "imu" in measurements.updated:
            imu_dt = max(float(t) - self._last_imu_time, dt)
            self.estimator.predict(measurements.imu, imu_dt)
            self._last_imu_time = float(t)
        elif not estimator_was_started:
            self._estimator_started = True
            self._last_imu_time = float(t)
        if estimator_was_started and "imu" in measurements.updated:
            self.estimator.update_accelerometer(measurements.imu)
        if "gps" in measurements.updated:
            self.estimator.update_gps(measurements.gps)
        if "barometer" in measurements.updated:
            self.estimator.update_baro(measurements.barometric_altitude)
        if "pitot" in measurements.updated:
            self.estimator.update_airspeed(measurements.pitot_airspeed)
        if "magnetometer" in measurements.updated:
            self.estimator.update_magnetometer(measurements.magnetic_heading)
        return self.estimator.state

    def prepare_step(
        self, t: float, x: np.ndarray, command: np.ndarray, dt: float
    ) -> np.ndarray:
        """Advance stateful inputs once before the four pure RK4 stages."""
        controls = self._validate_controls(command)
        if self.actuators is None:
            self._actual_cache = controls.copy()
        elif not self._actuator_primed:
            self.actuators.reset(controls)
            self._actual_cache = np.asarray(
                self.actuators.step(controls, dt), dtype=np.float64
            )
            self._actuator_primed = True
        else:
            self._actual_cache = np.asarray(
                self.actuators.step(controls, dt), dtype=np.float64
            )
        if self.wind_model is not None:
            state = State.from_array(x)
            previous_air = compute_air_data(
                state.vel_b, state.quat, state.altitude, self._wind_cache
            )
            if hasattr(self.wind_model, "velocity_n"):
                self._wind_cache = np.asarray(
                    self.wind_model.velocity_n(
                        t, state.pos_n, previous_air.V, dt, state.quat
                    ),
                    dtype=np.float64,
                )
            else:
                self._wind_cache = np.asarray(
                    self.wind_model(t, state.pos_n), dtype=np.float64
                )
        return self._actual_cache.copy()

    def adjust_initial_state(self, x: np.ndarray) -> np.ndarray:
        """Convert an air-relative trim state to ground-relative velocity."""
        adjusted = np.asarray(x, dtype=np.float64).copy()
        steady = getattr(self.wind_model, "steady", None)
        if steady is not None:
            state = State.from_array(adjusted)
            wind_n = steady.velocity(state.altitude)
            adjusted[IDX_VEL] += quat_to_dcm(state.quat).T @ wind_n
        return adjusted

    def sample_diagnostics(
        self, t: float, x: np.ndarray, actual: np.ndarray
    ) -> dict[str, Any]:
        """Compact per-log-sample diagnostics used by metrics and studies."""
        diagnostics = self.diagnostics(t, x, actual)
        sample: dict[str, Any] = {
            "V": diagnostics["V"],
            "alpha": diagnostics["alpha"],
            "beta": diagnostics["beta"],
            "wind_n": diagnostics["wind_n"],
        }
        if self.actuators is not None:
            sample["actuator_saturated"] = self.actuators.is_saturated
            sample["actuator_rate_limited"] = self.actuators.is_rate_limited
        if self.estimator is not None:
            sample["x_hat"] = self.estimator.state
            sample["covariance_diagonal"] = self.estimator.covariance_diagonal
        return sample

    def __call__(
        self,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        _cfg: AircraftConfig | None = None,
        _rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Adapter for ``integrate.simulate``."""
        return self.derivatives(t, x, u)

    @staticmethod
    def _validate_controls(u: np.ndarray) -> np.ndarray:
        controls = np.asarray(u, dtype=np.float64)
        if controls.shape != (4,):
            raise ValueError(f"controls must have shape (4,), got {controls.shape}")
        return controls
