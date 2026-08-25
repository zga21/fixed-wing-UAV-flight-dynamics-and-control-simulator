"""Composed fixed-wing aircraft plant."""

from __future__ import annotations

from typing import Any

import numpy as np

from uav_sim.aero import AeroModel, LinearAeroModel
from uav_sim.config import AircraftConfig
from uav_sim.dynamics import rigid_body_derivatives
from uav_sim.forces import aero_forces_moments, total_forces_moments
from uav_sim.propulsion import propulsion_forces_moments
from uav_sim.state import IDX_OMEGA, State


class AircraftPlant:
    """Compose dynamics, aerodynamics, propulsion, gravity, and wind."""

    def __init__(
        self,
        cfg: AircraftConfig,
        aero_model: AeroModel | None = None,
        wind_model: Any | None = None,
        actuators: Any | None = None,
    ):
        self.cfg = cfg
        self.aero_model = aero_model or LinearAeroModel(cfg.aero)
        self.wind_model = wind_model
        self.actuators = actuators

    def derivatives(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Full state derivative for the integrator."""
        controls = self._actual_controls(t, u)
        wind_n = self._wind_n(t, x)
        F_b, M_b, _air = total_forces_moments(
            x,
            controls,
            self.cfg,
            self.aero_model,
            wind_n=wind_n,
        )
        return rigid_body_derivatives(x, F_b, M_b, self.cfg)

    def diagnostics(self, t: float, x: np.ndarray, u: np.ndarray) -> dict[str, Any]:
        """Return force, moment, and air-data diagnostics for logging/debugging."""
        controls = self._actual_controls(t, u)
        wind_n = self._wind_n(t, x)
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

    def _wind_n(self, t: float, x: np.ndarray) -> np.ndarray:
        if self.wind_model is None:
            return np.zeros(3, dtype=np.float64)
        state = State.from_array(x)
        return np.asarray(self.wind_model(t, state.pos_n), dtype=np.float64)

    def _actual_controls(self, t: float, u: np.ndarray) -> np.ndarray:
        controls = np.asarray(u, dtype=np.float64)
        if controls.shape != (4,):
            raise ValueError(f"controls must have shape (4,), got {controls.shape}")
        if self.actuators is None:
            return controls
        return np.asarray(self.actuators.update(t, controls), dtype=np.float64)
