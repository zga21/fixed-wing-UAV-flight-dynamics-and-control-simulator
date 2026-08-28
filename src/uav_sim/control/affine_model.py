"""Controller-owned control-affine angular dynamics model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_sim.air_data import V_MIN_AERO, AirData
from uav_sim.config import AeroDerivatives, AircraftConfig
from uav_sim.state import IDX_OMEGA, N_STATES


@dataclass(frozen=True)
class EffectivenessDiagnostics:
    """Numerical health of the angular control-effectiveness matrix."""

    condition_number: float
    minimum_singular_value: float
    q_bar: float


class ControlAffineModel:
    """Internal NDI model, deliberately separate from the simulated plant."""

    def __init__(
        self,
        cfg: AircraftConfig,
        aero_derivs: AeroDerivatives | None = None,
    ):
        self.cfg = cfg
        self.aero_derivs = cfg.aero if aero_derivs is None else aero_derivs

    def f(
        self,
        x_hat: np.ndarray,
        air: AirData,
        static_scale: float = 1.0,
    ) -> np.ndarray:
        """Return control-independent body angular acceleration, rad/s^2."""
        x = np.asarray(x_hat, dtype=np.float64)
        if x.shape != (N_STATES,):
            raise ValueError(f"x_hat must have shape ({N_STATES},), got {x.shape}")
        if not 0.0 <= static_scale <= 1.0:
            raise ValueError("static_scale must be between zero and one")

        omega = x[IDX_OMEGA]
        p, q_rate, r = omega
        geom = self.cfg.geometry
        derivs = self.aero_derivs
        speed = max(air.V, V_MIN_AERO)
        p_hat = p * geom.b / (2.0 * speed)
        q_hat = q_rate * geom.c_bar / (2.0 * speed)
        r_hat = r * geom.b / (2.0 * speed)

        coefficients = np.array(
            [
                static_scale * derivs.C_l_beta * air.beta
                + derivs.C_l_p * p_hat
                + derivs.C_l_r * r_hat,
                static_scale * (derivs.C_m_0 + derivs.C_m_alpha * air.alpha)
                + derivs.C_m_q * q_hat,
                static_scale * derivs.C_n_beta * air.beta
                + derivs.C_n_p * p_hat
                + derivs.C_n_r * r_hat,
            ],
            dtype=np.float64,
        )
        moments = (
            air.q_bar
            * geom.S
            * np.array(
                [
                    geom.b * coefficients[0],
                    geom.c_bar * coefficients[1],
                    geom.b * coefficients[2],
                ],
                dtype=np.float64,
            )
        )
        inertia = self.cfg.mass.inertia_tensor
        gyroscopic = np.cross(omega, inertia @ omega)
        return self.cfg.mass.inertia_inverse @ (moments - gyroscopic)

    def g(self, air: AirData, q_bar: float | None = None) -> np.ndarray:
        """Return surface-to-angular-acceleration effectiveness, shape ``(3, 3)``."""
        dynamic_pressure = air.q_bar if q_bar is None else float(q_bar)
        geom = self.cfg.geometry
        derivs = self.aero_derivs
        coefficient_matrix = np.array(
            [
                [geom.b * derivs.C_l_delta_a, 0.0, geom.b * derivs.C_l_delta_r],
                [0.0, geom.c_bar * derivs.C_m_delta_e, 0.0],
                [geom.b * derivs.C_n_delta_a, 0.0, geom.b * derivs.C_n_delta_r],
            ],
            dtype=np.float64,
        )
        return self.cfg.mass.inertia_inverse @ (
            dynamic_pressure * geom.S * coefficient_matrix
        )

    def diagnostics(self, air: AirData) -> EffectivenessDiagnostics:
        """Return condition number and absolute control authority."""
        matrix = self.g(air)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        minimum = float(singular_values[-1])
        condition = float("inf") if minimum == 0.0 else float(np.linalg.cond(matrix))
        return EffectivenessDiagnostics(condition, minimum, float(air.q_bar))

    def perturb(self, **factors: float) -> ControlAffineModel:
        """Return an independent internal model with selected fields scaled."""
        aero_updates: dict[str, float] = {}
        mass_updates: dict[str, float] = {}
        for name, factor in factors.items():
            scale = float(factor)
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"perturbation factor for {name} must be positive")
            if hasattr(self.aero_derivs, name):
                aero_updates[name] = float(getattr(self.aero_derivs, name)) * scale
            elif hasattr(self.cfg.mass, name):
                mass_updates[name] = float(getattr(self.cfg.mass, name)) * scale
            elif name == "mass":
                mass_updates["m"] = self.cfg.mass.m * scale
            else:
                raise KeyError(f"unknown affine-model parameter: {name}")

        derivatives = self.aero_derivs.model_copy(update=aero_updates)
        mass = self.cfg.mass.model_copy(update=mass_updates)
        config = self.cfg.model_copy(update={"mass": mass, "aero": derivatives})
        return ControlAffineModel(config, derivatives)
