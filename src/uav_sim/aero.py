"""Aerodynamic coefficient models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from uav_sim.air_data import V_MIN_AERO, AirData
from uav_sim.config import AeroDerivatives, Geometry


@dataclass(frozen=True)
class AeroCoeffs:
    """Dimensionless aerodynamic force and moment coefficients."""

    C_L: float
    C_D: float
    C_Y: float
    C_l: float
    C_m: float
    C_n: float


@runtime_checkable
class AeroModel(Protocol):
    """Swap point for aerodynamic models."""

    def coefficients(
        self,
        air: AirData,
        omega_b: np.ndarray,
        controls: np.ndarray,
        geom: Geometry,
    ) -> AeroCoeffs:
        """Return aerodynamic coefficients."""


class LinearAeroModel:
    """Linear derivative build-up with polar drag."""

    def __init__(self, derivs: AeroDerivatives):
        self.derivs = derivs

    def coefficients(
        self,
        air: AirData,
        omega_b: np.ndarray,
        controls: np.ndarray,
        geom: Geometry,
    ) -> AeroCoeffs:
        omega = np.asarray(omega_b, dtype=np.float64)
        ctrl = np.asarray(controls, dtype=np.float64)
        if omega.shape != (3,):
            raise ValueError(f"omega_b must have shape (3,), got {omega.shape}")
        if ctrl.shape != (4,):
            raise ValueError(f"controls must have shape (4,), got {ctrl.shape}")

        delta_a, delta_e, delta_r, _delta_t = ctrl
        p, q_rate, r = omega
        V_denom = max(air.V, V_MIN_AERO)
        p_hat = p * geom.b / (2.0 * V_denom)
        q_hat = q_rate * geom.c_bar / (2.0 * V_denom)
        r_hat = r * geom.b / (2.0 * V_denom)
        d = self.derivs

        C_L = d.C_L_0 + d.C_L_alpha * air.alpha + d.C_L_q * q_hat
        C_L += d.C_L_delta_e * delta_e
        C_D = d.C_D_0 + C_L**2 / (np.pi * d.oswald_e * geom.aspect_ratio)
        C_Y = d.C_Y_beta * air.beta + d.C_Y_p * p_hat + d.C_Y_r * r_hat
        C_Y += d.C_Y_delta_a * delta_a + d.C_Y_delta_r * delta_r
        C_l = d.C_l_beta * air.beta + d.C_l_p * p_hat + d.C_l_r * r_hat
        C_l += d.C_l_delta_a * delta_a + d.C_l_delta_r * delta_r
        C_m = d.C_m_0 + d.C_m_alpha * air.alpha + d.C_m_q * q_hat
        C_m += d.C_m_delta_e * delta_e
        C_n = d.C_n_beta * air.beta + d.C_n_p * p_hat + d.C_n_r * r_hat
        C_n += d.C_n_delta_a * delta_a + d.C_n_delta_r * delta_r

        return AeroCoeffs(
            C_L=float(C_L),
            C_D=float(C_D),
            C_Y=float(C_Y),
            C_l=float(C_l),
            C_m=float(C_m),
            C_n=float(C_n),
        )
