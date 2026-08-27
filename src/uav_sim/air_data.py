"""Air-relative velocity and air-data calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_sim.environment.atmosphere import isa
from uav_sim.rotations import quat_to_dcm

V_MIN_AERO = 0.1


@dataclass(frozen=True)
class AirData:
    """Air-relative quantities used by aerodynamics and propulsion."""

    V: float
    alpha: float
    beta: float
    q_bar: float
    v_air_b: np.ndarray
    rho: float


def air_relative_velocity(
    vel_b: np.ndarray,
    quat: np.ndarray,
    wind_n: np.ndarray,
    dcm: np.ndarray | None = None,
) -> np.ndarray:
    """Velocity of the aircraft relative to the air mass, in body axes.

    ``wind_n`` is the velocity of the air mass in NED. A positive north wind
    vector means the air mass moves north. ``dcm`` optionally supplies a
    precomputed ``R_nb = quat_to_dcm(quat)`` to reuse (bit-identical).
    """
    velocity_b = np.asarray(vel_b, dtype=np.float64)
    wind_vec_n = np.asarray(wind_n, dtype=np.float64)
    if velocity_b.shape != (3,):
        raise ValueError(f"vel_b must have shape (3,), got {velocity_b.shape}")
    if wind_vec_n.shape != (3,):
        raise ValueError(f"wind_n must have shape (3,), got {wind_vec_n.shape}")
    rotation = quat_to_dcm(quat) if dcm is None else dcm
    return velocity_b - rotation.T @ wind_vec_n


def compute_air_data(
    vel_b: np.ndarray,
    quat: np.ndarray,
    altitude: float,
    wind_n: np.ndarray | None = None,
    dcm: np.ndarray | None = None,
) -> AirData:
    """Compute true airspeed, alpha, beta, and dynamic pressure."""
    wind = np.zeros(3, dtype=np.float64) if wind_n is None else wind_n
    v_air_b = air_relative_velocity(vel_b, quat, wind, dcm=dcm)
    V = float(np.linalg.norm(v_air_b))
    atmosphere = isa(altitude)
    if V < V_MIN_AERO:
        return AirData(
            V=V,
            alpha=0.0,
            beta=0.0,
            q_bar=0.0,
            v_air_b=v_air_b,
            rho=atmosphere.rho,
        )

    alpha = float(np.arctan2(v_air_b[2], v_air_b[0]))
    beta_arg = np.clip(v_air_b[1] / V, -1.0, 1.0)
    beta = float(np.arcsin(beta_arg))
    q_bar = 0.5 * atmosphere.rho * V**2
    return AirData(
        V=V,
        alpha=alpha,
        beta=beta,
        q_bar=float(q_bar),
        v_air_b=v_air_b,
        rho=atmosphere.rho,
    )
