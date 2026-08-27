"""Force and moment assembly."""

from __future__ import annotations

import numpy as np

from uav_sim.aero import AeroCoeffs, AeroModel
from uav_sim.air_data import AirData, compute_air_data
from uav_sim.config import AircraftConfig, Geometry
from uav_sim.dynamics import gravity_body
from uav_sim.propulsion import propulsion_forces_moments
from uav_sim.rotations import quat_to_dcm
from uav_sim.state import IDX_OMEGA, IDX_QUAT, State


def aero_forces_moments(
    air: AirData,
    coeffs: AeroCoeffs,
    geom: Geometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert aerodynamic coefficients into body-frame force and moment."""
    q_s = air.q_bar * geom.S
    cos_alpha = np.cos(air.alpha)
    sin_alpha = np.sin(air.alpha)
    wind_x = -coeffs.C_D
    wind_z = -coeffs.C_L
    F_x = q_s * (cos_alpha * wind_x - sin_alpha * wind_z)
    F_z = q_s * (sin_alpha * wind_x + cos_alpha * wind_z)
    F_y = q_s * coeffs.C_Y
    moments = (
        air.q_bar
        * geom.S
        * np.array(
            [
                geom.b * coeffs.C_l,
                geom.c_bar * coeffs.C_m,
                geom.b * coeffs.C_n,
            ],
            dtype=np.float64,
        )
    )
    return np.array([F_x, F_y, F_z], dtype=np.float64), moments


def total_forces_moments(
    x: np.ndarray,
    controls: np.ndarray,
    cfg: AircraftConfig,
    aero_model: AeroModel,
    wind_n: np.ndarray | None = None,
    dcm: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, AirData]:
    """Return total body-frame forces, moments, and air data.

    ``dcm`` optionally supplies a precomputed ``R_nb = quat_to_dcm(x_quat)`` so
    the attitude matrix is built once per derivative and reused by the air-data
    and gravity terms (bit-identical to recomputing it in each).
    """
    state = State.from_array(x)
    control = np.asarray(controls, dtype=np.float64)
    if control.shape != (4,):
        raise ValueError(f"controls must have shape (4,), got {control.shape}")
    rotation = quat_to_dcm(x[IDX_QUAT]) if dcm is None else dcm
    air = compute_air_data(
        state.vel_b, state.quat, state.altitude, wind_n, dcm=rotation
    )
    coeffs = aero_model.coefficients(air, x[IDX_OMEGA], control, cfg.geometry)
    F_aero, M_aero = aero_forces_moments(air, coeffs, cfg.geometry)
    F_prop, M_prop = propulsion_forces_moments(control[3], air, cfg.propulsion)
    F_gravity = gravity_body(x[IDX_QUAT], cfg.mass.m, dcm=rotation)
    return F_aero + F_prop + F_gravity, M_aero + M_prop, air
