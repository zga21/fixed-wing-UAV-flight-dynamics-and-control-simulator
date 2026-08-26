"""Steady-flight trim solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from uav_sim.config import AircraftConfig
from uav_sim.constants import G0
from uav_sim.environment.atmosphere import isa
from uav_sim.plant import AircraftPlant
from uav_sim.rotations import euler_to_quat, quat_to_euler
from uav_sim.state import IDX_OMEGA, IDX_QUAT, IDX_VEL, State, make_state


@dataclass(frozen=True)
class TrimPoint:
    """Trimmed state and controls for steady flight."""

    x: np.ndarray
    u: np.ndarray
    V: float
    gamma: float
    altitude: float
    alpha: float
    theta: float
    residual: float
    iterations: int


class TrimNotConverged(Exception):
    """Raised when a steady-flight trim cannot be found."""


def trim(
    V: float,
    gamma: float,
    altitude: float,
    cfg: AircraftConfig,
    plant: AircraftPlant,
    phi: float = 0.0,
) -> TrimPoint:
    """Solve wings-level coordinated steady flight at ``(V, gamma, altitude)``."""
    _check_requested_trim(V, gamma, altitude, cfg)
    initial = _initial_guess(V, gamma, altitude, cfg)
    lower = np.array([np.deg2rad(-25.0), -np.deg2rad(25.0), 0.0])
    upper = np.array([np.deg2rad(25.0), np.deg2rad(25.0), 1.0])

    result = least_squares(
        _trim_residual,
        initial,
        args=(V, gamma, altitude, cfg, plant, phi),
        bounds=(lower, upper),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        x_scale=np.array([0.1, 0.1, 0.5]),
        max_nfev=300,
    )
    alpha, delta_e, delta_t = result.x
    x = _trim_state(V, gamma, altitude, alpha, phi)
    u = np.array([0.0, delta_e, 0.0, delta_t], dtype=np.float64)
    residual = _dynamic_residual_norm(x, u, plant)
    if not result.success or residual > 1e-8:
        raise TrimNotConverged(
            "trim failed for "
            f"V={V:.3g} m/s, gamma={np.rad2deg(gamma):.3g} deg, "
            f"h={altitude:.3g} m: residual={residual:.3e}"
        )
    return TrimPoint(
        x=x,
        u=u,
        V=float(V),
        gamma=float(gamma),
        altitude=float(altitude),
        alpha=float(alpha),
        theta=float(alpha + gamma),
        residual=float(residual),
        iterations=int(result.nfev),
    )


def trim_grid(
    V_list: list[float] | np.ndarray,
    gamma_list: list[float] | np.ndarray,
    altitude: float,
    cfg: AircraftConfig,
    plant: AircraftPlant,
) -> list[TrimPoint]:
    """Trim a grid of speeds and flight-path angles."""
    points: list[TrimPoint] = []
    for gamma in gamma_list:
        for V in V_list:
            points.append(trim(float(V), float(gamma), altitude, cfg, plant))
    return points


def _check_requested_trim(
    V: float,
    gamma: float,
    altitude: float,
    cfg: AircraftConfig,
) -> None:
    if not (cfg.envelope.V_min <= V <= cfg.envelope.V_max):
        raise TrimNotConverged(f"V={V} is outside the v1 trim envelope")
    if not (cfg.envelope.h_min <= altitude <= cfg.envelope.h_max):
        raise TrimNotConverged(f"altitude={altitude} is outside the v1 envelope")
    if abs(gamma) > np.deg2rad(10.0):
        raise TrimNotConverged("flight-path angle is outside the Phase 3 trim range")


def _initial_guess(
    V: float,
    gamma: float,
    altitude: float,
    cfg: AircraftConfig,
) -> np.ndarray:
    rho = isa(altitude).rho
    lift_slope = max(cfg.aero.C_L_alpha, 1e-6)
    alpha = 2.0 * cfg.mass.m * G0 * np.cos(gamma)
    alpha /= rho * V**2 * cfg.geometry.S * lift_slope
    alpha -= cfg.aero.C_L_0 / lift_slope
    alpha = np.clip(alpha, cfg.envelope.alpha_min_rad, cfg.envelope.alpha_max_rad)
    delta_e = (cfg.aero.C_m_0 + cfg.aero.C_m_alpha * alpha) / -cfg.aero.C_m_delta_e
    delta_e = np.clip(delta_e, -np.deg2rad(20.0), np.deg2rad(20.0))
    throttle = max(0.35, V / cfg.propulsion.k_motor + 0.05)
    return np.array([alpha, delta_e, min(throttle, 0.95)], dtype=np.float64)


def _trim_state(
    V: float,
    gamma: float,
    altitude: float,
    alpha: float,
    phi: float = 0.0,
) -> np.ndarray:
    theta = alpha + gamma
    return make_state(
        pos_n=np.array([0.0, 0.0, -altitude], dtype=np.float64),
        vel_b=np.array([V * np.cos(alpha), 0.0, V * np.sin(alpha)], dtype=np.float64),
        quat=euler_to_quat(phi, theta, 0.0),
        omega_b=np.zeros(3, dtype=np.float64),
    )


def _trim_residual(
    variables: np.ndarray,
    V: float,
    gamma: float,
    altitude: float,
    cfg: AircraftConfig,
    plant: AircraftPlant,
    phi: float,
) -> np.ndarray:
    alpha, delta_e, delta_t = variables
    x = _trim_state(V, gamma, altitude, alpha, phi)
    u = np.array([0.0, delta_e, 0.0, delta_t], dtype=np.float64)
    xdot = plant.derivatives(0.0, x, u)
    state = State.from_array(x)
    gamma_error = xdot[2] + V * np.sin(gamma)
    return np.array(
        [
            xdot[IDX_VEL][0] / G0,
            xdot[IDX_VEL][2] / G0,
            xdot[IDX_OMEGA][1],
            gamma_error / max(V, 1.0),
            state.vel_b[1],
        ],
        dtype=np.float64,
    )


def _dynamic_residual_norm(
    x: np.ndarray,
    u: np.ndarray,
    plant: AircraftPlant,
) -> float:
    xdot = plant.derivatives(0.0, x, u)
    euler = quat_to_euler(x[IDX_QUAT])
    theta = euler[1]
    speed = np.linalg.norm(x[IDX_VEL])
    gamma = theta - np.arctan2(x[IDX_VEL][2], x[IDX_VEL][0])
    expected_pdot_d = -speed * np.sin(gamma)
    residual = np.array(
        [
            xdot[IDX_VEL][0],
            xdot[IDX_VEL][1],
            xdot[IDX_VEL][2],
            xdot[IDX_OMEGA][0],
            xdot[IDX_OMEGA][1],
            xdot[IDX_OMEGA][2],
            xdot[2] - expected_pdot_d,
        ],
        dtype=np.float64,
    )
    return float(np.linalg.norm(residual))
