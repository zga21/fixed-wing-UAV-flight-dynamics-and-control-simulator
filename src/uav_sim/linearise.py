"""Numerical linearisation utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from uav_sim.config import AircraftConfig
from uav_sim.plant import AircraftPlant
from uav_sim.rotations import euler_to_quat, quat_normalise, quat_to_euler
from uav_sim.state import IDX_OMEGA, IDX_POS, IDX_QUAT, IDX_VEL, N_STATES, State
from uav_sim.trim import TrimPoint


@dataclass(frozen=True)
class LinearModel:
    """Named linear subsystem."""

    A: np.ndarray
    B: np.ndarray
    state_names: tuple[str, ...]
    input_names: tuple[str, ...]


EULER_STATE_NAMES = (
    "p_N",
    "p_E",
    "p_D",
    "u",
    "v",
    "w",
    "phi",
    "theta",
    "psi",
    "p",
    "q",
    "r",
)
INPUT_NAMES = ("delta_a", "delta_e", "delta_r", "delta_t")
LONGITUDINAL_IDX = np.array([3, 5, 10, 7, 2])
LATERAL_IDX = np.array([4, 9, 11, 6])
LONGITUDINAL_INPUT_IDX = np.array([1, 3])
LATERAL_INPUT_IDX = np.array([0, 2])


def linearise(
    trim_pt: TrimPoint,
    plant: AircraftPlant,
    cfg: AircraftConfig,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerical Jacobians of the 13-state quaternion model."""

    def f_x(x: np.ndarray) -> np.ndarray:
        return plant.derivatives(0.0, x, trim_pt.u)

    A = _central_difference_state(f_x, trim_pt.x, eps)
    B = _central_difference_controls(
        lambda u: plant.derivatives(0.0, trim_pt.x, u),
        trim_pt.u,
        eps,
    )
    return A, B


def linearise_euler(
    trim_pt: TrimPoint,
    plant: AircraftPlant,
    cfg: AircraftConfig,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerical Jacobians of the conventional 12-state Euler model."""
    y0 = euler_state_from_full(trim_pt.x)

    def f_y(y: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = full_state_from_euler(y)
        xdot = plant.derivatives(0.0, x, u)
        return euler_derivative_from_full(x, xdot)

    A = _central_difference(lambda y: f_y(y, trim_pt.u), y0, eps)
    B = _central_difference(lambda u: f_y(y0, u), trim_pt.u, eps)
    return A, B


def reduce_to_euler_states(
    A: np.ndarray,
    B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop the redundant scalar quaternion row/column for simple diagnostics."""
    keep = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
    return A[np.ix_(keep, keep)], B[keep]


def decouple(A: np.ndarray, B: np.ndarray) -> tuple[LinearModel, LinearModel]:
    """Split a 12-state Euler model into longitudinal and lateral subsystems."""
    A_lon = A[np.ix_(LONGITUDINAL_IDX, LONGITUDINAL_IDX)]
    B_lon = B[np.ix_(LONGITUDINAL_IDX, LONGITUDINAL_INPUT_IDX)]
    A_lat = A[np.ix_(LATERAL_IDX, LATERAL_IDX)]
    B_lat = B[np.ix_(LATERAL_IDX, LATERAL_INPUT_IDX)]
    lon_names = tuple(EULER_STATE_NAMES[i] for i in LONGITUDINAL_IDX)
    lat_names = tuple(EULER_STATE_NAMES[i] for i in LATERAL_IDX)
    return (
        LinearModel(A_lon, B_lon, lon_names, ("delta_e", "delta_t")),
        LinearModel(A_lat, B_lat, lat_names, ("delta_a", "delta_r")),
    )


def coupling_norms(A: np.ndarray) -> tuple[float, float]:
    """Return max cross-coupling magnitudes between longitudinal and lateral sets."""
    lon_to_lat = np.abs(A[np.ix_(LATERAL_IDX, LONGITUDINAL_IDX)]).max()
    lat_to_lon = np.abs(A[np.ix_(LONGITUDINAL_IDX, LATERAL_IDX)]).max()
    return float(lon_to_lat), float(lat_to_lon)


def euler_state_from_full(x: np.ndarray) -> np.ndarray:
    """Convert a 13-state quaternion vector to a 12-state Euler vector."""
    state = State.from_array(x)
    y = np.empty(12, dtype=np.float64)
    y[0:3] = state.pos_n
    y[3:6] = state.vel_b
    y[6:9] = quat_to_euler(state.quat)
    y[9:12] = state.omega_b
    return y


def full_state_from_euler(y: np.ndarray) -> np.ndarray:
    """Convert a 12-state Euler vector to a 13-state quaternion vector."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.shape != (12,):
        raise ValueError(f"Euler state must have shape (12,), got {arr.shape}")
    x = np.empty(N_STATES, dtype=np.float64)
    x[IDX_POS] = arr[0:3]
    x[IDX_VEL] = arr[3:6]
    x[IDX_QUAT] = euler_to_quat(arr[6], arr[7], arr[8])
    x[IDX_OMEGA] = arr[9:12]
    return x


def euler_derivative_from_full(x: np.ndarray, xdot: np.ndarray) -> np.ndarray:
    """Convert full-state derivative to a 12-state Euler derivative."""
    ydot = np.empty(12, dtype=np.float64)
    ydot[0:3] = xdot[IDX_POS]
    ydot[3:6] = xdot[IDX_VEL]
    phi, theta, _psi = quat_to_euler(x[IDX_QUAT])
    p, q_rate, r = x[IDX_OMEGA]
    cos_theta = np.cos(theta)
    if abs(cos_theta) < 1e-8:
        raise ValueError("Euler derivative is singular near theta = +/- 90 deg")
    ydot[6] = p + np.sin(phi) * np.tan(theta) * q_rate
    ydot[6] += np.cos(phi) * np.tan(theta) * r
    ydot[7] = np.cos(phi) * q_rate - np.sin(phi) * r
    ydot[8] = np.sin(phi) / cos_theta * q_rate + np.cos(phi) / cos_theta * r
    ydot[9:12] = xdot[IDX_OMEGA]
    return ydot


def _central_difference_state(
    fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    eps: float,
) -> np.ndarray:
    A = np.empty((N_STATES, N_STATES), dtype=np.float64)
    for idx in range(N_STATES):
        step = np.zeros(N_STATES, dtype=np.float64)
        step[idx] = eps
        x_plus = x0 + step
        x_minus = x0 - step
        if idx in range(IDX_QUAT.start, IDX_QUAT.stop):
            x_plus = x_plus.copy()
            x_minus = x_minus.copy()
            x_plus[IDX_QUAT] = quat_normalise(x_plus[IDX_QUAT])
            x_minus[IDX_QUAT] = quat_normalise(x_minus[IDX_QUAT])
        A[:, idx] = (fn(x_plus) - fn(x_minus)) / (2.0 * eps)
    return A


def _central_difference_controls(
    fn: Callable[[np.ndarray], np.ndarray],
    u0: np.ndarray,
    eps: float,
) -> np.ndarray:
    return _central_difference(fn, u0, eps)


def _central_difference(
    fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    eps: float,
) -> np.ndarray:
    base = np.asarray(x0, dtype=np.float64)
    y0 = np.asarray(fn(base), dtype=np.float64)
    jac = np.empty((y0.size, base.size), dtype=np.float64)
    for idx in range(base.size):
        step = np.zeros_like(base)
        step[idx] = eps
        jac[:, idx] = (fn(base + step) - fn(base - step)) / (2.0 * eps)
    return jac
