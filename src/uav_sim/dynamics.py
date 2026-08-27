"""Nonlinear rigid-body dynamics for Phase 1."""

from __future__ import annotations

import numpy as np

from uav_sim.config import AircraftConfig
from uav_sim.constants import G0
from uav_sim.rotations import quat_derivative, quat_to_dcm
from uav_sim.state import IDX_OMEGA, IDX_POS, IDX_QUAT, IDX_VEL, N_STATES


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Explicit 3-vector cross product.

    ``np.cross`` routes 3-vectors through ``moveaxis``/``normalize_axis_tuple``
    and is ~10x slower than this on scalars; it dominated the RK4 hot loop
    (two calls per derivative, four derivatives per step). Bit-identical to
    ``np.cross`` for (3,) float64 inputs.
    """
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=np.float64,
    )


def gravity_body(
    quat: np.ndarray, mass: float, dcm: np.ndarray | None = None
) -> np.ndarray:
    """Gravity force in the body frame.

    Implements ``F_grav_b = m * R_nb.T @ [0, 0, g]``.

    ``dcm`` optionally supplies a precomputed ``R_nb = quat_to_dcm(quat)`` so the
    caller can build the matrix once per derivative and reuse it (bit-identical
    to recomputing it).
    """
    gravity_n = np.array([0.0, 0.0, G0], dtype=np.float64)
    rotation = quat_to_dcm(quat) if dcm is None else dcm
    return mass * (rotation.T @ gravity_n)


def rigid_body_derivatives(
    x: np.ndarray,
    F_b: np.ndarray,
    M_b: np.ndarray,
    cfg: AircraftConfig,
    dcm: np.ndarray | None = None,
) -> np.ndarray:
    """Nonlinear 6-DOF rigid-body state derivative.

    Implements:
        ``pdot_n = R_nb @ v_b``
        ``vdot_b = F_b / m - omega_b x v_b``
        ``quatdot = 0.5 * Omega(omega_b) @ quat``
        ``omegadot = I^-1 @ (M_b - omega_b x (I @ omega_b))``

    Args:
        x: (13,) state vector.
        F_b: (3,) total force in body frame, N, including gravity.
        M_b: (3,) total moment about the CG in body frame, N m.
        cfg: Frozen aircraft configuration.

    Returns:
        (13,) state derivative.

    Reference: Stevens & Lewis, Aircraft Control and Simulation, 3rd ed.,
    equations 1.7-18 and 1.7-19.
    """
    state = np.asarray(x, dtype=np.float64)
    force_b = np.asarray(F_b, dtype=np.float64)
    moment_b = np.asarray(M_b, dtype=np.float64)
    if state.shape != (N_STATES,):
        raise ValueError(f"x must have shape ({N_STATES},), got {state.shape}")
    if force_b.shape != (3,):
        raise ValueError(f"F_b must have shape (3,), got {force_b.shape}")
    if moment_b.shape != (3,):
        raise ValueError(f"M_b must have shape (3,), got {moment_b.shape}")

    quat = state[IDX_QUAT]
    vel_b = state[IDX_VEL]
    omega_b = state[IDX_OMEGA]

    rotation = quat_to_dcm(quat) if dcm is None else dcm
    xdot = np.zeros(N_STATES, dtype=np.float64)
    xdot[IDX_POS] = rotation @ vel_b
    xdot[IDX_VEL] = force_b / cfg.mass.m - _cross(omega_b, vel_b)
    xdot[IDX_QUAT] = quat_derivative(quat, omega_b)
    angular_momentum_b = cfg.mass.inertia_tensor @ omega_b
    xdot[IDX_OMEGA] = cfg.mass.inertia_inverse @ (
        moment_b - _cross(omega_b, angular_momentum_b)
    )
    return xdot
