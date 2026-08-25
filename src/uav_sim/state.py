"""State vector layout.

Single source of truth for indexing:

``x = [p_N, p_E, p_D, u, v, w, q0, q1, q2, q3, p, q, r]``

Bare ``q`` is pitch rate. The quaternion is always named ``quat``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_sim.rotations import euler_to_quat, quat_normalise, quat_to_euler

N_STATES = 13

IDX_POS = slice(0, 3)
IDX_VEL = slice(3, 6)
IDX_QUAT = slice(6, 10)
IDX_OMEGA = slice(10, 13)


@dataclass(frozen=True)
class State:
    """Readable view of a state vector. Avoid using this inside hot loops."""

    pos_n: np.ndarray
    vel_b: np.ndarray
    quat: np.ndarray
    omega_b: np.ndarray

    @classmethod
    def from_array(cls, x: np.ndarray) -> State:
        """Create a readable state view from a ``(13,)`` array."""
        arr = _as_state_array(x)
        return cls(
            pos_n=arr[IDX_POS].copy(),
            vel_b=arr[IDX_VEL].copy(),
            quat=arr[IDX_QUAT].copy(),
            omega_b=arr[IDX_OMEGA].copy(),
        )

    def to_array(self) -> np.ndarray:
        """Convert the readable state back to a ``(13,)`` array."""
        x = np.empty(N_STATES, dtype=np.float64)
        x[IDX_POS] = self.pos_n
        x[IDX_VEL] = self.vel_b
        x[IDX_QUAT] = self.quat
        x[IDX_OMEGA] = self.omega_b
        return x

    @property
    def altitude(self) -> float:
        """Altitude in metres, using the frozen convention ``h = -p_D``."""
        return float(-self.pos_n[2])

    @property
    def euler(self) -> np.ndarray:
        """(3,) Euler angles ``[phi, theta, psi]`` rad for output only."""
        return quat_to_euler(self.quat)


def make_state(
    pos_n: np.ndarray | list[float] | tuple[float, ...],
    vel_b: np.ndarray | list[float] | tuple[float, ...],
    quat: np.ndarray | list[float] | tuple[float, ...],
    omega_b: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Assemble and validate a ``(13,)`` state array."""
    pos = _as_vector(pos_n, 3, "pos_n")
    vel = _as_vector(vel_b, 3, "vel_b")
    q = _as_vector(quat, 4, "quat")
    omega = _as_vector(omega_b, 3, "omega_b")
    if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("quat must be unit norm to within 1e-6")

    x = np.empty(N_STATES, dtype=np.float64)
    x[IDX_POS] = pos
    x[IDX_VEL] = vel
    x[IDX_QUAT] = q
    x[IDX_OMEGA] = omega
    return x


def initial_state(
    V: float,
    alpha: float = 0.0,
    h: float = 100.0,
    psi: float = 0.0,
) -> np.ndarray:
    """Convenience constructor for level initial conditions."""
    vel_b = np.array([V * np.cos(alpha), 0.0, V * np.sin(alpha)], dtype=np.float64)
    quat = euler_to_quat(0.0, alpha, psi)
    return make_state(
        pos_n=np.array([0.0, 0.0, -h], dtype=np.float64),
        vel_b=vel_b,
        quat=quat,
        omega_b=np.zeros(3, dtype=np.float64),
    )


def set_quat(x: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Return a copy of ``x`` with a normalised quaternion."""
    arr = _as_state_array(x).copy()
    arr[IDX_QUAT] = quat_normalise(quat)
    return arr


def _as_state_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape != (N_STATES,):
        raise ValueError(f"state must have shape ({N_STATES},), got {arr.shape}")
    return arr


def _as_vector(
    value: np.ndarray | list[float] | tuple[float, ...],
    size: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {arr.shape}")
    return arr
