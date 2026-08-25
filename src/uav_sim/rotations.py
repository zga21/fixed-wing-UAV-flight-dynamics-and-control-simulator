"""Quaternion and rotation utilities.

All quaternions are scalar-first, body-to-NED, and unit norm unless otherwise
stated. Euler angles are for I/O only.
"""

from __future__ import annotations

import numpy as np


def quat_normalise(quat: np.ndarray) -> np.ndarray:
    """Return ``quat`` scaled to unit norm.

    Args:
        quat: (4,) scalar-first quaternion.

    Returns:
        (4,) unit quaternion.

    Raises:
        ValueError: If the norm is too small to normalise safely.
    """
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        raise ValueError("cannot normalise a near-zero quaternion")
    return q / norm


def quat_to_dcm(quat: np.ndarray) -> np.ndarray:
    """Body-to-NED rotation matrix.

    Args:
        quat: (4,) [q0, q1, q2, q3], scalar first.

    Returns:
        (3, 3) matrix ``R_nb`` such that ``v_n = R_nb @ v_b``.
    """
    q0, q1, q2, q3 = quat_normalise(quat)
    return np.array(
        [
            [
                q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
                2.0 * (q1 * q2 - q0 * q3),
                2.0 * (q1 * q3 + q0 * q2),
            ],
            [
                2.0 * (q1 * q2 + q0 * q3),
                q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
                2.0 * (q2 * q3 - q0 * q1),
            ],
            [
                2.0 * (q1 * q3 - q0 * q2),
                2.0 * (q2 * q3 + q0 * q1),
                q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
            ],
        ],
        dtype=np.float64,
    )


def quat_derivative(quat: np.ndarray, omega_b: np.ndarray) -> np.ndarray:
    """Quaternion rate from body angular rates.

    Implements ``qdot = 0.5 * Omega(omega_b) @ quat``.

    Args:
        quat: (4,) scalar-first quaternion.
        omega_b: (3,) body rates [p, q, r], rad/s.

    Returns:
        (4,) quaternion derivative.
    """
    q = np.asarray(quat, dtype=np.float64)
    p, q_rate, r = np.asarray(omega_b, dtype=np.float64)
    omega = np.array(
        [
            [0.0, -p, -q_rate, -r],
            [p, 0.0, r, -q_rate],
            [q_rate, -r, 0.0, p],
            [r, q_rate, -p, 0.0],
        ],
        dtype=np.float64,
    )
    return 0.5 * omega @ q


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to 3-2-1 Euler angles ``[phi, theta, psi]`` rad."""
    q0, q1, q2, q3 = quat_normalise(quat)
    phi = np.arctan2(
        2.0 * (q0 * q1 + q2 * q3),
        q0 * q0 + q3 * q3 - q1 * q1 - q2 * q2,
    )
    theta_arg = 2.0 * (q0 * q2 - q1 * q3)
    theta = np.arcsin(np.clip(theta_arg, -1.0, 1.0))
    psi = np.arctan2(
        2.0 * (q0 * q3 + q1 * q2),
        q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
    )
    return np.array([phi, theta, psi], dtype=np.float64)


def euler_to_quat(phi: float, theta: float, psi: float) -> np.ndarray:
    """Convert 3-2-1 Euler angles to a scalar-first body-to-NED quaternion."""
    half_phi = 0.5 * phi
    half_theta = 0.5 * theta
    half_psi = 0.5 * psi
    c_phi, s_phi = np.cos(half_phi), np.sin(half_phi)
    c_theta, s_theta = np.cos(half_theta), np.sin(half_theta)
    c_psi, s_psi = np.cos(half_psi), np.sin(half_psi)

    quat = np.array(
        [
            c_phi * c_theta * c_psi + s_phi * s_theta * s_psi,
            s_phi * c_theta * c_psi - c_phi * s_theta * s_psi,
            c_phi * s_theta * c_psi + s_phi * c_theta * s_psi,
            c_phi * c_theta * s_psi - s_phi * s_theta * c_psi,
        ],
        dtype=np.float64,
    )
    return quat_normalise(quat)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two scalar-first quaternions."""
    a0, a1, a2, a3 = np.asarray(a, dtype=np.float64)
    b0, b1, b2, b3 = np.asarray(b, dtype=np.float64)
    return np.array(
        [
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ],
        dtype=np.float64,
    )
