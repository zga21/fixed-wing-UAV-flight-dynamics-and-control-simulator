"""Simple propeller thrust model."""

from __future__ import annotations

import numpy as np

from uav_sim.air_data import AirData
from uav_sim.config import Propulsion


def propeller_thrust(
    delta_t: float,
    V: float,
    rho: float,
    prop: Propulsion,
) -> float:
    """Momentum-theory propeller thrust, clamped to non-negative values."""
    throttle = float(np.clip(delta_t, 0.0, 1.0))
    airspeed = max(float(V), 0.0)
    thrust = 0.5 * rho * prop.S_prop * prop.C_prop
    thrust *= (prop.k_motor * throttle) ** 2 - airspeed**2
    return float(max(thrust, 0.0))


def propulsion_forces_moments(
    delta_t: float,
    air: AirData,
    prop: Propulsion,
) -> tuple[np.ndarray, np.ndarray]:
    """Return propulsive force and moment in body axes."""
    thrust = propeller_thrust(delta_t, air.V, air.rho, prop)
    return (
        np.array([thrust, 0.0, 0.0], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
    )
