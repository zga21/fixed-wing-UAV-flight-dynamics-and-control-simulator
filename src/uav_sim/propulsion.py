"""Simple propeller thrust model.

Theory / assumptions (Phase 2):
- Momentum-theory (actuator-disk) thrust, Beard & McLain eq. 4.20 form:
  ``T = 0.5 * rho * S_prop * C_prop * ((k_motor * delta_t)**2 - V**2)``,
  clamped to non-negative values.
- Thrust acts along ``+x_b`` through the CG, so it contributes NO moment.
- KNOWN SIMPLIFICATIONS, deliberately omitted: propeller reaction torque
  (``-Q`` about the roll axis), gyroscopic precession from the spinning prop,
  and p-factor. These are small for this airframe but are a modelling choice,
  not an oversight. If Phase 6 NDI needs a higher-fidelity plant model, add
  them here (and, being a benchmark change, bump to a new config version).
"""

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
