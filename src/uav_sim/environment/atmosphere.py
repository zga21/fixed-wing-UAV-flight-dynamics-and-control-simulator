"""International Standard Atmosphere model for the troposphere."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_sim.constants import G0, GAMMA_AIR, LAPSE_RATE, P_SL, R_AIR, RHO_SL, T_SL


@dataclass(frozen=True)
class AtmosphereState:
    """Atmospheric properties at one altitude."""

    rho: float
    temperature: float
    pressure: float
    sound_speed: float


def isa(altitude_m: float, delta_isa: float = 0.0) -> AtmosphereState:
    """ISA troposphere model, valid for this project's 0-1000 m envelope."""
    altitude = max(float(altitude_m), 0.0)
    standard_temperature = T_SL - LAPSE_RATE * altitude
    temperature = standard_temperature + delta_isa
    if temperature <= 0.0:
        raise ValueError("temperature must remain positive")
    pressure = P_SL * (standard_temperature / T_SL) ** (G0 / (LAPSE_RATE * R_AIR))
    rho = pressure / (R_AIR * temperature)
    sound_speed = float(np.sqrt(GAMMA_AIR * R_AIR * temperature))
    return AtmosphereState(
        rho=float(rho),
        temperature=float(temperature),
        pressure=float(pressure),
        sound_speed=sound_speed,
    )


def density_ratio(altitude_m: float) -> float:
    """Return ``rho(h) / rho_sea_level``."""
    return isa(altitude_m).rho / RHO_SL
