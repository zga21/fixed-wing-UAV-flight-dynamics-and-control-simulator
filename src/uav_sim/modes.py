"""Flight-dynamics mode identification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mode:
    """Named aircraft mode."""

    name: str
    eigenvalue: complex
    omega_n: float
    zeta: float
    period: float | None
    time_to_half: float
    is_stable: bool


def identify_modes(A_lon: np.ndarray, A_lat: np.ndarray) -> dict[str, Mode]:
    """Identify conventional longitudinal and lateral aircraft modes."""
    lon_modes = _classify_longitudinal(np.linalg.eigvals(A_lon))
    lat_modes = _classify_lateral(np.linalg.eigvals(A_lat))
    return {**lon_modes, **lat_modes}


def check_modes_plausible(modes: dict[str, Mode]) -> list[str]:
    """Return warnings for modes outside broad v1 Aerosonde plausibility ranges."""
    warnings: list[str] = []
    short = modes["short_period"]
    if not (0.5 <= short.omega_n <= 12.0 and 0.05 <= short.zeta <= 1.5):
        warnings.append("short_period outside expected small-UAV range")
    phugoid = modes["phugoid"]
    if not (0.05 <= phugoid.omega_n <= 1.0 and -0.2 <= phugoid.zeta <= 1.0):
        warnings.append("phugoid outside expected small-UAV range")
    dutch = modes["dutch_roll"]
    if not (0.05 <= dutch.omega_n <= 14.0 and -0.5 <= dutch.zeta <= 1.5):
        warnings.append("dutch_roll outside expected small-UAV range")
    roll = modes["roll_subsidence"]
    if not (roll.eigenvalue.real < -0.1):
        warnings.append("roll_subsidence is not a stable real pole")
    spiral = modes["spiral"]
    if abs(spiral.eigenvalue.real) > 1.0:
        warnings.append("spiral pole is too fast")
    return warnings


def _classify_longitudinal(eigenvalues: np.ndarray) -> dict[str, Mode]:
    pairs = _complex_representatives(eigenvalues)
    if len(pairs) < 2:
        real_sorted = sorted(
            eigenvalues,
            key=lambda value: abs(value.imag),
            reverse=True,
        )
        pairs = [complex(value) for value in real_sorted[:2]]
    pairs = sorted(pairs, key=lambda value: abs(value), reverse=True)
    return {
        "short_period": _mode("short_period", pairs[0]),
        "phugoid": _mode("phugoid", pairs[-1]),
    }


def _classify_lateral(eigenvalues: np.ndarray) -> dict[str, Mode]:
    complex_pairs = _complex_representatives(eigenvalues)
    dutch_value = (
        sorted(complex_pairs, key=lambda value: abs(value.imag), reverse=True)[0]
        if complex_pairs
        else complex(eigenvalues[np.argmax(np.abs(eigenvalues.imag))])
    )
    real_values = [
        complex(value)
        for value in eigenvalues
        if abs(value.imag) < 1e-7 and abs(value - dutch_value) > 1e-7
    ]
    real_values = sorted(real_values, key=lambda value: abs(value.real), reverse=True)
    stable_reals = [value for value in real_values if value.real < 0.0]
    roll_value = stable_reals[0] if stable_reals else real_values[0]
    remaining = [value for value in real_values if value != roll_value]
    spiral_value = (
        min(remaining, key=lambda value: abs(value.real)) if remaining else 0j
    )
    return {
        "dutch_roll": _mode("dutch_roll", dutch_value),
        "roll_subsidence": _mode("roll_subsidence", roll_value),
        "spiral": _mode("spiral", spiral_value),
    }


def _complex_representatives(eigenvalues: np.ndarray) -> list[complex]:
    values = [complex(value) for value in eigenvalues if value.imag > 1e-7]
    return sorted(values, key=lambda value: abs(value), reverse=True)


def _mode(name: str, eigenvalue: complex) -> Mode:
    omega_n = abs(eigenvalue)
    zeta = -eigenvalue.real / omega_n if omega_n > 0.0 else np.inf
    omega_d = abs(eigenvalue.imag)
    period = 2.0 * np.pi / omega_d if omega_d > 1e-12 else None
    if eigenvalue.real == 0.0:
        time_to_half = np.inf
    else:
        time_to_half = float(np.log(2.0) / abs(eigenvalue.real))
    return Mode(
        name=name,
        eigenvalue=eigenvalue,
        omega_n=float(omega_n),
        zeta=float(zeta),
        period=period,
        time_to_half=time_to_half,
        is_stable=eigenvalue.real < 0.0,
    )
