"""Controller data contracts shared by baseline and robust controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Reference:
    """What the guidance layer is asking for."""

    altitude: float | None = None
    airspeed: float | None = None
    heading: float | None = None
    roll: float | None = None
    pitch: float | None = None
    waypoint_n: np.ndarray | None = None


@dataclass(frozen=True)
class Controls:
    """Aircraft control vector in the frozen Phase 0 order."""

    delta_a: float
    delta_e: float
    delta_r: float
    delta_t: float

    def to_array(self) -> np.ndarray:
        """Return ``[delta_a, delta_e, delta_r, delta_t]``."""
        return np.array(
            [self.delta_a, self.delta_e, self.delta_r, self.delta_t],
            dtype=np.float64,
        )


@runtime_checkable
class Controller(Protocol):
    """Common controller protocol for baseline PID and later NDI."""

    def update(
        self,
        x_hat: np.ndarray,
        ref: Reference,
        t: float,
        dt: float,
    ) -> Controls:
        """Return controls from an estimated state and command reference."""

    def reset(self) -> None:
        """Clear all internal controller state."""

    @property
    def gains(self) -> np.ndarray:
        """Flat tunable gain vector."""

    @gains.setter
    def gains(self, k: np.ndarray) -> None:
        """Set all tunable gains from a flat vector."""

    @property
    def gain_names(self) -> list[str]:
        """Names matching the flat gain vector."""

    @property
    def gain_bounds(self) -> list[tuple[float, float]]:
        """Bounds matching the flat gain vector."""


class ZeroController:
    """Protocol-compliant zero controller for harness tests."""

    def update(
        self,
        _x_hat: np.ndarray,
        _ref: Reference,
        _t: float,
        _dt: float,
    ) -> Controls:
        return Controls(0.0, 0.0, 0.0, 0.0)

    def __call__(self, t: float, x_hat: np.ndarray, _cfg, _rng) -> np.ndarray:
        """Adapter for ``integrate.simulate``."""
        return self.update(x_hat, Reference(), t, _cfg.integration.dt).to_array()

    def reset(self) -> None:
        """No internal state to clear."""

    @property
    def gains(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float64)

    @gains.setter
    def gains(self, k: np.ndarray) -> None:
        arr = np.asarray(k, dtype=np.float64)
        if arr.shape != (0,):
            raise ValueError(
                f"ZeroController gains must have shape (0,), got {arr.shape}"
            )

    @property
    def gain_names(self) -> list[str]:
        return []

    @property
    def gain_bounds(self) -> list[tuple[float, float]]:
        return []
