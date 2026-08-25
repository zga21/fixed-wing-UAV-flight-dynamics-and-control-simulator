"""Simulation result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SimResult:
    """Time histories with time as axis 0."""

    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    u_actual: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)
    terminated_early: bool = False
    termination_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary for later export stages."""
        return {
            "t": self.t.tolist(),
            "x": self.x.tolist(),
            "u": self.u.tolist(),
            "u_actual": self.u_actual.tolist(),
            "diagnostics": _json_ready(self.diagnostics),
            "terminated_early": self.terminated_early,
            "termination_reason": self.termination_reason,
        }

    def metrics(self) -> dict[str, float]:
        """Placeholder populated by Phase 4 scenario metrics."""
        return {}


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _json_ready(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_ready(child) for child in value]
    return value
