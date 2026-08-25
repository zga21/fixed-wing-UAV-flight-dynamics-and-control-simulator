"""Typed, frozen project configuration.

Configuration is loaded once at the boundary, validated once, then passed into
pure physics functions. YAML may use degrees for human-facing fields; loaded
objects expose radians.
"""

from __future__ import annotations

import hashlib
import json
import math
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from uav_sim.constants import G0


class FrozenModel(BaseModel):
    """Base model for benchmark config objects."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


def _deg_to_rad(value: Any) -> float:
    return math.radians(float(value))


class MassProperties(FrozenModel):
    m: float = Field(gt=0)
    I_xx: float = Field(gt=0)
    I_yy: float = Field(gt=0)
    I_zz: float = Field(gt=0)
    I_xz: float = 0.0
    x_cg_frac: float = Field(ge=0, le=1)

    @cached_property
    def inertia_tensor(self) -> np.ndarray:
        """(3, 3) body-axis inertia tensor, kg m^2.

        Cached: built once per config object. The dynamics loop calls this
        millions of times per Monte Carlo campaign (see T1.3), so it must not
        be rebuilt on every access.
        """
        return np.array(
            [
                [self.I_xx, 0.0, -self.I_xz],
                [0.0, self.I_yy, 0.0],
                [-self.I_xz, 0.0, self.I_zz],
            ],
            dtype=np.float64,
        )

    @cached_property
    def inertia_inverse(self) -> np.ndarray:
        """(3, 3) inverse inertia tensor, precomputed outside the dynamics loop.

        Cached: T1.3 requires the dynamics derivative to use a precomputed
        inverse and never call ``np.linalg.inv`` inside the hot loop.
        """
        return np.linalg.inv(self.inertia_tensor)

    @model_validator(mode="after")
    def inertia_must_be_positive_definite(self) -> MassProperties:
        eigvals = np.linalg.eigvalsh(self.inertia_tensor)
        if np.any(eigvals <= 0.0):
            raise ValueError("inertia tensor must be symmetric positive-definite")
        return self


class Geometry(FrozenModel):
    S: float = Field(gt=0)
    b: float = Field(gt=0)
    c_bar: float = Field(gt=0)

    @property
    def aspect_ratio(self) -> float:
        return self.b**2 / self.S


class Propulsion(FrozenModel):
    S_prop: float = Field(gt=0)
    C_prop: float = Field(gt=0)
    k_motor: float = Field(gt=0)


class AeroDerivatives(FrozenModel):
    C_L_0: float
    C_L_alpha: float
    C_L_q: float
    C_L_delta_e: float
    C_D_0: float = Field(gt=0)
    C_D_alpha: float = Field(ge=0)
    C_D_q: float
    C_D_delta_e: float
    oswald_e: float = Field(gt=0, le=1)
    C_m_0: float
    C_m_alpha: float
    C_m_q: float
    C_m_delta_e: float
    C_Y_0: float
    C_Y_beta: float
    C_Y_p: float
    C_Y_r: float
    C_Y_delta_a: float
    C_Y_delta_r: float
    C_l_0: float
    C_l_beta: float
    C_l_p: float
    C_l_r: float
    C_l_delta_a: float
    C_l_delta_r: float
    C_n_0: float
    C_n_beta: float
    C_n_p: float
    C_n_r: float
    C_n_delta_a: float
    C_n_delta_r: float

    @field_validator("C_m_alpha", "C_m_q", "C_l_p", "C_n_r", "C_l_beta", "C_Y_beta")
    @classmethod
    def must_be_negative(cls, value: float, info) -> float:
        if value >= 0.0:
            raise ValueError(
                f"{info.field_name} must be negative for a stable, damped "
                "aircraft. Check docs/conventions.md sign convention."
            )
        return value

    @field_validator("C_L_alpha", "C_l_delta_a", "C_n_beta", "C_n_delta_r")
    @classmethod
    def must_be_positive(cls, value: float, info) -> float:
        if value <= 0.0:
            raise ValueError(
                f"{info.field_name} must be positive under "
                "docs/conventions.md sign convention."
            )
        return value

    @field_validator("C_m_delta_e")
    @classmethod
    def elevator_moment_must_be_negative(cls, value: float) -> float:
        if value >= 0.0:
            raise ValueError(
                "C_m_delta_e must be negative because positive elevator is "
                "defined as trailing-edge down."
            )
        return value


class SurfaceActuator(FrozenModel):
    min_rad: float
    max_rad: float
    rate_rad_s: float = Field(gt=0)
    tau: float = Field(gt=0)

    @model_validator(mode="before")
    @classmethod
    def convert_degree_inputs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        converted = dict(data)
        if "min_deg" in converted:
            converted["min_rad"] = _deg_to_rad(converted.pop("min_deg"))
        if "max_deg" in converted:
            converted["max_rad"] = _deg_to_rad(converted.pop("max_deg"))
        if "rate_dps" in converted:
            converted["rate_rad_s"] = _deg_to_rad(converted.pop("rate_dps"))
        return converted

    @model_validator(mode="after")
    def limits_must_be_ordered(self) -> SurfaceActuator:
        if self.min_rad >= self.max_rad:
            raise ValueError("surface actuator min angle must be below max angle")
        return self


class ThrottleActuator(FrozenModel):
    min: float = Field(ge=0, le=1)
    max: float = Field(ge=0, le=1)
    rate: float = Field(gt=0)
    tau: float = Field(gt=0)

    @model_validator(mode="after")
    def limits_must_be_ordered(self) -> ThrottleActuator:
        if self.min >= self.max:
            raise ValueError("throttle min must be below max")
        return self


class Actuators(FrozenModel):
    aileron: SurfaceActuator
    elevator: SurfaceActuator
    rudder: SurfaceActuator
    throttle: ThrottleActuator


class Envelope(FrozenModel):
    V_min: float = Field(gt=0)
    V_nominal: float = Field(gt=0)
    V_max: float = Field(gt=0)
    h_min: float = Field(ge=0)
    h_nominal: float = Field(ge=0)
    h_max: float = Field(gt=0)
    alpha_min_rad: float
    alpha_max_rad: float
    beta_max_rad: float = Field(gt=0)
    phi_max_rad: float = Field(gt=0)
    load_factor_min: float
    load_factor_max: float = Field(gt=0)

    @model_validator(mode="before")
    @classmethod
    def convert_degree_inputs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        converted = dict(data)
        aliases = {
            "alpha_min_deg": "alpha_min_rad",
            "alpha_max_deg": "alpha_max_rad",
            "beta_max_deg": "beta_max_rad",
            "phi_max_deg": "phi_max_rad",
        }
        for source, target in aliases.items():
            if source in converted:
                converted[target] = _deg_to_rad(converted.pop(source))
        return converted

    @model_validator(mode="after")
    def ranges_must_be_ordered(self) -> Envelope:
        if not (self.V_min < self.V_nominal < self.V_max):
            raise ValueError("airspeed envelope must satisfy V_min < V_nominal < V_max")
        if not (self.h_min <= self.h_nominal < self.h_max):
            raise ValueError(
                "altitude envelope must satisfy h_min <= h_nominal < h_max"
            )
        if self.alpha_min_rad >= self.alpha_max_rad:
            raise ValueError("alpha_min_rad must be below alpha_max_rad")
        if self.load_factor_min >= self.load_factor_max:
            raise ValueError("load factor bounds are not ordered")
        return self


class Integration(FrozenModel):
    method: Literal["rk4"]
    dt: float = Field(gt=0)


class AircraftConfig(FrozenModel):
    name: str
    conventions_version: Literal["v1"]
    mass: MassProperties
    geometry: Geometry
    propulsion: Propulsion
    aero: AeroDerivatives
    actuators: Actuators
    envelope: Envelope
    integration: Integration

    @property
    def wing_loading(self) -> float:
        return self.mass.m * G0 / self.geometry.S


def load_aircraft(path: str | Path) -> AircraftConfig:
    """Load, validate, convert I/O degrees to radians, and freeze config."""
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return AircraftConfig.model_validate(data)


def config_hash(cfg: AircraftConfig) -> str:
    """Short stable hash recorded with benchmark results."""
    payload = json.dumps(
        cfg.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
