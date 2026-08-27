"""Versioned Phase 5 realism configuration and plant construction."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import yaml

from uav_sim.config import AircraftConfig
from uav_sim.environment.wind import DrydenTurbulence, SteadyWind, WindField
from uav_sim.estimation.ekf import NavigationEKF
from uav_sim.hardware.actuator import ActuatorBank
from uav_sim.hardware.sensors import SensorSuite, load_sensor_spec
from uav_sim.plant import AircraftPlant
from uav_sim.state import initial_state


@dataclass(frozen=True)
class ActuatorRealism:
    ideal: bool = True


@dataclass(frozen=True)
class SensorRealism:
    ideal: bool = True
    spec_path: str = "config/sensors_v1.yaml"


@dataclass(frozen=True)
class EstimatorRealism:
    mode: Literal["truth", "ekf"] = "truth"


@dataclass(frozen=True)
class WindRealism:
    steady_speed: float = 0.0
    direction_rad: float = 0.0
    shear_exponent: float = 0.0
    turbulence: Literal["zero", "light", "moderate", "severe"] = "zero"


@dataclass(frozen=True)
class RealismConfig:
    actuators: ActuatorRealism = ActuatorRealism()
    sensors: SensorRealism = SensorRealism()
    estimator: EstimatorRealism = EstimatorRealism()
    wind: WindRealism = WindRealism()


def load_realism(path: str | Path) -> RealismConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    wind = raw.get("wind", {})
    return RealismConfig(
        actuators=ActuatorRealism(bool(raw.get("actuators", {}).get("ideal", True))),
        sensors=SensorRealism(
            ideal=bool(raw.get("sensors", {}).get("ideal", True)),
            spec_path=str(
                raw.get("sensors", {}).get("spec_path", "config/sensors_v1.yaml")
            ),
        ),
        estimator=EstimatorRealism(str(raw.get("estimator", {}).get("mode", "truth"))),  # type: ignore[arg-type]
        wind=WindRealism(
            steady_speed=float(wind.get("steady_speed", 0.0)),
            direction_rad=float(np.deg2rad(wind.get("direction_deg", 0.0))),
            shear_exponent=float(wind.get("shear_exponent", 0.0)),
            turbulence=str(wind.get("turbulence", "zero")),  # type: ignore[arg-type]
        ),
    )


def build_plant(cfg: AircraftConfig, realism: RealismConfig) -> AircraftPlant:
    """Build a plant whose optional components are reset from each run seed."""
    actuators = (
        None if realism.actuators.ideal else ActuatorBank.from_config(cfg, ideal=False)
    )
    wind_model = None
    if realism.wind.steady_speed > 0.0 or realism.wind.turbulence != "zero":
        steady = (
            None
            if realism.wind.steady_speed <= 0.0
            else SteadyWind(
                realism.wind.steady_speed,
                realism.wind.direction_rad,
                realism.wind.shear_exponent,
            )
        )
        turbulence = (
            None
            if realism.wind.turbulence == "zero"
            else DrydenTurbulence(
                realism.wind.turbulence,
                cfg.envelope.h_nominal,
                np.random.default_rng(0),
            )
        )
        wind_model = WindField(steady=steady, turbulence=turbulence)

    sensors = None
    estimator = None
    if realism.estimator.mode == "ekf":
        sensor_spec = replace(
            load_sensor_spec(realism.sensors.spec_path), ideal=realism.sensors.ideal
        )
        sensors = SensorSuite(sensor_spec, np.random.default_rng(0))
        estimator = NavigationEKF(
            cfg, x0=initial_state(cfg.envelope.V_nominal, h=cfg.envelope.h_nominal)
        )
    return AircraftPlant(
        cfg,
        wind_model=wind_model,
        actuators=actuators,
        sensors=sensors,
        estimator=estimator,
    )


def degradation_configurations() -> dict[str, RealismConfig]:
    """The frozen C0-C5 ablation matrix from T5.6."""
    perfect = RealismConfig()
    return {
        "C0 perfect": perfect,
        "C1 actuators": replace(perfect, actuators=ActuatorRealism(False)),
        "C2 sensors+EKF": replace(
            perfect,
            sensors=SensorRealism(False),
            estimator=EstimatorRealism("ekf"),
        ),
        "C3 steady wind": replace(
            perfect,
            wind=WindRealism(steady_speed=5.0, direction_rad=np.deg2rad(45.0)),
        ),
        "C4 turbulence": replace(perfect, wind=WindRealism(turbulence="moderate")),
        "C5 realistic": RealismConfig(
            actuators=ActuatorRealism(False),
            sensors=SensorRealism(False),
            estimator=EstimatorRealism("ekf"),
            wind=WindRealism(
                steady_speed=5.0,
                direction_rad=np.deg2rad(45.0),
                turbulence="moderate",
            ),
        ),
    }
