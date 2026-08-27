"""Scenario definitions and execution harness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from uav_sim.config import AircraftConfig
from uav_sim.control.base import Controller, Reference
from uav_sim.integrate import simulate
from uav_sim.metrics import Metrics, compute_metrics
from uav_sim.plant import AircraftPlant
from uav_sim.rotations import euler_to_quat
from uav_sim.state import IDX_QUAT
from uav_sim.trim import trim


@dataclass(frozen=True)
class ScenarioInitial:
    """Initial condition requested by a scenario."""

    V: float
    altitude: float
    heading: float = 0.0


@dataclass(frozen=True)
class ScenarioSegment:
    """Piecewise-constant command segment."""

    t: float
    altitude: float
    airspeed: float
    heading: float


@dataclass(frozen=True)
class Scenario:
    """Version-controlled mission definition."""

    name: str
    duration: float
    initial: ScenarioInitial
    segments: tuple[ScenarioSegment, ...]
    path: Path | None = None

    def reference_at(self, t: float) -> Reference:
        """Return the active zero-order-held reference at time ``t``."""
        active = self.segments[0]
        for segment in self.segments:
            if t + 1e-12 >= segment.t:
                active = segment
            else:
                break
        return Reference(
            altitude=active.altitude,
            airspeed=active.airspeed,
            heading=active.heading,
        )

    def reference_history(self, t: np.ndarray) -> dict[str, np.ndarray]:
        """Return arrays of commanded altitude, airspeed, and heading."""
        altitude = np.empty_like(t, dtype=np.float64)
        airspeed = np.empty_like(t, dtype=np.float64)
        heading = np.empty_like(t, dtype=np.float64)
        for idx, time in enumerate(t):
            ref = self.reference_at(float(time))
            altitude[idx] = 0.0 if ref.altitude is None else ref.altitude
            airspeed[idx] = 0.0 if ref.airspeed is None else ref.airspeed
            heading[idx] = 0.0 if ref.heading is None else ref.heading
        return {"altitude": altitude, "airspeed": airspeed, "heading": heading}


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario YAML file."""
    scenario_path = Path(path)
    with scenario_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    return scenario_from_dict(data, scenario_path)


def scenario_from_dict(data: dict[str, Any], path: Path | None = None) -> Scenario:
    """Build a scenario from parsed YAML data."""
    initial_raw = data["initial"]
    initial = ScenarioInitial(
        V=float(initial_raw["V"]),
        altitude=float(initial_raw["altitude"]),
        heading=_heading_from_mapping(initial_raw),
    )
    segments = tuple(
        ScenarioSegment(
            t=float(item["t"]),
            altitude=float(item["altitude"]),
            airspeed=float(item["airspeed"]),
            heading=_heading_from_mapping(item),
        )
        for item in data["segments"]
    )
    if not segments:
        raise ValueError("scenario must contain at least one segment")
    if any(
        next_segment.t <= segment.t
        for segment, next_segment in zip(segments, segments[1:], strict=False)
    ):
        raise ValueError("scenario segment times must be strictly increasing")
    return Scenario(
        name=str(data["name"]),
        duration=float(data["duration"]),
        initial=initial,
        segments=segments,
        path=path,
    )


def run_scenario(
    scenario: Scenario,
    controller: Controller,
    plant: AircraftPlant,
    cfg: AircraftConfig,
    seed: int | None = None,
) -> tuple[Any, Metrics]:
    """Run a scenario deterministically and compute its metrics."""
    controller.reset()
    x0 = _initial_trimmed_state(scenario, cfg, plant)

    def adapter(t: float, x_hat: np.ndarray, _cfg, _rng) -> np.ndarray:
        ref = scenario.reference_at(t)
        return controller.update(x_hat, ref, t, cfg.integration.dt).to_array()

    result = simulate(
        x0,
        adapter,
        scenario.duration,
        cfg,
        plant,
        log_every=50,
        rng=np.random.default_rng(seed),
    )
    metrics = compute_metrics(result, scenario, cfg)
    result.diagnostics["reference"] = scenario.reference_history(result.t)
    result.diagnostics["metrics"] = metrics.to_dict()
    return result, metrics


def _initial_trimmed_state(
    scenario: Scenario,
    cfg: AircraftConfig,
    plant: AircraftPlant,
) -> np.ndarray:
    trim_plant = AircraftPlant(cfg, aero_model=plant.aero_model)
    point = trim(
        scenario.initial.V,
        0.0,
        scenario.initial.altitude,
        cfg,
        trim_plant,
    )
    x0 = point.x.copy()
    x0[IDX_QUAT] = euler_to_quat(0.0, point.theta, scenario.initial.heading)
    return plant.adjust_initial_state(x0)


def _heading_from_mapping(data: dict[str, Any]) -> float:
    if "heading" in data:
        return float(data["heading"])
    if "heading_deg" in data:
        return float(np.deg2rad(data["heading_deg"]))
    return 0.0
