"""Mission metric computation frozen for Phase 4 onward."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from uav_sim.air_data import compute_air_data
from uav_sim.config import AircraftConfig, load_aircraft
from uav_sim.control.outer_loop import wrap_angle
from uav_sim.result import SimResult
from uav_sim.rotations import quat_to_euler
from uav_sim.state import IDX_QUAT, IDX_VEL

if TYPE_CHECKING:
    from uav_sim.scenario import Scenario


SUCCESS_CRITERIA_FROZEN_DATE = "2026-08-26"
SUCCESS_CRITERIA = {
    "rmse_h": 12.0,
    "rmse_V": 3.0,
    "max_phi_deg": 60.0,
    "saturation_fraction": 0.20,
    "envelope_violations": 0,
}


@dataclass(frozen=True)
class Metrics:
    """Flat scenario metrics used by later optimisation and campaigns."""

    rmse_h: float
    rmse_V: float
    rmse_phi: float
    rmse_theta: float
    max_h_error: float
    max_V_error: float
    overshoot_h: float
    settling_time_h: float
    control_effort: float
    control_rate_effort: float
    saturation_fraction: float
    envelope_violations: int
    max_load_factor: float
    disturbance_recovery_time: float
    integration_failures: int
    terminated_early: bool
    success: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a flat JSON/Parquet-ready dictionary."""
        return asdict(self)


def compute_metrics(
    result: SimResult,
    scenario: Scenario,
    cfg: AircraftConfig | None = None,
) -> Metrics:
    """Compute frozen Phase 4 mission metrics.

    Settling time is measured from the final command change to the last sample
    whose altitude error exceeds 5 percent of that final altitude step, with a
    1 m floor. Piecewise-constant command segments are considered active for the
    full interval after their start time.
    """
    cfg = load_aircraft("config/aircraft_v1.yaml") if cfg is None else cfg
    refs = scenario.reference_history(result.t)
    altitude = -result.x[:, 2]
    airspeed = np.asarray(
        result.diagnostics.get("V", np.linalg.norm(result.x[:, IDX_VEL], axis=1)),
        dtype=np.float64,
    )
    euler = np.asarray([quat_to_euler(row[IDX_QUAT]) for row in result.x])
    phi = euler[:, 0]
    theta = euler[:, 1]
    h_error = altitude - refs["altitude"]
    V_error = airspeed - refs["airspeed"]
    phi_error = phi
    theta_error = theta
    control_effort = _control_effort(result)
    control_rate_effort = _control_rate_effort(result)
    envelope_violations, max_load_factor = _envelope_metrics(result, cfg)
    saturation_fraction = _saturation_fraction(result, cfg)

    max_phi_deg = float(np.rad2deg(np.max(np.abs(phi)))) if phi.size else 0.0
    success = (
        _rmse(h_error) <= SUCCESS_CRITERIA["rmse_h"]
        and _rmse(V_error) <= SUCCESS_CRITERIA["rmse_V"]
        and max_phi_deg <= SUCCESS_CRITERIA["max_phi_deg"]
        and saturation_fraction <= SUCCESS_CRITERIA["saturation_fraction"]
        and envelope_violations <= SUCCESS_CRITERIA["envelope_violations"]
        and not result.terminated_early
    )
    return Metrics(
        rmse_h=_rmse(h_error),
        rmse_V=_rmse(V_error),
        rmse_phi=_rmse(phi_error),
        rmse_theta=_rmse(theta_error),
        max_h_error=float(np.max(np.abs(h_error))) if h_error.size else 0.0,
        max_V_error=float(np.max(np.abs(V_error))) if V_error.size else 0.0,
        overshoot_h=_altitude_overshoot(altitude, scenario),
        settling_time_h=_settling_time(result.t, h_error, scenario),
        control_effort=control_effort,
        control_rate_effort=control_rate_effort,
        saturation_fraction=saturation_fraction,
        envelope_violations=envelope_violations,
        max_load_factor=max_load_factor,
        disturbance_recovery_time=0.0,
        integration_failures=int(result.terminated_early),
        terminated_early=bool(result.terminated_early),
        success=success,
    )


def _rmse(error: np.ndarray) -> float:
    if error.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(error))))


def _control_effort(result: SimResult) -> float:
    if result.t.size < 2:
        return 0.0
    weights = np.array([1.0, 1.0, 1.0, 0.2], dtype=np.float64)
    effort = np.sum(np.square(result.u_actual) * weights, axis=1)
    return float(np.trapezoid(effort, result.t))


def _control_rate_effort(result: SimResult) -> float:
    if result.t.size < 2:
        return 0.0
    dt = np.diff(result.t)
    du = np.diff(result.u_actual, axis=0)
    rate = du / dt[:, None]
    effort = np.sum(np.square(rate), axis=1)
    return float(np.trapezoid(effort, result.t[1:]))


def _altitude_overshoot(altitude: np.ndarray, scenario: Scenario) -> float:
    if altitude.size == 0 or len(scenario.segments) < 2:
        return 0.0
    start = scenario.segments[0].altitude
    final = scenario.segments[-1].altitude
    if final >= start:
        return max(float(np.max(altitude) - final), 0.0)
    return max(float(final - np.min(altitude)), 0.0)


def _settling_time(t: np.ndarray, h_error: np.ndarray, scenario: Scenario) -> float:
    if t.size == 0:
        return 0.0
    final_change = scenario.segments[-1].t
    previous = (
        scenario.segments[-2].altitude
        if len(scenario.segments) >= 2
        else scenario.segments[0].altitude
    )
    final = scenario.segments[-1].altitude
    tolerance = max(0.05 * abs(final - previous), 1.0)
    mask = t >= final_change
    if not np.any(mask):
        return float("inf")
    local_t = t[mask]
    local_error = np.abs(h_error[mask])
    unsettled = np.nonzero(local_error > tolerance)[0]
    if unsettled.size == 0:
        return 0.0
    last_unsettled_t = local_t[unsettled[-1]]
    return float(last_unsettled_t - final_change)


def _saturation_fraction(result: SimResult, cfg: AircraftConfig) -> float:
    if "actuator_saturated" in result.diagnostics:
        return float(np.mean(result.diagnostics["actuator_saturated"]))
    if result.u_actual.size == 0:
        return 0.0
    controls = result.u_actual[:, 0:3]
    limits = np.array(
        [
            [cfg.actuators.aileron.min_rad, cfg.actuators.aileron.max_rad],
            [cfg.actuators.elevator.min_rad, cfg.actuators.elevator.max_rad],
            [cfg.actuators.rudder.min_rad, cfg.actuators.rudder.max_rad],
        ],
        dtype=np.float64,
    )
    near_min = np.isclose(controls, limits[:, 0], atol=1e-6)
    near_max = np.isclose(controls, limits[:, 1], atol=1e-6)
    return float(np.mean(np.any(near_min | near_max, axis=1)))


def _envelope_metrics(result: SimResult, cfg: AircraftConfig) -> tuple[int, float]:
    violations = 0
    max_load_factor = 1.0
    logged_V = result.diagnostics.get("V")
    logged_alpha = result.diagnostics.get("alpha")
    logged_beta = result.diagnostics.get("beta")
    for index, row in enumerate(result.x):
        altitude = -row[2]
        air = compute_air_data(row[IDX_VEL], row[IDX_QUAT], altitude)
        V = air.V if logged_V is None else float(logged_V[index])
        alpha = air.alpha if logged_alpha is None else float(logged_alpha[index])
        beta = air.beta if logged_beta is None else float(logged_beta[index])
        phi, _theta, _psi = quat_to_euler(row[IDX_QUAT])
        if not (cfg.envelope.V_min <= V <= cfg.envelope.V_max):
            violations += 1
        if not (cfg.envelope.h_min <= altitude <= cfg.envelope.h_max):
            violations += 1
        if not (cfg.envelope.alpha_min_rad <= alpha <= cfg.envelope.alpha_max_rad):
            violations += 1
        if abs(beta) > cfg.envelope.beta_max_rad:
            violations += 1
        if abs(phi) > cfg.envelope.phi_max_rad:
            violations += 1
        load = 1.0 / max(np.cos(phi), 0.25)
        max_load_factor = max(max_load_factor, float(load))
        if not (cfg.envelope.load_factor_min <= load <= cfg.envelope.load_factor_max):
            violations += 1
    return violations, max_load_factor


def heading_rmse(actual: np.ndarray, commanded: np.ndarray) -> float:
    """Compute wrapped heading RMSE."""
    return _rmse(
        np.array(
            [wrap_angle(a - c) for a, c in zip(actual, commanded, strict=True)],
            dtype=np.float64,
        )
    )
