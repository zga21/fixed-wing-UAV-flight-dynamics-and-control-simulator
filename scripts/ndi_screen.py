"""Fast noisy inner-loop screen before expensive Phase 6 mission runs."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import cache
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from uav_sim.air_data import compute_air_data
from uav_sim.config import load_aircraft
from uav_sim.control.affine_model import ControlAffineModel
from uav_sim.control.baseline import CascadedPIDAutopilot
from uav_sim.control.ndi import NDIController
from uav_sim.control.rate_loop import ActuatorLimits, RateLoop
from uav_sim.realism import build_plant, degradation_configurations
from uav_sim.scenario import load_scenario, run_scenario
from uav_sim.state import IDX_OMEGA, State, initial_state

SCREEN_DT = 0.005
SCREEN_DURATION = 4.0
OUTER_SCREEN_VERSION = "c2_30s_v2_augmentation"
OUTER_SCREEN_CACHE = "results/phase6/ndi_outer_screen.jsonl"


def screen_candidate(
    static_cancellation: float,
    ndi_authority: float,
    filter_tau: float,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, float]:
    """Score one NDI candidate against PID in a noisy angular surrogate."""
    ndi_scores = []
    pid_scores = []
    for seed in seeds:
        ndi_scores.append(
            _simulate_inner_loop(
                "NDI",
                seed,
                static_cancellation,
                ndi_authority,
                filter_tau,
            )
        )
        pid_scores.append(_pid_score(seed))
    ndi_mean = float(np.mean(ndi_scores))
    pid_mean = float(np.mean(pid_scores))
    return {
        "static_cancellation": static_cancellation,
        "ndi_authority": ndi_authority,
        "filter_tau": filter_tau,
        "ndi_score": ndi_mean,
        "pid_score": pid_mean,
        "relative_score": ndi_mean / pid_mean,
    }


def screen_grid() -> pd.DataFrame:
    """Rank a compact grid; lower relative score is better."""
    candidates = product((0.0, 0.1, 0.2), (0.3, 0.5, 0.7), (0.02, 0.05, 0.08))
    rows = [screen_candidate(*candidate) for candidate in candidates]
    return pd.DataFrame(rows).sort_values("relative_score").reset_index(drop=True)


def screen_outer_loop(workers: int | None = None) -> pd.DataFrame:
    """Rank a compact shortlist in the real nonlinear/EKF pipeline."""
    candidates = [(-1.0, -1.0, -1.0)]
    candidates += list(product((0.05, 0.1), (0.05, 0.1, 0.2), (0.05, 0.08)))
    cache_path = Path(OUTER_SCREEN_CACHE)
    rows = _read_screen_rows(cache_path)
    completed = {
        (
            float(row["static_cancellation"]),
            float(row["ndi_authority"]),
            float(row["filter_tau"]),
        )
        for row in rows
        if row.get("screen_version") == OUTER_SCREEN_VERSION
    }
    jobs = [candidate for candidate in candidates if candidate not in completed]
    max_workers = workers or min(max((os.cpu_count() or 2) - 1, 1), 8)
    if max_workers == 1:
        for job in jobs:
            rows.append(_outer_loop_job(job))
            _write_screen_rows(cache_path, rows)
    elif jobs:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_outer_loop_job, job): job for job in jobs}
            for future in as_completed(futures):
                rows.append(future.result())
                _write_screen_rows(cache_path, rows)
    selected = [
        row
        for row in rows
        if row.get("screen_version") == OUTER_SCREEN_VERSION
        and (
            float(row["static_cancellation"]),
            float(row["ndi_authority"]),
            float(row["filter_tau"]),
        )
        in candidates
    ]
    return pd.DataFrame(selected).sort_values("score").reset_index(drop=True)


def _outer_loop_job(
    candidate: tuple[float, float, float],
) -> dict[str, float | str]:
    static_cancellation, ndi_authority, filter_tau = candidate
    cfg = load_aircraft("config/aircraft_v1.yaml")
    scenario = replace(
        load_scenario("config/scenarios/disturbance_rejection.yaml"), duration=30.0
    )
    is_pid = static_cancellation < 0.0
    controller = (
        CascadedPIDAutopilot(cfg)
        if is_pid
        else NDIController(
            ControlAffineModel(cfg),
            static_cancellation=static_cancellation,
            ndi_authority=ndi_authority,
            ndi_filter_tau=filter_tau,
            pid_retention=1.0,
            cfg=cfg,
        )
    )
    result, metrics = run_scenario(
        scenario,
        controller,
        build_plant(cfg, degradation_configurations()["C2 sensors+EKF"]),
        cfg,
        seed=0,
    )
    score = metrics.rmse_h + 4.0 * metrics.rmse_V
    if result.terminated_early:
        score += 1.0e3
    return {
        "screen_version": OUTER_SCREEN_VERSION,
        "controller": "PID" if is_pid else "NDI",
        "static_cancellation": static_cancellation,
        "ndi_authority": ndi_authority,
        "filter_tau": filter_tau,
        "score": score,
        "rmse_h": metrics.rmse_h,
        "rmse_V": metrics.rmse_V,
        "terminated": float(result.terminated_early),
    }


def _read_screen_rows(path: Path) -> list[dict[str, float | str]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_screen_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _simulate_inner_loop(
    controller_name: str,
    seed: int,
    static_cancellation: float,
    ndi_authority: float,
    filter_tau: float,
) -> float:
    cfg = load_aircraft("config/aircraft_v1.yaml")
    model = ControlAffineModel(cfg)
    x = initial_state(25.0, alpha=0.05, h=100.0)
    state = State.from_array(x)
    true_air = compute_air_data(state.vel_b, state.quat, state.altitude)
    trim_surfaces = np.linalg.solve(model.g(true_air), -model.f(x, true_air))
    actual_surfaces = trim_surfaces.copy()
    rng = np.random.default_rng(seed)
    gyro_bias = rng.normal(0.0, 0.003, 3)
    alpha_bias = rng.normal(0.0, 0.005)
    beta_bias = rng.normal(0.0, 0.008)
    ndi = NDIController(
        model,
        static_cancellation=static_cancellation,
        ndi_authority=ndi_authority,
        ndi_filter_tau=filter_tau,
        pid_retention=1.0,
        cfg=cfg,
    )
    ndi._last_trim_controls[:3] = trim_surfaces
    pid = RateLoop(limits=ActuatorLimits.from_config(cfg))
    squared_error = 0.0
    effort = 0.0
    steps = round(SCREEN_DURATION / SCREEN_DT)
    for step in range(steps):
        command = _command_at(step * SCREEN_DT)
        x_est = x.copy()
        x_est[IDX_OMEGA] += gyro_bias + rng.normal(0.0, 0.01, 3)
        air_est = replace(
            true_air,
            V=max(true_air.V + rng.normal(0.0, 0.5), 1.0),
            alpha=true_air.alpha + alpha_bias + rng.normal(0.0, 0.005),
            beta=true_air.beta + beta_bias + rng.normal(0.0, 0.008),
        )
        if controller_name == "NDI":
            increments = ndi._rate_surfaces(command, x_est, air_est, SCREEN_DT)
        else:
            increments = pid.update(command, x_est[IDX_OMEGA], SCREEN_DT)
        requested = trim_surfaces + increments
        lag_fraction = SCREEN_DT / (cfg.actuators.aileron.tau + SCREEN_DT)
        actual_surfaces += lag_fraction * (requested - actual_surfaces)
        x[IDX_OMEGA] += SCREEN_DT * (
            model.f(x, true_air) + model.g(true_air) @ actual_surfaces
        )
        squared_error += float(np.sum((command - x[IDX_OMEGA]) ** 2))
        effort += float(np.sum(increments**2))
        if not np.all(np.isfinite(x[IDX_OMEGA])) or np.max(np.abs(x[IDX_OMEGA])) > 5:
            return 1.0e6
    return squared_error / steps + 0.01 * effort / steps


@cache
def _pid_score(seed: int) -> float:
    return _simulate_inner_loop("PID", seed, 0.0, 0.0, 0.0)


def _command_at(t: float) -> np.ndarray:
    if 0.5 <= t < 1.5:
        return np.array([0.2, 0.0, 0.0])
    if 1.5 <= t < 2.5:
        return np.array([0.0, 0.1, 0.0])
    if 2.5 <= t < 3.5:
        return np.array([0.0, 0.0, 0.1])
    return np.zeros(3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--outer", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    ranking = screen_outer_loop(args.workers) if args.outer else screen_grid()
    print(ranking.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
