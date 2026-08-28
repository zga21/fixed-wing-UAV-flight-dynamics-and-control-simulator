"""Run and record the provisional Phase 6 PID-versus-NDI comparison."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uav_sim.air_data import compute_air_data
from uav_sim.compare import (
    assert_fair_comparison,
    comparison_summary,
    plot_paired_differences,
)
from uav_sim.config import config_hash, load_aircraft
from uav_sim.control.affine_model import ControlAffineModel
from uav_sim.control.baseline import CascadedPIDAutopilot
from uav_sim.control.ndi import NDIController
from uav_sim.control.rate_loop import ActuatorLimits, RateLoop
from uav_sim.realism import build_plant, degradation_configurations
from uav_sim.scenario import load_scenario, run_scenario
from uav_sim.state import IDX_OMEGA, State, initial_state

SCENARIO_PATHS = (
    "config/scenarios/mission_a.yaml",
    "config/scenarios/step_responses.yaml",
    "config/scenarios/disturbance_rejection.yaml",
)
CONTROLLER_VERSION = "robust_ndi_v4_screened"
CACHE_PATH = Path("results/phase6/comparison_cases.jsonl")


def _run_case(job: tuple[str, str, int, str, str]) -> dict[str, Any]:
    controller_name, scenario_path, seed, commit, cfg_hash = job
    cfg = load_aircraft("config/aircraft_v1.yaml")
    realism = degradation_configurations()["C5 realistic"]
    controller = (
        CascadedPIDAutopilot(cfg)
        if controller_name == "PID"
        else NDIController(
            ControlAffineModel(cfg),
            static_cancellation=0.1,
            ndi_authority=0.3,
            ndi_filter_tau=0.08,
            cfg=cfg,
        )
    )
    scenario = load_scenario(scenario_path)
    result, metrics = run_scenario(
        scenario,
        controller,
        build_plant(cfg, realism),
        cfg,
        seed=seed,
    )
    return {
        "controller_version": CONTROLLER_VERSION,
        "controller": controller_name,
        "scenario": scenario.name,
        "scenario_path": scenario_path,
        "seed": seed,
        "git_commit": commit,
        "config_hash": cfg_hash,
        **metrics.to_dict(),
        "final_time": float(result.t[-1]),
        "termination_reason": result.termination_reason,
    }


def run_comparison(
    seeds: list[int], workers: int | None = None, allow_dirty: bool = False
) -> pd.DataFrame:
    """Run or resume paired Phase 5-realistic controller cases."""
    commit, dirty = _git_state()
    if dirty and not allow_dirty:
        raise RuntimeError(
            "final comparison requires a clean worktree; use --allow-dirty only "
            "for development smoke runs"
        )
    cfg = load_aircraft("config/aircraft_v1.yaml")
    cfg_hash = config_hash(cfg)
    rows = _read_rows(CACHE_PATH)
    completed = {
        (row["controller"], row["scenario_path"], int(row["seed"]))
        for row in rows
        if row.get("controller_version") == CONTROLLER_VERSION
        and row.get("git_commit") == commit
        and row.get("config_hash") == cfg_hash
    }
    jobs = [
        (controller, path, seed, commit, cfg_hash)
        for controller in ("PID", "NDI")
        for path in SCENARIO_PATHS
        for seed in seeds
        if (controller, path, seed) not in completed
    ]
    if jobs:
        max_workers = workers or min(max((os.cpu_count() or 2) - 1, 1), 8)
        if max_workers == 1:
            for count, job in enumerate(jobs, start=1):
                row = _run_case(job)
                rows.append(row)
                _write_rows(CACHE_PATH, rows)
                print(f"[{count}/{len(jobs)}] {_case_label(row)}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_run_case, job): job for job in jobs}
                for count, future in enumerate(as_completed(futures), start=1):
                    row = future.result()
                    rows.append(row)
                    _write_rows(CACHE_PATH, rows)
                    print(f"[{count}/{len(jobs)}] {_case_label(row)}", flush=True)

    selected = pd.DataFrame(
        [
            row
            for row in rows
            if row.get("controller_version") == CONTROLLER_VERSION
            and row.get("git_commit") == commit
            and row.get("config_hash") == cfg_hash
            and row["scenario_path"] in SCENARIO_PATHS
            and int(row["seed"]) in seeds
        ]
    ).sort_values(["controller", "scenario", "seed"])
    expected = 2 * len(SCENARIO_PATHS) * len(seeds)
    if len(selected) != expected:
        raise RuntimeError(f"expected {expected} paired rows, found {len(selected)}")
    assert_fair_comparison(selected)
    selected.attrs["dirty_worktree"] = dirty
    return selected.reset_index(drop=True)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()
    return commit, bool(status)


def _case_label(row: dict[str, Any]) -> str:
    return f"{row['controller']} {row['scenario']} seed={row['seed']}"


def write_evidence(frame, seeds: list[int]) -> None:
    """Write machine-readable and human-readable comparison evidence."""
    output = Path("results/phase6")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_json(output / "comparison_cases.jsonl", orient="records", lines=True)
    summary = comparison_summary(frame)
    plot_paired_differences(
        frame, "docs/figures/phase6/paired_tracking_differences.png"
    )
    plot_speed_independence("docs/figures/phase6/rate_speed_independence.png")
    record = {
        "git_commit": str(frame["git_commit"].iloc[0]),
        "config_hash": str(frame["config_hash"].iloc[0]),
        "seeds": seeds,
        "scenarios": list(SCENARIO_PATHS),
        "realism": "C5 realistic",
        "effective_flights": len(frame),
        "provisional_until_phase7": True,
        "dirty_worktree": bool(frame.attrs.get("dirty_worktree", False)),
        "controller_version": CONTROLLER_VERSION,
        "summary": summary.to_dict(orient="records"),
    }
    Path("docs/phase6_comparison_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = "\n".join(
        "| {controller} | {flights:d} | {success_rate:.1%} | "
        "{tracking_score_mean:.3f} | {tracking_score_p95:.3f} | "
        "{control_effort_mean:.3f} | {saturation_fraction_mean:.2%} |".format(**row)
        for row in summary.to_dict(orient="records")
    )
    Path("docs/phase6_comparison.md").write_text(
        f"""# Phase 6 Provisional Controller Comparison

Both controllers flew the same three scenarios under the Phase 5 `C5 realistic`
configuration with {len(seeds)} paired seeds ({len(frame)} total flights). Lower
tracking score is better; it is the frozen `RMSE_h + 4*RMSE_V` score used by the
Phase 5 degradation study.

| Controller | Flights | Success | Mean tracking | P95 tracking | Effort | Saturation |
|---|---:|---:|---:|---:|---:|---:|
{rows}

This hand-tuned comparison is provisional. Phase 7 must give both controllers
the same optimisation budget before making a final architecture claim.
""",
        encoding="utf-8",
    )


def plot_speed_independence(path: str | Path) -> Path:
    """Plot PID and NDI roll-rate responses at two airspeeds."""
    cfg = load_aircraft("config/aircraft_v1.yaml")
    figure_path = Path(path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(1, round(2.0 / cfg.integration.dt) + 1) * cfg.integration.dt
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for axis, controller_name in zip(axes, ("PID", "NDI"), strict=True):
        for speed in (18.0, 35.0):
            response = _inner_rate_response(controller_name, speed, cfg)
            axis.plot(time, response, label=f"{speed:.0f} m/s")
        axis.axhline(0.2, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(controller_name)
        axis.set_xlabel("Time [s]")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_ylabel("Roll rate [rad/s]")
    fig.suptitle("Inner-loop speed independence")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return figure_path


def _inner_rate_response(controller_name: str, speed: float, cfg) -> np.ndarray:
    model = ControlAffineModel(cfg)
    x = initial_state(speed, alpha=0.05, h=100.0)
    state = State.from_array(x)
    air = compute_air_data(state.vel_b, state.quat, state.altitude)
    command = np.array([0.2, 0.0, 0.0])
    trim_surfaces = np.linalg.solve(model.g(air), -model.f(x, air))
    limits = np.array(
        [
            [cfg.actuators.aileron.min_rad, cfg.actuators.aileron.max_rad],
            [cfg.actuators.elevator.min_rad, cfg.actuators.elevator.max_rad],
            [cfg.actuators.rudder.min_rad, cfg.actuators.rudder.max_rad],
        ]
    )
    ndi = NDIController(model, K_I=(0.0, 0.0, 0.0), cfg=cfg)
    pid = RateLoop(limits=ActuatorLimits.from_config(cfg))
    history = []
    for _ in range(round(2.0 / cfg.integration.dt)):
        if controller_name == "NDI":
            nu = ndi._virtual_control(command, x[IDX_OMEGA], cfg.integration.dt)
            surfaces = ndi._invert(nu, x, air).delta
        else:
            increments = pid.update(command, x[IDX_OMEGA], cfg.integration.dt)
            surfaces = np.clip(trim_surfaces + increments, limits[:, 0], limits[:, 1])
        x[IDX_OMEGA] += cfg.integration.dt * (model.f(x, air) + model.g(air) @ surfaces)
        history.append(float(x[IDX_OMEGA][0]))
    return np.asarray(history)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    seeds = list(range(args.seeds))
    frame = run_comparison(seeds, workers=args.workers, allow_dirty=args.allow_dirty)
    write_evidence(frame, seeds)
    print(comparison_summary(frame).to_string(index=False))


if __name__ == "__main__":
    main()
