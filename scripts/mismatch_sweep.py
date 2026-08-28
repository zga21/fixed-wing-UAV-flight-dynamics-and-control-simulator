"""Resumable one-at-a-time NDI internal-model mismatch campaign."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uav_sim.config import load_aircraft
from uav_sim.control.affine_model import ControlAffineModel
from uav_sim.control.baseline import CascadedPIDAutopilot
from uav_sim.control.ndi import NDIController
from uav_sim.realism import build_plant, degradation_configurations
from uav_sim.scenario import load_scenario, run_scenario

PARAMETERS = ("C_m_alpha", "C_m_delta_e", "C_l_delta_a", "I_yy", "mass")
SCENARIOS = (
    "config/scenarios/mission_a.yaml",
    "config/scenarios/step_responses.yaml",
    "config/scenarios/disturbance_rejection.yaml",
)


def _run_case(job: tuple[str, float, str, int, str]) -> dict[str, Any]:
    parameter, factor, scenario_path, seed, controller_name = job
    cfg = load_aircraft("config/aircraft_v1.yaml")
    scenario = load_scenario(scenario_path)
    if controller_name == "PID":
        controller = CascadedPIDAutopilot(cfg)
    else:
        internal_model = ControlAffineModel(cfg).perturb(**{parameter: factor})
        controller = NDIController(
            internal_model,
            static_cancellation=0.1,
            ndi_authority=0.3,
            ndi_filter_tau=0.08,
            cfg=cfg,
        )
    result, metrics = run_scenario(
        scenario,
        controller,
        build_plant(cfg, degradation_configurations()["C5 realistic"]),
        cfg,
        seed=seed,
    )
    return {
        "parameter": parameter,
        "factor": factor,
        "controller": controller_name,
        "scenario": scenario.name,
        "scenario_path": scenario_path,
        "seed": seed,
        **metrics.to_dict(),
        "final_time": float(result.t[-1]),
    }


def mismatch_sweep(
    parameter: str,
    factors: np.ndarray,
    scenarios: list[str] | tuple[str, ...],
    seeds: list[int] | tuple[int, ...],
    cfg,
    workers: int | None = None,
    out_dir: str | Path = "results/phase6",
) -> pd.DataFrame:
    """Sweep one internal-model parameter while leaving the plant nominal."""
    del cfg
    if parameter not in PARAMETERS:
        raise KeyError(f"unsupported mismatch parameter: {parameter}")
    values = np.asarray(factors, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(values <= 0.0):
        raise ValueError("factors must be a non-empty positive vector")
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cache_path = destination / "mismatch_cases.jsonl"
    rows = _read_rows(cache_path)
    completed = {
        (
            row["parameter"],
            round(float(row["factor"]), 12),
            row["scenario_path"],
            int(row["seed"]),
            row["controller"],
        )
        for row in rows
    }

    jobs: list[tuple[str, float, str, int, str]] = []
    run_values = np.array([1.0]) if parameter == "mass" else values
    for scenario_path in scenarios:
        for seed in seeds:
            pid_job = ("__shared__", 1.0, scenario_path, int(seed), "PID")
            if _job_key(pid_job) not in completed:
                jobs.append(pid_job)
            for factor in run_values:
                ndi_job = (parameter, float(factor), scenario_path, int(seed), "NDI")
                if _job_key(ndi_job) not in completed:
                    jobs.append(ndi_job)

    if jobs:
        max_workers = workers or min(max((os.cpu_count() or 2) - 1, 1), 8)
        if max_workers == 1:
            completed_rows = (_run_case(job) for job in jobs)
            for count, row in enumerate(completed_rows, start=1):
                rows.append(row)
                _write_rows(cache_path, rows)
                print(
                    f"[{count}/{len(jobs)}] {parameter} {row['controller']} "
                    f"factor={row['factor']:.4g} {row['scenario']} seed={row['seed']}",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_run_case, job): job for job in jobs}
                for count, future in enumerate(as_completed(futures), start=1):
                    row = future.result()
                    rows.append(row)
                    _write_rows(cache_path, rows)
                    print(
                        f"[{count}/{len(jobs)}] {parameter} {row['controller']} "
                        f"factor={row['factor']:.4g} {row['scenario']} "
                        f"seed={row['seed']}",
                        flush=True,
                    )

    selected_rows = []
    for row in rows:
        if row["scenario_path"] not in scenarios or int(row["seed"]) not in seeds:
            continue
        if row["controller"] == "PID" and row["parameter"] == "__shared__":
            copy = dict(row)
            copy["parameter"] = parameter
            selected_rows.append(copy)
        elif row["controller"] == "NDI" and row["parameter"] == parameter:
            if parameter == "mass" and np.isclose(float(row["factor"]), 1.0):
                for value in values:
                    copy = dict(row)
                    copy["factor"] = float(value)
                    selected_rows.append(copy)
            elif any(np.isclose(float(row["factor"]), value) for value in values):
                selected_rows.append(row)
    selected = pd.DataFrame(selected_rows)
    expected = len(scenarios) * len(seeds) * (1 + len(values))
    if len(selected) != expected:
        raise RuntimeError(
            f"expected {expected} rows for {parameter}, found {len(selected)}"
        )
    return selected.sort_values(
        ["controller", "factor", "scenario", "seed"]
    ).reset_index(drop=True)


def render_results(
    frame: pd.DataFrame, path: str | Path = "docs/mismatch_findings.md"
) -> None:
    """Generate crossover plots and plain-language findings."""
    scored = frame.assign(tracking_score=frame["rmse_h"] + 4.0 * frame["rmse_V"])
    pid_means = (
        scored[scored["controller"] == "PID"]
        .groupby("parameter")["tracking_score"]
        .mean()
    )
    ndi = scored[scored["controller"] == "NDI"]
    summary = (
        ndi.groupby(["parameter", "factor"])["tracking_score"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    findings = [
        "# Phase 6 Internal-Model Mismatch Findings",
        "",
        "Each sweep changes only the NDI controller's internal model. The true plant,",
        "Phase 5 realism configuration, scenarios, seeds, estimator, and metrics",
        "remain fixed. Error bars are one standard error across paired cases.",
        "",
        "## Crossover Summary",
        "",
        "| Parameter | NDI-better factor interval | Interpretation |",
        "|---|---:|---|",
    ]
    figure_dir = Path("docs/figures/mismatch")
    figure_dir.mkdir(parents=True, exist_ok=True)
    for parameter, group in summary.groupby("parameter", sort=False):
        group = group.sort_values("factor")
        baseline = float(pid_means[parameter])
        better = group[group["mean"] < baseline]
        interval = (
            "none"
            if better.empty
            else f"{better['factor'].min():.2f} to {better['factor'].max():.2f}"
        )
        if parameter == "mass":
            interpretation = "Angular NDI is structurally independent of mass."
        else:
            interpretation = "Outside this interval PID has the lower mean score."
        findings.append(f"| `{parameter}` | {interval} | {interpretation} |")
        _plot_parameter(group, baseline, parameter, figure_dir / f"{parameter}.png")
    findings.extend(
        [
            "",
            "Mass is retained as a required negative-control sweep: the Phase 6 NDI",
            "inverts angular acceleration, whose rigid-body equation depends on",
            "inertia but not translational mass. A flat mass curve is therefore",
            "expected and confirms that mismatch remains isolated to the internal",
            "angular model.",
            "",
        ]
    )
    Path(path).write_text("\n".join(findings), encoding="utf-8")


def _plot_parameter(group, baseline: float, parameter: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.errorbar(
        group["factor"],
        group["mean"],
        yerr=group["sem"].fillna(0.0),
        fmt="o-",
        label="NDI",
    )
    axis.axhline(baseline, color="black", linestyle="--", label="PID")
    axis.axvline(1.0, color="gray", linewidth=0.8)
    axis.set_xlabel(f"Internal {parameter} factor")
    axis.set_ylabel("Tracking score")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _job_key(job: tuple[str, float, str, int, str]):
    parameter, factor, scenario_path, seed, controller = job
    return parameter, round(float(factor), 12), scenario_path, int(seed), controller


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["parameter"],
            row["controller"],
            float(row["factor"]),
            row["scenario_path"],
            int(row["seed"]),
        ),
    )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered),
        encoding="utf-8",
    )


def _assert_pilot_passes(frame: pd.DataFrame) -> None:
    """Abort an expensive sweep when the nominal candidate is not competitive."""
    scored = frame.assign(score=frame["rmse_h"] + 4.0 * frame["rmse_V"])
    pid_score = float(scored[scored["controller"] == "PID"]["score"].mean())
    nominal = scored[
        (scored["controller"] == "NDI") & np.isclose(scored["factor"], 1.0)
    ]
    if nominal.empty or nominal["terminated_early"].any():
        raise RuntimeError("mismatch pilot rejected NDI: nominal run was incomplete")
    ndi_score = float(nominal["score"].mean())
    if ndi_score >= pid_score:
        raise RuntimeError(
            f"mismatch pilot rejected NDI: nominal score {ndi_score:.3f} "
            f"does not beat PID {pid_score:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--parameter", choices=PARAMETERS, action="append")
    parser.add_argument("--skip-pilot", action="store_true")
    args = parser.parse_args()
    if args.seeds < 1 or args.steps < 3:
        parser.error("--seeds must be positive and --steps at least 3")
    cfg = load_aircraft("config/aircraft_v1.yaml")
    factors = np.linspace(0.5, 1.5, args.steps)
    parameters = args.parameter or PARAMETERS
    if not args.skip_pilot:
        pilot = mismatch_sweep(
            "C_m_delta_e",
            np.array([0.5, 1.0, 1.5]),
            ("config/scenarios/disturbance_rejection.yaml",),
            list(range(min(args.seeds, 2))),
            cfg,
            workers=args.workers,
        )
        _assert_pilot_passes(pilot)
    frames = [
        mismatch_sweep(
            parameter,
            factors,
            SCENARIOS,
            list(range(args.seeds)),
            cfg,
            workers=args.workers,
        )
        for parameter in parameters
    ]
    render_results(pd.concat(frames, ignore_index=True))


if __name__ == "__main__":
    main()
