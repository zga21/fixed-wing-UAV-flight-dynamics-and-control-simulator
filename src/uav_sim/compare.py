"""Paired, reproducible controller-comparison harness."""

from __future__ import annotations

import copy
import subprocess
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uav_sim.config import AircraftConfig, config_hash
from uav_sim.control.base import Controller
from uav_sim.scenario import Scenario, run_scenario

IDENTITY_COLUMNS = ["controller", "scenario", "seed"]


def compare_controllers(
    controllers: Mapping[str, Controller],
    scenarios: Sequence[Scenario],
    seeds: Sequence[int],
    plant_factory: Callable[[], Any],
    cfg: AircraftConfig,
    workers: int = 1,
) -> pd.DataFrame:
    """Run every controller on identical scenario/seed pairs."""
    if len(controllers) < 2:
        raise ValueError("at least two controllers are required for comparison")
    if not scenarios or not seeds:
        raise ValueError("comparison requires at least one scenario and seed")
    if workers < 1:
        raise ValueError("workers must be at least one")
    commit = _git_commit()
    cfg_hash = config_hash(cfg)
    jobs = [
        (name, controller, scenario, int(seed))
        for name, controller in controllers.items()
        for scenario in scenarios
        for seed in seeds
    ]

    def run(job: tuple[str, Controller, Scenario, int]) -> dict[str, Any]:
        name, prototype, scenario, seed = job
        controller = copy.deepcopy(prototype)
        plant = plant_factory()
        result, metrics = run_scenario(scenario, controller, plant, cfg, seed=seed)
        row: dict[str, Any] = {
            "controller": name,
            "scenario": scenario.name,
            "scenario_path": None if scenario.path is None else str(scenario.path),
            "seed": seed,
            "git_commit": commit,
            "config_hash": cfg_hash,
            **metrics.to_dict(),
            "final_time": float(result.t[-1]),
        }
        return row

    if workers == 1:
        rows = [run(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(run, jobs))
    frame = pd.DataFrame(rows).sort_values(IDENTITY_COLUMNS).reset_index(drop=True)
    assert_fair_comparison(frame)
    return frame


def assert_fair_comparison(results: pd.DataFrame) -> None:
    """Raise when controller rows cannot support a paired comparison."""
    required = {*IDENTITY_COLUMNS, "git_commit", "config_hash"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"comparison results missing columns: {sorted(missing)}")
    if results.empty:
        raise ValueError("comparison results are empty")
    if results.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("comparison contains duplicate controller/scenario/seed runs")
    if results["git_commit"].nunique(dropna=False) != 1:
        raise ValueError("controller runs use different git commits")
    if results["config_hash"].nunique(dropna=False) != 1:
        raise ValueError("controller runs use different aircraft configurations")

    expected_pairs: set[tuple[str, int]] | None = None
    for _name, group in results.groupby("controller", sort=False):
        pairs = set(zip(group["scenario"], group["seed"], strict=True))
        if expected_pairs is None:
            expected_pairs = pairs
        elif pairs != expected_pairs:
            raise ValueError("controllers were run on different scenario/seed pairs")


def paired_differences(
    results: pd.DataFrame,
    metric: str = "tracking_score",
    reference: str = "PID",
) -> pd.DataFrame:
    """Return controller-minus-reference differences for each paired case."""
    assert_fair_comparison(results)
    frame = results.copy()
    if metric == "tracking_score" and metric not in frame:
        frame[metric] = frame["rmse_h"] + 4.0 * frame["rmse_V"]
    if metric not in frame:
        raise KeyError(metric)
    pivot = frame.pivot(index=["scenario", "seed"], columns="controller", values=metric)
    if reference not in pivot:
        raise KeyError(f"reference controller {reference!r} is absent")
    differences = pivot.subtract(pivot[reference], axis=0).drop(columns=reference)
    return differences.reset_index()


def comparison_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise paired tracking and success metrics by controller."""
    assert_fair_comparison(results)
    frame = results.assign(tracking_score=results["rmse_h"] + 4.0 * results["rmse_V"])
    return (
        frame.groupby("controller", sort=False)
        .agg(
            flights=("seed", "size"),
            success_rate=("success", "mean"),
            tracking_score_mean=("tracking_score", "mean"),
            tracking_score_p95=("tracking_score", lambda values: values.quantile(0.95)),
            control_effort_mean=("control_effort", "mean"),
            saturation_fraction_mean=("saturation_fraction", "mean"),
        )
        .reset_index()
    )


def plot_paired_differences(
    results: pd.DataFrame,
    path: str | Path,
    reference: str = "PID",
) -> Path:
    """Write a paired tracking-score difference plot."""
    differences = paired_differences(results, reference=reference)
    value_columns = [
        column for column in differences.columns if column not in {"scenario", "seed"}
    ]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    positions = np.arange(len(differences))
    for controller in value_columns:
        axis.plot(positions, differences[controller], "o-", label=controller)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Paired scenario/seed case")
    axis.set_ylabel(f"Tracking score minus {reference}")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    return destination


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
