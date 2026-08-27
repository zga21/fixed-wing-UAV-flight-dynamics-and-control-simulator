"""Run and render the Phase 5 C0-C5 degradation ablation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uav_sim.config import config_hash, load_aircraft
from uav_sim.control.baseline import CascadedPIDAutopilot
from uav_sim.realism import build_plant, degradation_configurations
from uav_sim.scenario import load_scenario, run_scenario

SCENARIOS = (
    "config/scenarios/mission_a.yaml",
    "config/scenarios/step_responses.yaml",
    "config/scenarios/disturbance_rejection.yaml",
)
DETERMINISTIC_CONFIGS = frozenset({"C0 perfect", "C1 actuators", "C3 steady wind"})
IDENTITY_COLUMNS = ["configuration", "scenario", "seed"]
NON_AGGREGATE_METRICS = {"success", "terminated_early"}


def _run_case(job: tuple[str, str, int]) -> dict[str, Any]:
    configuration, scenario_path, seed = job
    cfg = load_aircraft("config/aircraft_v1.yaml")
    realism = degradation_configurations()[configuration]
    scenario = load_scenario(scenario_path)
    result, metrics = run_scenario(
        scenario,
        CascadedPIDAutopilot(cfg),
        build_plant(cfg, realism),
        cfg,
        seed=seed,
    )
    row: dict[str, Any] = {
        "configuration": configuration,
        "scenario": scenario.name,
        "scenario_path": scenario_path,
        "seed": seed,
        **metrics.to_dict(),
    }
    row["final_time"] = float(result.t[-1])
    return row


def run_degradation_study(
    controller: Any,
    scenarios: list[str] | tuple[str, ...],
    seeds: list[int] | tuple[int, ...],
    out_dir: str | Path,
    workers: int | None = None,
) -> pd.DataFrame:
    """Run the same scenarios and seeds across the frozen C0-C5 matrix."""
    del controller
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_path = output_dir / "degradation_cases.jsonl"
    existing = _read_cases(case_path)
    completed = {
        (row["configuration"], row["scenario_path"], int(row["seed"]))
        for row in existing
    }
    jobs = []
    for configuration in degradation_configurations():
        case_seeds = (0,) if configuration in DETERMINISTIC_CONFIGS else seeds
        for scenario_path in scenarios:
            for seed in case_seeds:
                key = (configuration, scenario_path, int(seed))
                if key not in completed:
                    jobs.append(key)

    if jobs:
        max_workers = workers or min(max((os.cpu_count() or 2) - 1, 1), 8)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_case, job): job for job in jobs}
            for count, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                existing.append(row)
                _write_cases(case_path, existing)
                print(
                    f"[{count}/{len(jobs)}] {row['configuration']} "
                    f"{row['scenario']} seed={row['seed']}",
                    flush=True,
                )

    expanded = []
    for row in existing:
        if row["configuration"] in DETERMINISTIC_CONFIGS:
            for seed in seeds:
                copy = dict(row)
                copy["seed"] = int(seed)
                expanded.append(copy)
        else:
            expanded.append(row)
    frame = pd.DataFrame(expanded)
    expected = len(degradation_configurations()) * len(scenarios) * len(seeds)
    if len(frame) != expected:
        raise RuntimeError(f"expected {expected} ablation rows, found {len(frame)}")
    return frame.sort_values(IDENTITY_COLUMNS).reset_index(drop=True)


def _read_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_cases(path: Path, rows: list[dict[str, Any]]) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["configuration"],
            row["scenario_path"],
            int(row["seed"]),
        ),
    )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered),
        encoding="utf-8",
    )


def render_results(frame: pd.DataFrame, seeds: list[int] | tuple[int, ...]) -> None:
    metric_columns = [
        column
        for column in frame.columns
        if column not in {*IDENTITY_COLUMNS, "scenario_path", "final_time"}
        and column not in NON_AGGREGATE_METRICS
    ]
    grouped = frame.groupby("configuration", sort=False)
    baseline = grouped[metric_columns].mean().loc["C0 perfect"]
    records = []
    for configuration, group in grouped:
        for metric in metric_columns:
            mean = float(group[metric].mean())
            denominator = abs(float(baseline[metric]))
            percent = (
                None
                if denominator < 1e-12
                else 100.0 * (mean - float(baseline[metric])) / denominator
            )
            records.append(
                {
                    "configuration": configuration,
                    "metric": metric,
                    "mean": mean,
                    "p95": float(group[metric].quantile(0.95)),
                    "change_percent": percent,
                }
            )
    summary = pd.DataFrame(records)
    contributions = _tracking_contributions(frame)
    dominant = max(contributions, key=contributions.get)
    _validate_phase4_baseline(frame)
    _write_markdown(summary, frame, seeds, contributions, dominant)
    _plot_contributions(contributions)
    _write_record(summary, frame, seeds, contributions, dominant)


def _tracking_contributions(frame: pd.DataFrame) -> dict[str, float]:
    scored = frame.assign(tracking_score=frame["rmse_h"] + 4.0 * frame["rmse_V"])
    means = scored.groupby("configuration")["tracking_score"].mean()
    baseline = float(means["C0 perfect"])
    return {
        name: 100.0 * (float(means[name]) - baseline) / baseline
        for name in (
            "C1 actuators",
            "C2 sensors+EKF",
            "C3 steady wind",
            "C4 turbulence",
            "C5 realistic",
        )
    }


def _write_markdown(
    summary: pd.DataFrame,
    frame: pd.DataFrame,
    seeds: list[int] | tuple[int, ...],
    contributions: dict[str, float],
    dominant: str,
) -> None:
    table = summary.copy()
    table["mean"] = table["mean"].map(lambda value: f"{value:.4g}")
    table["p95"] = table["p95"].map(lambda value: f"{value:.4g}")
    table["change_percent"] = table["change_percent"].map(
        lambda value: "n/a (C0=0)" if pd.isna(value) else f"{value:+.1f}%"
    )
    c0 = frame[frame["configuration"] == "C0 perfect"]
    c5 = frame[frame["configuration"] == "C5 realistic"]
    c0_score = float((c0["rmse_h"] + 4.0 * c0["rmse_V"]).mean())
    c5_score = float((c5["rmse_h"] + 4.0 * c5["rmse_V"]).mean())
    dominant_percent = contributions[dominant]
    text = f"""# Phase 5 Degradation Study

Generated from the frozen C0-C5 ablation using all three Phase 4 scenarios and
{len(seeds)} paired seeds per configuration ({len(frame)} effective flights).
Deterministic C0, C1, and C3 cases are simulated once per scenario and repeated
across seed rows; stochastic sensor/EKF and turbulence cases execute every seed.

The combined tracking score is `RMSE_h + 4*RMSE_V`, which gives one metre per
second of airspeed error the same weight as four metres of altitude error. It
increased from {c0_score:.3f} in C0 to {c5_score:.3f} in C5. This is the measured
answer to RQ1: the unchanged Phase 4 controller loses tracking quality when its
actuators, state information, and atmosphere are made realistic.

## Dominant mechanism

**{dominant}** is the largest isolated contribution, changing the tracking score
by {dominant_percent:+.1f}% relative to C0. The mechanism ranking is based on
paired scenarios and seeds, so it separates the individual effects before the
nonlinear interactions in C5. Rate limiting adds phase lag, estimation adds
noise and delay, and Dryden turbulence injects energy around the control-loop
bandwidth; their combined effect is not expected to equal the sum of ablations.

## Mechanism contribution

| Configuration | Tracking-score change |
|---|---:|
"""
    for configuration, value in contributions.items():
        text += f"| {configuration} | {value:+.1f}% |\n"
    text += "\n## All metrics\n\n"
    text += _markdown_table(table)
    Path("docs/degradation_table.md").write_text(text, encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _plot_contributions(contributions: dict[str, float]) -> None:
    output = Path("docs/figures/degradation")
    output.mkdir(parents=True, exist_ok=True)
    labels = [name.split(" ", 1)[1] for name in contributions]
    values = list(contributions.values())
    colors = ["#3B82F6", "#D97706", "#0F766E", "#DC2626", "#6B7280"]
    fig, axis = plt.subplots(figsize=(8.0, 4.8))
    axis.bar(labels, values, color=colors)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("Tracking-score change from C0 [%]")
    axis.set_title("Phase 5 degradation contribution")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "mechanism_contributions.png", dpi=180)
    plt.close(fig)


def _validate_phase4_baseline(frame: pd.DataFrame) -> None:
    record = json.loads(Path("docs/baseline_record.json").read_text(encoding="utf-8"))
    c0 = frame[frame["configuration"] == "C0 perfect"]
    for scenario_name, expected in record["scenarios"].items():
        actual = c0[c0["scenario"] == scenario_name]
        if actual.empty:
            raise RuntimeError(f"C0 is missing Phase 4 scenario {scenario_name}")
        for metric, expected_value in expected.items():
            value = actual.iloc[0][metric]
            if isinstance(expected_value, bool):
                if bool(value) is not expected_value:
                    raise RuntimeError(
                        f"C0 no longer reproduces {scenario_name}.{metric}"
                    )
            elif not np.isclose(
                float(value), float(expected_value), rtol=0.0, atol=0.0
            ):
                raise RuntimeError(f"C0 no longer reproduces {scenario_name}.{metric}")


def _write_record(
    summary: pd.DataFrame,
    frame: pd.DataFrame,
    seeds: list[int] | tuple[int, ...],
    contributions: dict[str, float],
    dominant: str,
) -> None:
    cfg = load_aircraft("config/aircraft_v1.yaml")
    record = {
        "git_commit": _git_commit(),
        "config_hash": config_hash(cfg),
        "scenarios": list(SCENARIOS),
        "seeds": list(seeds),
        "effective_flights": len(frame),
        "dominant_mechanism": dominant,
        "tracking_score_change_percent": contributions,
        "summary": summary.to_dict(orient="records"),
    }
    Path("docs/degradation_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    seeds = list(range(args.seeds))
    frame = run_degradation_study(
        None, SCENARIOS, seeds, "results/phase5", workers=args.workers
    )
    render_results(frame, seeds)
    print("docs/degradation_table.md")


if __name__ == "__main__":
    main()
