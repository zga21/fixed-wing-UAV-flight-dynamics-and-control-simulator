"""Regenerate the Phase 4 baseline record and scenario figures."""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.config import config_hash, load_aircraft
from uav_sim.control.baseline import CascadedPIDAutopilot
from uav_sim.plant import AircraftPlant
from uav_sim.rotations import quat_to_euler
from uav_sim.scenario import load_scenario, run_scenario
from uav_sim.state import IDX_QUAT, IDX_VEL

SCENARIOS = [
    "config/scenarios/mission_a.yaml",
    "config/scenarios/step_responses.yaml",
    "config/scenarios/disturbance_rejection.yaml",
]


def build_record() -> dict[str, object]:
    """Run all baseline scenarios and return a JSON-ready record."""
    cfg = load_aircraft("config/aircraft_v1.yaml")
    plant = AircraftPlant(cfg)
    controller = CascadedPIDAutopilot(cfg)
    scenario_records = {}
    for scenario_path in SCENARIOS:
        scenario = load_scenario(scenario_path)
        result, metrics = run_scenario(scenario, controller, plant, cfg, seed=0)
        _plot_scenario(result, scenario)
        scenario_records[scenario.name] = metrics.to_dict()
    return {
        "date": date.today().isoformat(),
        "generated_utc": date.today().isoformat(),
        "git_commit": _git_commit(),
        "config_hash": config_hash(cfg),
        "controller": "CascadedPIDAutopilot",
        "gain_names": controller.gain_names,
        "gains": controller.gains.tolist(),
        "scenarios": scenario_records,
    }


def write_record(path: str | Path = "docs/baseline_record.json") -> Path:
    """Write the committed baseline record."""
    record_path = Path(path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record = build_record()
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record_path


def _plot_scenario(result, scenario) -> Path:
    output_dir = Path("docs/figures/baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    refs = scenario.reference_history(result.t)
    altitude = -result.x[:, 2]
    airspeed = np.linalg.norm(result.x[:, IDX_VEL], axis=1)
    euler = np.asarray([quat_to_euler(row[IDX_QUAT]) for row in result.x])
    path = output_dir / f"{scenario.name}.png"
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=True)
    axes[0].plot(result.t, altitude, label="actual")
    axes[0].plot(result.t, refs["altitude"], "--", label="command")
    axes[0].set_ylabel("Altitude [m]")
    axes[1].plot(result.t, airspeed, label="actual")
    axes[1].plot(result.t, refs["airspeed"], "--", label="command")
    axes[1].set_ylabel("Airspeed [m/s]")
    axes[2].plot(result.t, np.rad2deg(euler[:, 2]), label="heading")
    axes[2].plot(result.t, np.rad2deg(refs["heading"]), "--", label="command")
    axes[2].set_ylabel("Heading [deg]")
    axes[2].set_xlabel("Time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")
    fig.suptitle(scenario.name)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    path = write_record()
    print(path)


if __name__ == "__main__":
    main()
