"""Plot Phase 3 modal root locus across airspeed."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uav_sim.config import load_aircraft
from uav_sim.linearise import decouple, linearise_euler
from uav_sim.modes import identify_modes
from uav_sim.plant import AircraftPlant
from uav_sim.trim import trim


def collect_modes(
    speeds: list[float] | None = None,
    altitude: float = 100.0,
) -> dict[str, list[complex]]:
    """Return identified eigenvalues keyed by mode name."""
    speeds = [18.0, 20.0, 25.0, 30.0, 35.0] if speeds is None else speeds
    cfg = load_aircraft("config/aircraft_v1.yaml")
    plant = AircraftPlant(cfg)
    values: dict[str, list[complex]] = {}
    for speed in speeds:
        point = trim(speed, 0.0, altitude, cfg, plant)
        A, B = linearise_euler(point, plant, cfg)
        lon, lat = decouple(A, B)
        modes = identify_modes(lon.A, lat.A)
        for name, mode in modes.items():
            values.setdefault(name, []).append(mode.eigenvalue)
    return values


def plot_modes(
    output: str | Path = "docs/figures/phase3/modes_root_locus.png",
) -> Path:
    """Save a root-locus-style modal pole plot."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    modes = collect_modes()
    markers = {
        "short_period": "o",
        "phugoid": "s",
        "dutch_roll": "^",
        "roll_subsidence": "D",
        "spiral": "x",
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    for name, eigenvalues in modes.items():
        roots = np.asarray(eigenvalues, dtype=np.complex128)
        ax.plot(
            roots.real,
            roots.imag,
            marker=markers[name],
            linewidth=1.5,
            label=name.replace("_", " "),
        )
        if np.any(np.abs(roots.imag) > 1e-12):
            ax.plot(
                roots.real,
                -roots.imag,
                marker=markers[name],
                linewidth=1.0,
                alpha=0.45,
            )
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Real(lambda) [1/s]")
    ax.set_ylabel("Imag(lambda) [rad/s]")
    ax.set_title("Phase 3 Modal Pole Map")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    path = plot_modes()
    print(path)


if __name__ == "__main__":
    main()
