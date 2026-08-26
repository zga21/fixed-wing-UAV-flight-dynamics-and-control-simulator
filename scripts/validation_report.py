"""Generate Phase 3 validation evidence figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_modes import plot_modes

from uav_sim.config import load_aircraft
from uav_sim.integrate import rk4_step, simulate
from uav_sim.linearise import euler_state_from_full, linearise_euler
from uav_sim.plant import AircraftPlant
from uav_sim.trim import trim, trim_grid

OUTPUT_DIR = Path("docs/figures/phase3")


def _doublet_controls(trim_u: np.ndarray, amplitude: float, t: float) -> np.ndarray:
    controls = trim_u.copy()
    if t < 1.0:
        controls[1] += amplitude
    elif t < 2.0:
        controls[1] -= amplitude
    return controls


def _constant_controller(controls: np.ndarray):
    def controller(_t, _x, _cfg, _rng):
        return controls

    return controller


def _doublet_controller(trim_u: np.ndarray, amplitude: float):
    def controller(t, _x, _cfg, _rng):
        return _doublet_controls(trim_u, amplitude, t)

    return controller


def _linear_doublet(
    A: np.ndarray,
    B: np.ndarray,
    trim_u: np.ndarray,
    amplitude: float,
    cfg,
    duration: float,
) -> np.ndarray:
    dt = cfg.integration.dt
    n_steps = int(round(duration / dt))
    z = np.zeros(12, dtype=np.float64)
    history = [z.copy()]

    def derivative(t: float, state: np.ndarray) -> np.ndarray:
        du = _doublet_controls(trim_u, amplitude, t) - trim_u
        return A @ state + B @ du

    t = 0.0
    for _step in range(n_steps):
        z = rk4_step(lambda t_stage, z_stage: derivative(t_stage, z_stage), t, z, dt)
        t += dt
        history.append(z.copy())
    return np.asarray(history)


def _nonlinear_delta(
    point,
    amplitude: float,
    cfg,
    plant,
    duration: float,
) -> np.ndarray:
    disturbed = simulate(
        point.x,
        _doublet_controller(point.u, amplitude),
        duration,
        cfg,
        plant,
        log_every=1,
    )
    baseline = simulate(
        point.x,
        _constant_controller(point.u),
        duration,
        cfg,
        plant,
        log_every=1,
    )
    return np.asarray(
        [
            euler_state_from_full(row) - euler_state_from_full(base)
            for row, base in zip(disturbed.x, baseline.x, strict=True)
        ],
        dtype=np.float64,
    )


def plot_trim_envelope(cfg, plant) -> Path:
    path = OUTPUT_DIR / "trim_envelope.png"
    speeds = np.array([18.0, 25.0, 35.0])
    gammas = np.deg2rad(np.array([-2.0, 0.0, 5.0]))
    points = trim_grid(speeds, gammas, 100.0, cfg, plant)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), constrained_layout=True)
    for gamma in gammas:
        row = [point for point in points if np.isclose(point.gamma, gamma)]
        row = sorted(row, key=lambda point: point.V)
        label = f"gamma={np.rad2deg(gamma):.0f} deg"
        axes[0].plot(
            [point.V for point in row],
            np.rad2deg([point.alpha for point in row]),
            marker="o",
            label=label,
        )
        axes[1].plot(
            [point.V for point in row],
            [point.u[3] for point in row],
            marker="o",
            label=label,
        )
    axes[0].set_xlabel("Airspeed [m/s]")
    axes[0].set_ylabel("Alpha [deg]")
    axes[0].set_title("Trim Angle Of Attack")
    axes[1].set_xlabel("Airspeed [m/s]")
    axes[1].set_ylabel("Throttle")
    axes[1].set_title("Trim Throttle")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_signal_overlay(cfg, plant, amplitude_deg: float, output_name: str) -> Path:
    path = OUTPUT_DIR / output_name
    point = trim(25.0, 0.0, 100.0, cfg, plant)
    A, B = linearise_euler(point, plant, cfg)
    amplitude = np.deg2rad(amplitude_deg)
    linear = _linear_doublet(A, B, point.u, amplitude, cfg, 5.0)
    nonlinear = _nonlinear_delta(point, amplitude, cfg, plant, 5.0)
    t = np.arange(linear.shape[0]) * cfg.integration.dt

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.8), sharex=True)
    axes[0].plot(t, linear[:, 7], label="linear")
    axes[0].plot(t, nonlinear[:, 7], "--", label="nonlinear")
    axes[0].set_ylabel("Delta theta [rad]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, linear[:, 3], label="linear")
    axes[1].plot(t, nonlinear[:, 3], "--", label="nonlinear")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Delta u [m/s]")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Elevator Doublet: {amplitude_deg:g} deg")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_aircraft("config/aircraft_v1.yaml")
    plant = AircraftPlant(cfg)
    paths = [
        plot_trim_envelope(cfg, plant),
        plot_modes(OUTPUT_DIR / "modes_root_locus.png"),
        plot_signal_overlay(cfg, plant, 0.5, "small_signal_overlay.png"),
        plot_signal_overlay(cfg, plant, 15.0, "large_signal_divergence.png"),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
