"""Minimal command-line interface for smoke flights and Phase 3 analysis."""

from __future__ import annotations

import argparse

import numpy as np

from uav_sim.config import load_aircraft
from uav_sim.integrate import simulate
from uav_sim.linearise import decouple, linearise_euler
from uav_sim.modes import identify_modes
from uav_sim.plant import AircraftPlant
from uav_sim.state import initial_state
from uav_sim.trim import trim


def main() -> None:
    parser = argparse.ArgumentParser(prog="uav_sim")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fly = subparsers.add_parser("fly")
    fly.add_argument("--config", default="config/aircraft_v1.yaml")
    fly.add_argument("--V", type=float, default=25.0)
    fly.add_argument("--altitude", type=float, default=100.0)
    fly.add_argument("--duration", type=float, default=10.0)
    fly.add_argument("--alpha-deg", type=float, default=5.0)
    fly.add_argument("--elevator-deg", type=float, default=-6.5)
    fly.add_argument("--throttle", type=float, default=0.35)
    trim_cmd = subparsers.add_parser("trim")
    trim_cmd.add_argument("--config", default="config/aircraft_v1.yaml")
    trim_cmd.add_argument("--V", type=float, default=25.0)
    trim_cmd.add_argument("--gamma-deg", type=float, default=0.0)
    trim_cmd.add_argument("--altitude", type=float, default=100.0)
    modes_cmd = subparsers.add_parser("modes")
    modes_cmd.add_argument("--config", default="config/aircraft_v1.yaml")
    modes_cmd.add_argument("--V", type=float, default=25.0)
    modes_cmd.add_argument("--altitude", type=float, default=100.0)
    args = parser.parse_args()

    if args.command == "fly":
        cfg = load_aircraft(args.config)
        plant = AircraftPlant(cfg)
        x0 = initial_state(
            V=args.V,
            alpha=np.deg2rad(args.alpha_deg),
            h=args.altitude,
        )
        controls = np.array(
            [0.0, np.deg2rad(args.elevator_deg), 0.0, args.throttle],
            dtype=np.float64,
        )

        def controller(_t, _x, _cfg, _rng):
            return controls

        result = simulate(x0, controller, args.duration, cfg, plant, log_every=50)
        print(
            f"terminated={result.terminated_early} "
            f"reason={result.termination_reason!r} "
            f"final_altitude={-result.x[-1, 2]:.2f} m"
        )
    elif args.command == "trim":
        cfg = load_aircraft(args.config)
        plant = AircraftPlant(cfg)
        point = trim(args.V, np.deg2rad(args.gamma_deg), args.altitude, cfg, plant)
        print(
            f"V={point.V:.3f} m/s "
            f"gamma={np.rad2deg(point.gamma):.3f} deg "
            f"alpha={np.rad2deg(point.alpha):.6f} deg "
            f"theta={np.rad2deg(point.theta):.6f} deg "
            f"delta_e={np.rad2deg(point.u[1]):.6f} deg "
            f"delta_t={point.u[3]:.6f} "
            f"residual={point.residual:.3e}"
        )
    elif args.command == "modes":
        cfg = load_aircraft(args.config)
        plant = AircraftPlant(cfg)
        point = trim(args.V, 0.0, args.altitude, cfg, plant)
        A, B = linearise_euler(point, plant, cfg)
        lon, lat = decouple(A, B)
        modes = identify_modes(lon.A, lat.A)
        for name, mode in modes.items():
            print(
                f"{name}: lambda={mode.eigenvalue.real:.6g}"
                f"{mode.eigenvalue.imag:+.6g}j "
                f"omega_n={mode.omega_n:.6g} zeta={mode.zeta:.6g}"
            )


if __name__ == "__main__":
    main()
