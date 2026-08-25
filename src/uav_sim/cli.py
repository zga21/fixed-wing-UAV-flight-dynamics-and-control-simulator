"""Minimal command-line interface for Phase 2 smoke flights."""

from __future__ import annotations

import argparse

import numpy as np

from uav_sim.config import load_aircraft
from uav_sim.integrate import simulate
from uav_sim.plant import AircraftPlant
from uav_sim.state import initial_state


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


if __name__ == "__main__":
    main()
