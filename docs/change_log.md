# Project Change Log

## 2026-08-26 - Phase 2 Clarification And Test Markers

- Marked the parabolic polar drag law as the resolved authoritative v1 Phase 2
  drag model in `config/aircraft_v1.yaml` and `docs/benchmark_contract.md`.
- Documented Phase 2 propulsion assumptions in `src/uav_sim/propulsion.py`,
  including omitted propeller reaction torque, gyroscopic precession, and
  p-factor.
- Marked long-running integration and plant smoke tests as `slow`, keeping the
  fast test loop focused while preserving full-suite coverage.

## 2026-08-26 - Phase 2 Aerodynamics, Propulsion, And Plant

- Added ISA troposphere atmosphere modelling with density ratio and hot/cold day
  support.
- Added air-relative velocity and air-data calculation with wind already wired
  through the interface for Phase 5.
- Added the `AeroModel` protocol and `LinearAeroModel` coefficient build-up
  using the v1 polar drag choice recorded in `docs/phase2_model_choices.md`.
- Added aerodynamic force/moment assembly, propeller thrust, and full plant
  composition over rigid-body dynamics.
- Added a minimal `python -m uav_sim.cli fly` smoke-flight command.
- Added Phase 2 unit tests and gates for atmosphere references, force signs,
  aerodynamic monotonicity, control signs, propulsion behaviour, and plausible
  open-loop flight.
- Recorded that the source Aerosonde `k_motor = 80.0` value gives high static
  thrust under the simple Phase 2 formula and is kept for v1 source fidelity.

## 2026-08-26 - Phase 1 Rigid-Body Core

- Added scalar-first body-to-NED quaternion utilities in `src/uav_sim/rotations.py`,
  including DCM conversion, Euler I/O conversion, quaternion derivative,
  normalisation, and Hamilton product.
- Added the canonical 13-state layout and boundary accessors in
  `src/uav_sim/state.py`; raw state slicing outside that module is now enforced
  by test.
- Added gravity and pure 6-DOF rigid-body derivatives in
  `src/uav_sim/dynamics.py`, including Coriolis and gyroscopic terms.
- Added fixed-step RK4 integration and a deterministic simulation loop in
  `src/uav_sim/integrate.py`, plus `SimResult` for logged trajectories.
- Added Phase 1 unit tests and gate tests covering free fall, ballistic motion,
  quaternion norm, angular momentum conservation, and energy conservation.

## 2026-08-25 - Phase 0 Benchmark Clarifications

- Clarified that `config/aircraft_v1.yaml` uses the Aerosonde linear
  coefficient set matching Shalaby's `aerosonde.m` and Beard & McLain, and must
  not be mixed with newer MAVSim Python coefficients without creating a v2
  benchmark.
- Recorded that v1 carries both the linear drag parameter `C_D_alpha` and the
  polar drag parameter `oswald_e`; Phase 2 must declare which drag model is
  authoritative before implementation, and changing that choice later requires
  a benchmark version bump.
- Documented the project rudder sign convention and the resulting sign flips
  for all `delta_r` derivatives relative to the source dataset.
- Cached `MassProperties.inertia_tensor` and `MassProperties.inertia_inverse`
  so later dynamics code can reuse precomputed inertia data inside hot loops.
- Expanded aerodynamic sign validation to include `C_l_beta` and `C_Y_beta`.
