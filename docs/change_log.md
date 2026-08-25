# Project Change Log

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
