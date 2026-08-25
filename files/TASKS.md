# TASKS.md — index and dependency order

Work top to bottom. A task may only start when every task in its **Depends on**
list is ticked. Each task file lives in `tasks/`.

Size key: **S** ≈ 1–2 h · **M** ≈ half a day · **L** ≈ 1–2 days

---

## Phase 0 — Freeze the rules

- [ ] **T0.1** Repository skeleton and tooling — *S* — `tasks/T0.1-repo-skeleton.md`
- [ ] **T0.2** Write and freeze the conventions document — *M* — `tasks/T0.2-conventions.md`
- [ ] **T0.3** Config schema, YAML files and loader — *M* — `tasks/T0.3-config-loader.md`

**Gate 0:** conventions documented, parameters frozen with sources, repo green.

## Phase 1 — Rigid body 6-DOF

- [ ] **T1.1** Quaternion and rotation utilities — *M* — `tasks/T1.1-quaternion-utils.md`
- [ ] **T1.2** State vector layout and accessors — *S* — `tasks/T1.2-state-layout.md`
- [ ] **T1.3** Rigid-body state derivative — *M* — `tasks/T1.3-rigid-body-derivatives.md`
- [ ] **T1.4** RK4 integrator and simulation loop — *M* — `tasks/T1.4-integrator.md`
- [ ] **T1.5** Phase 1 gate tests — *M* — `tasks/T1.5-gate-tests.md`

**Gate 1:** free fall, ballistic, quaternion norm, angular momentum, energy.

## Phase 2 — Aerodynamics and propulsion

- [ ] **T2.1** ISA atmosphere model — *S* — `tasks/T2.1-atmosphere.md`
- [ ] **T2.2** Air data: V, alpha, beta — *S* — `tasks/T2.2-air-data.md`
- [ ] **T2.3** AeroModel protocol and LinearAeroModel — *L* — `tasks/T2.3-aero-model.md`
- [ ] **T2.4** Force and moment assembly (wind → body) — *M* — `tasks/T2.4-forces-moments.md`
- [ ] **T2.5** Propulsion model — *S* — `tasks/T2.5-propulsion.md`
- [ ] **T2.6** Aircraft plant assembly — *M* — `tasks/T2.6-plant.md`
- [ ] **T2.7** Phase 2 gate tests — *M* — `tasks/T2.7-gate-tests.md`

**Gate 2:** lift monotonic, `C_m_alpha < 0`, damping signs negative, symmetry exact.

## Phase 3 — Trim and linearisation

- [ ] **T3.1** Trim solver — *L* — `tasks/T3.1-trim-solver.md`
- [ ] **T3.2** Numerical linearisation — *M* — `tasks/T3.2-linearisation.md`
- [ ] **T3.3** Mode identification and naming — *M* — `tasks/T3.3-modes.md`
- [ ] **T3.4** Phase 3 gate tests — *M* — `tasks/T3.4-gate-tests.md`

**Gate 3:** trim converges across envelope, five modes correct, linear ≈ nonlinear.
**This is the most important gate in the project.**

## Phase 4 — Baseline controller

- [ ] **T4.1** Controller protocol and data contracts — *S* — `tasks/T4.1-controller-protocol.md`
- [ ] **T4.2** PID with anti-windup — *M* — `tasks/T4.2-pid.md`
- [ ] **T4.3** Inner loop: angular rate control — *M* — `tasks/T4.3-rate-loop.md`
- [ ] **T4.4** Middle loop: attitude control — *M* — `tasks/T4.4-attitude-loop.md`
- [ ] **T4.5** Outer loop: altitude and airspeed (TECS-lite) — *L* — `tasks/T4.5-outer-loop.md`
- [ ] **T4.6** Scenario runner and metrics — *M* — `tasks/T4.6-scenario-metrics.md`
- [ ] **T4.7** Phase 4 gate tests and baseline record — *M* — `tasks/T4.7-gate-tests.md`

**Gate 4:** baseline flies the full mission in the perfect world; numbers recorded.

## Phase 5 — Sensors, actuators, wind, estimation

- [ ] **T5.1** Actuator model — *M* — `tasks/T5.1-actuators.md`
- [ ] **T5.2** Sensor suite — *L* — `tasks/T5.2-sensors.md`
- [ ] **T5.3** Wind field and Dryden turbulence — *L* — `tasks/T5.3-wind-turbulence.md`
- [ ] **T5.4** Wire wind into air data — *S* — `tasks/T5.4-wind-integration.md`
- [ ] **T5.5** Attitude and position EKF — *L* — `tasks/T5.5-ekf.md`
- [ ] **T5.6** Degradation study (answers RQ1) — *M* — `tasks/T5.6-degradation-study.md`
- [ ] **T5.7** Phase 5 gate tests — *M* — `tasks/T5.7-gate-tests.md`

**Gate 5:** degradation table exists; runs bit-reproducible from a seed.

## Phase 6 — Robust / nonlinear controller

- [ ] **T6.1** Control-affine model extraction — *M* — `tasks/T6.1-control-affine.md`
- [ ] **T6.2** NDI inner loop with singularity guards — *L* — `tasks/T6.2-ndi-core.md`
- [ ] **T6.3** Control allocation — *M* — `tasks/T6.3-allocation.md`
- [ ] **T6.4** Fair comparison harness — *M* — `tasks/T6.4-comparison-harness.md`
- [ ] **T6.5** Model-mismatch robustness sweep — *M* — `tasks/T6.5-mismatch-sweep.md`

**Gate 6:** advanced controller beats baseline fairly and degrades gracefully.

## Phase 7 — Controller optimisation

- [ ] **T7.1** Scenario sets: train / validation split — *M* — `tasks/T7.1-scenario-sets.md`
- [ ] **T7.2** Objective function — *L* — `tasks/T7.2-objective.md`
- [ ] **T7.3** CMA-ES runner with parallel evaluation — *M* — `tasks/T7.3-cmaes-runner.md`
- [ ] **T7.4** Tune both controllers, check overfitting — *M* — `tasks/T7.4-tune-both.md`

**Gate 7:** both controllers optimised; generalisation gap below 0.2.

## Phase 8 — Probabilistic / ML modelling (optional)

- [ ] **T8.1** Truth model and residual dataset — *M* — `tasks/T8.1-residual-data.md`
- [ ] **T8.2** GP residual aero model — *L* — `tasks/T8.2-gp-model.md`
- [ ] **T8.3** Calibration check and confidence map — *M* — `tasks/T8.3-confidence-map.md`

**Gate 8:** GP improves held-out RMSE, is calibrated, confidence map exists.

## Phase 9 — Monte Carlo campaign

- [ ] **T9.1** Uncertainty specification and sampler — *M* — `tasks/T9.1-uncertainty-spec.md`
- [ ] **T9.2** Single-case runner — *M* — `tasks/T9.2-case-runner.md`
- [ ] **T9.3** Campaign orchestration (parallel, resumable) — *L* — `tasks/T9.3-campaign.md`
- [ ] **T9.4** Statistical analysis — *M* — `tasks/T9.4-statistics.md`
- [ ] **T9.5** Failure triage — *M* — `tasks/T9.5-failure-triage.md`

**Gate 9:** success rate converged with CI; McNemar significant; failures inspected.

## Phase 10 — Report and demonstration site

- [ ] **T10.1** Results export contract — *M* — `tasks/T10.1-export-contract.md`
- [ ] **T10.2** Figure generation scripts — *L* — `tasks/T10.2-figures.md`
- [ ] **T10.3** Demonstration website — *L* — `tasks/T10.3-website.md`
- [ ] **T10.4** Technical report — *L* — `tasks/T10.4-report.md`

**Gate 10:** `make reproduce` regenerates every number and figure on a clean clone.

---

## If a task grows

Split it. Add `T4.5a`, `T4.5b` as new files and update this index. Never let a
single task sprawl — small tasks are the mechanism that keeps the project
finishable.
