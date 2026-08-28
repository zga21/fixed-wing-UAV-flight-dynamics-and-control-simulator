# Phase 6 Nonlinear Dynamic Inversion Controller

Phase 6 replaces only the Phase 4 body-rate PID layer. NDI retains the baseline
trim feed-forward, TECS-lite altitude/airspeed loop, heading loop, attitude loop,
loop rates, estimator input, and actuator limits. This keeps the comparison
focused on model-based angular-rate control.

## Internal Model

`ControlAffineModel` is owned by the controller and never imports the plant. It
expresses angular acceleration as

```text
omega_dot = f(x, air) + g(air) delta
```

where `f` contains neutral-surface aerodynamic moments and rigid-body gyroscopic
coupling, and `g` maps aileron, elevator, and rudder angles to angular
acceleration. `perturb()` creates an independent internal model; the true plant
and the nominal configuration used by the shared controller layers remain
unchanged.

The operational comparison uses `static_cancellation=0.1`: rate damping and
gyroscopic coupling are inverted fully around the Phase 4 trim feed-forward,
while only 10 percent of static angle-of-attack and sideslip moments are
cancelled. This partial inversion prevents small EKF air-angle errors from being
amplified into large surface commands. Full cancellation remains the default
and is used by the exact-model inversion tests.

The best sensor-aware candidate found by the benchmark-timestep screen uses 30
percent partially inverted command and 70 percent unchanged Phase 4 rate PID,
with a 0.08 s first-order filter. This robust blend is explicit in the evidence
scripts; it is not presented as a full-cancellation result.
`ndi_authority=1` with no filter remains available for ideal-model experiments
and the speed-independence figure.

For this linear-derivative aircraft, `g = q_bar G0`. Its condition number is
therefore constant for every positive `q_bar`; the matrix becomes ineffective,
not progressively ill-conditioned, as airspeed approaches zero. Diagnostics log
both condition number and minimum singular value so this distinction is visible.

## Guards And Allocation

The implementation clamps dynamic pressure used for inversion, falls back to a
damped pseudo-inverse when the condition threshold is exceeded, limits virtual
angular acceleration, and smoothly blends toward the Phase 4 rate PID below
14 m/s. Weighted least-squares allocation clamps unreachable surfaces and
redistributes the remaining demand among free surfaces. The unachievable
acceleration residual is fed back to the NDI integral term as multivariable
anti-windup.

The controller records condition number, minimum singular value, guard flags,
allocation residual, and saturated surfaces. `run_scenario` exports the numeric
health history into simulation diagnostics for later failure analysis.

## Mismatch Interpretation

Aerodynamic control/stability derivatives and inertia affect the angular model.
Translational mass does not appear in Euler's angular-momentum equation, so the
required mass sweep is a negative control and should be flat. The full mismatch
runner changes one internal parameter at a time while retaining the nominal
plant and identical realistic scenario/seed pairs.
