# Phase 6 Controller Fairness Protocol

This protocol was frozen before generating Phase 6 comparison results. It
governs every PID-versus-NDI result in Phases 6 and 7.

## Enforced Rules

1. **Same estimator.** Both controllers receive the same estimated-state
   interface. Neither controller may access plant truth when sensors and the EKF
   are enabled.
2. **Same hardware and environment.** Each paired run constructs a fresh plant
   with identical actuator, sensor, wind, turbulence, and estimator settings.
3. **Same scenarios and seeds.** Every controller flies every identical
   `(scenario, seed)` pair. A seed drives the same stochastic streams for each
   controller, and missing or duplicated pairs invalidate the comparison.
4. **Same metrics and repository state.** Both controllers use the frozen Phase
   4 metrics and success criteria. All rows must carry one aircraft-config hash
   and one git commit.

The harness in `src/uav_sim/compare.py` rejects results that violate the paired
scenario/seed, config, or commit requirements.

## Tuning Effort

Phase 6 uses competent hand-tuned gains for both controllers but remains a
provisional comparison. Phase 7 will give PID and NDI the same optimiser,
training scenarios, validation scenarios, and objective-evaluation budget.
Claims about controller architecture must distinguish the provisional Phase 6
result from that equal-budget result.

The NDI controller reuses the baseline trim feed-forward, TECS, heading,
attitude loops, loop rates, estimated state, and actuator limits unchanged. The
body-rate law is the only controller layer replaced in Phase 6.
