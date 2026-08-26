# Phase 4 Success Criteria

Frozen date: 2026-08-26

These mission-level criteria are defined before the Phase 4 baseline record and
are used by `src/uav_sim/metrics.py`:

- `rmse_h <= 12.0 m`
- `rmse_V <= 3.0 m/s`
- `max_phi_deg <= 60.0 deg`
- `saturation_fraction <= 0.20` for control-surface saturation
- `envelope_violations == 0`

These thresholds define the Phase 4 perfect-world baseline pass/fail rule. Later
controller, robustness, optimisation, and Monte Carlo phases must not tune these
criteria after seeing results.
