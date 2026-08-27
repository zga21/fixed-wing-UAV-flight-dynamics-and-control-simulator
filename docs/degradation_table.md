# Phase 5 Degradation Study

Generated from the frozen C0-C5 ablation using all three Phase 4 scenarios and
6 paired seeds per configuration (108 effective flights).
Deterministic C0, C1, and C3 cases are simulated once per scenario and repeated
across seed rows; stochastic sensor/EKF and turbulence cases execute every seed.

The combined tracking score is `RMSE_h + 4*RMSE_V`, which gives one metre per
second of airspeed error the same weight as four metres of altitude error. It
increased from 16.008 in C0 to 37.937 in C5. This is the measured
answer to RQ1: the unchanged Phase 4 controller loses tracking quality when its
actuators, state information, and atmosphere are made realistic.

## Dominant mechanism

**C2 sensors+EKF** is the largest isolated contribution, changing the tracking score
by +177.5% relative to C0. The mechanism ranking is based on
paired scenarios and seeds, so it separates the individual effects before the
nonlinear interactions in C5. Rate limiting adds phase lag, estimation adds
noise and delay, and Dryden turbulence injects energy around the control-loop
bandwidth; their combined effect is not expected to equal the sum of ablations.

## Mechanism contribution

| Configuration | Tracking-score change |
|---|---:|
| C1 actuators | -0.1% |
| C2 sensors+EKF | +177.5% |
| C3 steady wind | +0.8% |
| C4 turbulence | +25.3% |
| C5 realistic | +137.0% |

## All metrics

| configuration | metric | mean | p95 | change_percent |
|---|---|---|---|---|
| C0 perfect | control_effort | 2.853 | 3.561 | -0.0% |
| C0 perfect | control_rate_effort | 5.41 | 7.544 | +0.0% |
| C0 perfect | disturbance_recovery_time | 0 | 0 | n/a (C0=0) |
| C0 perfect | envelope_violations | 0 | 0 | n/a (C0=0) |
| C0 perfect | integration_failures | 0 | 0 | n/a (C0=0) |
| C0 perfect | max_V_error | 9.09 | 12.88 | +0.0% |
| C0 perfect | max_h_error | 35.04 | 50 | +0.0% |
| C0 perfect | max_load_factor | 1.716 | 2 | +0.0% |
| C0 perfect | overshoot_h | 13.35 | 31.88 | +0.0% |
| C0 perfect | rmse_V | 2.064 | 2.965 | +0.0% |
| C0 perfect | rmse_h | 7.751 | 11.27 | +0.0% |
| C0 perfect | rmse_phi | 0.1567 | 0.2182 | -0.0% |
| C0 perfect | rmse_theta | 0.09328 | 0.09608 | -0.0% |
| C0 perfect | saturation_fraction | 0 | 0 | n/a (C0=0) |
| C0 perfect | settling_time_h | 12.03 | 14.2 | -0.0% |
| C1 actuators | control_effort | 2.871 | 3.611 | +0.6% |
| C1 actuators | control_rate_effort | 4.4 | 6.125 | -18.7% |
| C1 actuators | disturbance_recovery_time | 0 | 0 | n/a (C0=0) |
| C1 actuators | envelope_violations | 1 | 3 | n/a (C0=0) |
| C1 actuators | integration_failures | 0 | 0 | n/a (C0=0) |
| C1 actuators | max_V_error | 10.01 | 14.1 | +10.2% |
| C1 actuators | max_h_error | 35.04 | 50 | -0.0% |
| C1 actuators | max_load_factor | 1.723 | 2.011 | +0.4% |
| C1 actuators | overshoot_h | 13.34 | 31.88 | -0.0% |
| C1 actuators | rmse_V | 2.033 | 2.911 | -1.5% |
| C1 actuators | rmse_h | 7.867 | 11.42 | +1.5% |
| C1 actuators | rmse_phi | 0.156 | 0.2166 | -0.5% |
| C1 actuators | rmse_theta | 0.09372 | 0.09659 | +0.5% |
| C1 actuators | saturation_fraction | 0 | 0 | n/a (C0=0) |
| C1 actuators | settling_time_h | 12 | 14.1 | -0.3% |
| C2 sensors+EKF | control_effort | 7.756 | 15.49 | +171.8% |
| C2 sensors+EKF | control_rate_effort | 212.1 | 403 | +3820.6% |
| C2 sensors+EKF | disturbance_recovery_time | 0 | 0 | n/a (C0=0) |
| C2 sensors+EKF | envelope_violations | 233.2 | 503.2 | n/a (C0=0) |
| C2 sensors+EKF | integration_failures | 0 | 0 | n/a (C0=0) |
| C2 sensors+EKF | max_V_error | 14.54 | 21.12 | +60.0% |
| C2 sensors+EKF | max_h_error | 51.83 | 84.64 | +47.9% |
| C2 sensors+EKF | max_load_factor | 2.182 | 4 | +27.1% |
| C2 sensors+EKF | overshoot_h | 34.73 | 55.35 | +160.2% |
| C2 sensors+EKF | rmse_V | 6.281 | 9.517 | +204.3% |
| C2 sensors+EKF | rmse_h | 19.3 | 32.98 | +148.9% |
| C2 sensors+EKF | rmse_phi | 0.226 | 0.3233 | +44.2% |
| C2 sensors+EKF | rmse_theta | 0.2274 | 0.312 | +143.8% |
| C2 sensors+EKF | saturation_fraction | 0.09293 | 0.2966 | n/a (C0=0) |
| C2 sensors+EKF | settling_time_h | 14.98 | 20 | +24.5% |
| C3 steady wind | control_effort | 2.844 | 3.552 | -0.3% |
| C3 steady wind | control_rate_effort | 7.466 | 9.598 | +38.0% |
| C3 steady wind | disturbance_recovery_time | 0 | 0 | n/a (C0=0) |
| C3 steady wind | envelope_violations | 0 | 0 | n/a (C0=0) |
| C3 steady wind | integration_failures | 0 | 0 | n/a (C0=0) |
| C3 steady wind | max_V_error | 9.675 | 12.89 | +6.4% |
| C3 steady wind | max_h_error | 35.15 | 50.17 | +0.3% |
| C3 steady wind | max_load_factor | 1.716 | 2 | -0.0% |
| C3 steady wind | overshoot_h | 13.35 | 31.88 | +0.0% |
| C3 steady wind | rmse_V | 2.087 | 2.977 | +1.1% |
| C3 steady wind | rmse_h | 7.779 | 11.3 | +0.4% |
| C3 steady wind | rmse_phi | 0.1567 | 0.2182 | -0.0% |
| C3 steady wind | rmse_theta | 0.09323 | 0.09603 | -0.1% |
| C3 steady wind | saturation_fraction | 0 | 0 | n/a (C0=0) |
| C3 steady wind | settling_time_h | 12.03 | 14.2 | -0.0% |
| C4 turbulence | control_effort | 2.542 | 3.443 | -10.9% |
| C4 turbulence | control_rate_effort | 20.81 | 28.3 | +284.6% |
| C4 turbulence | disturbance_recovery_time | 0 | 0 | n/a (C0=0) |
| C4 turbulence | envelope_violations | 5.556 | 14.45 | n/a (C0=0) |
| C4 turbulence | integration_failures | 0 | 0 | n/a (C0=0) |
| C4 turbulence | max_V_error | 10.41 | 14.99 | +14.5% |
| C4 turbulence | max_h_error | 37.73 | 54.55 | +7.7% |
| C4 turbulence | max_load_factor | 1.733 | 2.055 | +1.0% |
| C4 turbulence | overshoot_h | 19.62 | 39.49 | +47.0% |
| C4 turbulence | rmse_V | 2.662 | 3.671 | +29.0% |
| C4 turbulence | rmse_h | 9.409 | 13.25 | +21.4% |
| C4 turbulence | rmse_phi | 0.1737 | 0.2366 | +10.9% |
| C4 turbulence | rmse_theta | 0.1044 | 0.1199 | +11.9% |
| C4 turbulence | saturation_fraction | 0 | 0 | n/a (C0=0) |
| C4 turbulence | settling_time_h | 14.74 | 20 | +22.5% |
| C5 realistic | control_effort | 6.715 | 11.33 | +135.4% |
| C5 realistic | control_rate_effort | 72.48 | 91.56 | +1239.7% |
| C5 realistic | disturbance_recovery_time | 0 | 0 | n/a (C0=0) |
| C5 realistic | envelope_violations | 167.2 | 404.6 | n/a (C0=0) |
| C5 realistic | integration_failures | 0 | 0 | n/a (C0=0) |
| C5 realistic | max_V_error | 14.74 | 19.31 | +62.2% |
| C5 realistic | max_h_error | 47.74 | 71.38 | +36.2% |
| C5 realistic | max_load_factor | 2.71 | 4 | +57.9% |
| C5 realistic | overshoot_h | 34.75 | 60.25 | +160.3% |
| C5 realistic | rmse_V | 5.407 | 7.097 | +162.0% |
| C5 realistic | rmse_h | 16.31 | 22.64 | +110.4% |
| C5 realistic | rmse_phi | 0.2755 | 0.4625 | +75.9% |
| C5 realistic | rmse_theta | 0.2124 | 0.3128 | +127.7% |
| C5 realistic | saturation_fraction | 0 | 0 | n/a (C0=0) |
| C5 realistic | settling_time_h | 14.88 | 20 | +23.6% |
