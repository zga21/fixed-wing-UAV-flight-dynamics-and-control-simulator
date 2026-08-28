# Phase 6 Controller Screening Result

Phase 6 uses staged screening to avoid spending hours on weak candidates:

1. A noisy angular surrogate rejects unstable or high-effort settings.
2. A 30-second nonlinear aircraft run uses the frozen `dt=0.002`, the Phase 5
   sensor suite, and EKF (`C2 sensors+EKF`) at seed 0.
3. Only a candidate that beats PID advances to full paired missions and the
   mismatch campaign.

The benchmark-timestep screen found that no tested NDI replacement or
augmentation beat the competent PID reference:

| Candidate | Static cancellation | NDI authority | Filter tau | Tracking score |
|---|---:|---:|---:|---:|
| PID reference | n/a | n/a | n/a | 10.031 |
| Best convex NDI/PID blend | 0.10 | 0.30 | 0.08 s | 12.627 |
| Best additive NDI augmentation | 0.05 | 0.05 | 0.05 s | 14.546 |

Lower is better (`RMSE_h + 4*RMSE_V`). Full textbook NDI remains valuable in
the ideal-model inner-loop test: it gives speed-independent rate dynamics and
beats PID there. Under the Phase 5 estimator, however, air-angle errors make
model cancellation fragile. Partial cancellation, filtering, and blending
prevented early ground impact but did not improve tracking enough to pass Gate
6.

Therefore the expensive 20-seed, five-parameter mismatch campaign is not run at
this point. Its script is complete and resumable, but its pilot intentionally
rejects the current nominal candidate. This is an open research result, not a
weakened benchmark or a reason to alter the frozen Phase 0/4 rules.
