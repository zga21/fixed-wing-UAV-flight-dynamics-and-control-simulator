# Phase 6 Provisional Controller Comparison

Both controllers flew the same three scenarios under the Phase 5 `C5 realistic`
configuration with 6 paired seeds (36 total flights). Lower
tracking score is better; it is the frozen `RMSE_h + 4*RMSE_V` score used by the
Phase 5 degradation study.

| Controller | Flights | Success | Mean tracking | P95 tracking | Effort | Saturation |
|---|---:|---:|---:|---:|---:|---:|
| NDI | 18 | 0.0% | 35.957 | 48.463 | 5.751 | 0.00% |
| PID | 18 | 0.0% | 37.937 | 50.325 | 6.715 | 0.00% |

This hand-tuned comparison is provisional. Phase 7 must give both controllers
the same optimisation budget before making a final architecture claim.
