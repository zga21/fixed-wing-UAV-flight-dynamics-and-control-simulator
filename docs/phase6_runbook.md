# Phase 6 Runbook

## Focused Verification

```bash
pytest -m gate tests/gates/test_phase_6.py
pytest -m "not slow"
ruff check src tests scripts
black --check src tests scripts
mypy src
```

## Paired Controller Record

```bash
python scripts/phase6_comparison.py --seeds 6 --workers 1
```

This uses the accepted six-seed Phase 5 runtime compromise and the complete
three-scenario `C5 realistic` plant. Keep `workers=1` for the reference record;
the harness separately tests that parallel and serial rows are bit-identical.

Before a full comparison, rank candidates with the real nonlinear/EKF pipeline
for only the first 30 seconds. This keeps the benchmark timestep and runs cases
in parallel:

```bash
python scripts/ndi_screen.py --outer --workers 8
```

## Full Mismatch Campaign

```bash
python scripts/mismatch_sweep.py --seeds 20 --steps 15 --workers 8
```

The runner first executes a two-seed, three-factor pilot. It aborts before the
full campaign if nominal NDI terminates or does not beat PID, avoiding thousands
of flights for a controller that cannot pass Gate 6.

The campaign covers five parameters over +/-50 percent, one at a time. It is
resumable through `results/phase6/mismatch_cases.jsonl`; completed cases are not
rerun. Results remain ignored cache, while the script writes committed plots to
`docs/figures/mismatch/` and crossover statements to
`docs/mismatch_findings.md` after all requested parameter sweeps finish.
