# Fixed-Wing UAV Flight Dynamics and Control Simulator

This repository is being built from a phase-gated project kit for a research
grade fixed-wing UAV simulator. The goal is not just to draw trajectories; the
goal is to produce reproducible evidence that compares baseline and robust
controllers across uncertainty.

Start here:

- `files/GUIDELINES.md` explains the research objective and build order.
- `files/TASKS.md` lists all phases and gate criteria.
- `docs/conventions.md` freezes the frame, sign, unit, integration, and
  aircraft-parameter rules used by every later phase.
- `docs/benchmark_contract.md` states which rules are benchmark-defining and
  how changes must be versioned.

Daily loop:

```bash
pip install -e ".[dev]"
make test
make lint
```

Phase 0 is the foundation. Do not begin Phase 1 until the conventions and
configuration are frozen and committed.
