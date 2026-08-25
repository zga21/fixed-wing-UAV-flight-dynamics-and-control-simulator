# Benchmark Contract - v1

This file states what later phases are not allowed to move just because a
result looks bad.

## Frozen Inputs

The following files define the v1 benchmark:

- `docs/conventions.md`
- `docs/sources.bib`
- `config/aircraft_v1.yaml`

After Phase 1 starts, these files are treated as historical evidence. Do not
edit them to make a gate easier, improve a controller comparison, or change a
Monte Carlo success rate.

## Change Rule

If a benchmark-defining rule is wrong:

1. Create a new versioned file, for example `docs/conventions_v2.md`.
2. Create a matching versioned config, for example `config/aircraft_v2.yaml`.
3. Explain what changed, why it changed, and which gates/results are invalid.
4. Re-run every affected gate before comparing new results to old results.

Silent benchmark edits are treated as invalidating previous results.

## Benchmark-Defining Rules

- Reference frames and altitude sign.
- Quaternion ordering and rotation direction.
- State vector layout.
- Control input signs.
- Aircraft mass, geometry, inertia, propulsion, and aero derivatives.
- Flight envelope and constraint limits.
- Actuator limits and actuator operation order.
- Integration method, step size, and quaternion renormalisation rule.
- Scenario seeds, uncertainty distributions, and train/validation splits once
  those files are introduced in later phases.

## Result Metadata

Every future result row or exported result file must record:

- git commit
- dirty working-tree flag
- config hash
- conventions version
- random seed
- controller name and version

Any result missing these fields is exploratory only and cannot be used in the
final benchmark claim.
