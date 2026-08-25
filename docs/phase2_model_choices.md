# Phase 2 Model Choices

## Aerodynamic Drag

Phase 2 v1 uses the polar drag law from `T2.3`:

```text
C_D = C_D_0 + C_L^2 / (pi * oswald_e * AR)
```

`config/aircraft_v1.yaml` still carries `C_D_alpha` because it is present in the
source Aerosonde linear coefficient file. It is reference metadata for v1, not
the authoritative Phase 2 drag implementation.

Changing from this polar drag law to a linear `C_D_alpha` law after Phase 1 is
a benchmark change and requires a v2 config/conventions update.

## Propeller Torque

Phase 2 v1 models propulsion as thrust only:

```text
F_prop_b = [T, 0, 0]
M_prop_b = [0, 0, 0]
```

Propeller reaction torque is deferred until there is a sourced torque
coefficient. Adding it later changes trim and controller loads, so it requires a
benchmark version note.

## Aerosonde Propulsion Constant

`config/aircraft_v1.yaml` keeps the source `k_motor = 80.0` value from the
Aerosonde benchmark set. Under the simple Phase 2 momentum-theory formula, this
produces high full-throttle static thrust. We keep it in v1 for source fidelity
rather than silently retuning the benchmark after `phase-1-complete`.

If later validation shows this value must be changed for the research benchmark,
create `aircraft_v2.yaml` and record which Phase 2/3 gates and trim results are
invalidated.
