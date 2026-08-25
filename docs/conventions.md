# Conventions - FROZEN

Status: **frozen as v1 from Phase 0**.

Changing this file changes the benchmark. Once Phase 1 starts, do not edit this
file to fix a test, improve a controller score, or rescue a Monte Carlo result.
If a convention is genuinely wrong, create `docs/conventions_v2.md`, create the
matching `config/*_v2.yaml` file, explain the change in the change log, and
re-run every affected gate.

---

## 1. Reference Frames

### Inertial frame: NED

The inertial frame is North-East-Down.

| Axis | Direction | Sign note |
|------|-----------|-----------|
| `x_n` | true north | positive north |
| `y_n` | east | positive east |
| `z_n` | down, toward Earth | positive downward |

**Altitude is `h = -p_D`.** A state with `p_D = -250 m` has altitude
`h = 250 m`.

Flat, non-rotating Earth is assumed. The planned envelope stays below 1000 m
and flights are short enough that curvature and Coriolis effects are below the
model uncertainty being studied.

### Body frame: FRD

The body frame is Forward-Right-Down with origin at the centre of gravity.

| Axis | Direction |
|------|-----------|
| `x_b` | forward, out the nose |
| `y_b` | out the right wing |
| `z_b` | down, through the belly |

When centre of gravity is perturbed in Monte Carlo studies, inertia is updated
with the parallel-axis theorem. The body-frame origin is not silently moved.

### Wind frame

`x_w` is aligned with the relative wind. Drag acts along `-x_w`; lift acts
along `-z_w`. Rotations between wind and body axes use `alpha` and `beta`.

---

## 2. Attitude Representation

Internal attitude is a scalar-first, body-to-NED unit quaternion:

```text
quat = [q0, q1, q2, q3]^T
q0^2 + q1^2 + q2^2 + q3^2 = 1
```

The identity attitude, level and nose north, is `[1, 0, 0, 0]`.

Euler angles are allowed only at I/O boundaries such as plots, CLI arguments,
and reports. Euler angles use a 3-2-1 sequence: `psi -> theta -> phi`.

The quaternion is renormalised after each completed integration step, not inside
intermediate RK4 stages.

---

## 3. State Vector

```text
x = [p_N, p_E, p_D, u, v, w, q0, q1, q2, q3, p, q, r]^T in R^13
```

| Index | Symbol | Meaning | Unit |
|-------|--------|---------|------|
| 0-2 | `p_N, p_E, p_D` | position in NED frame | m |
| 3-5 | `u, v, w` | velocity in body frame | m/s |
| 6-9 | `q0, q1, q2, q3` | attitude quaternion | dimensionless |
| 10-12 | `p, q, r` | angular rate in body frame | rad/s |

These ranges are defined once in `src/uav_sim/state.py`. Later code must use
named slices and accessors rather than hard-coded state indices.

Naming rule: `quat` means quaternion; bare `q` means pitch rate.

---

## 4. Control Inputs And Sign Conventions

```text
u_ctrl = [delta_a, delta_e, delta_r, delta_t]^T
```

| Input | Positive means | Produces |
|-------|----------------|----------|
| `delta_a` | right aileron up, left aileron down | positive roll moment `L` |
| `delta_e` | elevator trailing edge down | nose-down pitch, negative `M` |
| `delta_r` | rudder trailing edge left | nose-right yaw, positive `N` |
| `delta_t` | throttle, 0 idle to 1 full | thrust along `+x_b` |

Consequences checked by later gates:

| Derivative | Required sign | Reason |
|------------|---------------|--------|
| `C_m_delta_e` | negative | positive elevator gives nose-down pitch |
| `C_l_delta_a` | positive | positive aileron gives positive roll |
| `C_n_delta_r` | positive | positive rudder gives positive yaw |
| `C_m_alpha` | negative | static pitch stability |
| `C_m_q`, `C_l_p`, `C_n_r` | negative | damping opposes motion |
| `C_L_alpha` | positive | increasing angle of attack increases lift |
| `C_n_beta` | positive | weathercock stability |

The Aerosonde source data use a rudder sign opposite to this project convention.
The rudder derivative column is therefore multiplied by `-1` in
`config/aircraft_v1.yaml`.

---

## 5. Aircraft Parameters

The benchmark aircraft for v1 is the Aerosonde UAV parameter set commonly used
with Beard and McLain's small UAV model. The source values are recorded in
`docs/sources.bib`.

| Quantity | Symbol | Value | Unit | Source |
|----------|--------|-------|------|--------|
| Mass | `m` | 13.5 | kg | `BeardMcLain2012Aerosonde`, `ShalabyAerosondeParams` |
| Wing area | `S` | 0.55 | m^2 | `BeardMcLain2012Aerosonde`, `ShalabyAerosondeParams` |
| Wing span | `b` | 2.8956 | m | `BeardMcLain2012Aerosonde`, `ShalabyAerosondeParams` |
| Mean chord | `c_bar` | 0.18994 | m | `BeardMcLain2012Aerosonde`, `ShalabyAerosondeParams` |
| Roll inertia | `I_xx` | 0.8244 | kg m^2 | `BeardMcLain2012Aerosonde`, `ShalabyAerosondeParams` |
| Pitch inertia | `I_yy` | 1.135 | kg m^2 | `BeardMcLain2012Aerosonde`, `ShalabyAerosondeParams` |
| Yaw inertia | `I_zz` | 1.759 | kg m^2 | `BeardMcLain2012Aerosonde`, `ShalabyAerosondeParams` |
| Product of inertia | `I_xz` | 0.1204 | kg m^2 | `BeardMcLain2012Aerosonde`, `ShalabyAerosondeParams` |
| CG position | `x_cg_frac` | 0.25 | fraction of `c_bar` | project benchmark assumption |

Derived values:

| Quantity | Formula | v1 value |
|----------|---------|----------|
| Aspect ratio | `AR = b^2 / S` | 15.25 |
| Wing loading | `W/S = m g / S` | 240.71 N/m^2 |

---

## 6. Flight Envelope

| Quantity | Min | Nominal | Max |
|----------|-----|---------|-----|
| True airspeed `V` | 15 m/s | 25 m/s | 40 m/s |
| Altitude `h` | 0 m | 100 m | 1000 m |
| Angle of attack `alpha` | -10 deg | about 3 deg | 12 deg |
| Sideslip `beta` | -15 deg | 0 deg | 15 deg |
| Roll angle `phi` | -60 deg | 0 deg | 60 deg |
| Load factor `n` | -1 g | 1 g | 4 g |

Outside this envelope, the linear coefficient model is not claimed valid. Phase
9 counts envelope excursions as constraint violations.

---

## 7. Actuator Limits

| Surface | Position limit | Rate limit | Time constant |
|---------|----------------|------------|---------------|
| Aileron | +/- 25 deg | 286 deg/s | 0.05 s |
| Elevator | +/- 25 deg | 286 deg/s | 0.05 s |
| Rudder | +/- 25 deg | 286 deg/s | 0.05 s |
| Throttle | 0 to 1 | 2.0 /s | 0.20 s |

Application order is fixed: lag, then rate limit, then position saturation.

---

## 8. Environment Models

| Model | Choice |
|-------|--------|
| Gravity | constant `g = 9.80665 m/s^2` along `+z_n` |
| Atmosphere | ISA troposphere |
| Steady wind | constant vector in NED |
| Turbulence | Dryden shaping filters |
| Gusts | optional 1-cosine discrete gusts |

---

## 9. Numerical Integration

| Setting | Value | Justification |
|---------|-------|---------------|
| Method | classical RK4, fixed step | deterministic cost and adequate accuracy |
| Step `dt` | 0.002 s | 500 Hz plant update |
| Quaternion renormalisation | after each completed step | prevents norm drift |
| Convergence check | halve `dt`, require below 0.1 percent change | Phase 1 gate |

Fixed-step integration is part of the benchmark because Monte Carlo cases must
be reproducible and cost-predictable.

---

## 10. Controller Sample Rates

| Loop | Rate | Ratio |
|------|------|-------|
| Inner angular-rate loop | 200 Hz | 1x |
| Middle attitude loop | 50 Hz | 4x slower |
| Outer guidance/energy loop | 10 Hz | 5x slower |

Each outer loop must be 3-5 times slower than the loop inside it.

---

## 11. Trim Definition

`(x_star, u_star)` is a trim point at commanded `V_star`, `gamma_star`, and
`h_star` when:

```text
udot = vdot = wdot = 0
pdot = qdot = rdot = 0
phi = 0
beta = 0
```

The convergence criterion is `||xdot_dynamic|| < 1e-10`.

---

## 12. Initial Verification Cases

| # | Case | Acceptance |
|---|------|------------|
| 1 | Free fall from rest | matches `0.5 g t^2` to 1e-9 m |
| 2 | 45 deg ballistic launch | range matches analytic result to 1e-6 |
| 3 | Constant `omega`, 100 s | quaternion norm within 1e-10 of 1 |
| 4 | Torque-free asymmetric body | inertial angular momentum constant to 1e-8 |
| 5 | Torque-free body | kinetic energy constant to 1e-8 |
| 6 | Trim at 6 envelope points | residual below 1e-10, alpha rises as V falls |
| 7 | Modes at `V = 25 m/s` | five modes present in expected ranges |
| 8 | Small doublet | linear and nonlinear response within 2 percent over 5 s |

---

## 13. Change Log

| Date | Version | Change | Author |
|------|---------|--------|--------|
| 2026-08-25 | v1 | Initial Phase 0 freeze of frames, signs, state, controls, Aerosonde parameters, envelope, integration, and benchmark contract. | zga21 |
