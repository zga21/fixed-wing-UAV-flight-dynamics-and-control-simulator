# Conventions — FROZEN

Status: **frozen** as of Phase 0. Changing anything here invalidates every
result produced before the change. If you must change it, bump to
`conventions_v2.md` and re-run the affected gates.

---

## 1. Reference frames

### Inertial frame: NED (North-East-Down)

| Axis | Direction | Sign note |
|------|-----------|-----------|
| `x_n` | True north | — |
| `y_n` | East | — |
| `z_n` | **Down**, toward the centre of the Earth | positive = downward |

**Altitude is `h = -p_D`.** This single line causes more bugs than any other in
the project. `p_D = -100` means the aircraft is 100 m *above* the origin.

Flat, non-rotating Earth is assumed. Justification: the envelope is under
1000 m and flights are under 10 minutes, so curvature and Coriolis effects are
orders of magnitude below the modelled uncertainties.

### Body frame: FRD (Forward-Right-Down)

| Axis | Direction |
|------|-----------|
| `x_b` | Forward, out the nose |
| `y_b` | Out the right wing |
| `z_b` | Down, through the belly |

Origin at the centre of gravity. When the CG is perturbed in Phase 9, the
inertia tensor is transformed with the parallel-axis theorem rather than the
frame origin being moved.

### Wind frame

`x_w` aligned with the relative wind. Lift acts along `-z_w`, drag along
`-x_w`. Rotation to body axes uses `alpha` and `beta`.

---

## 2. Attitude representation

**Internal: quaternion, scalar-first, body-to-NED, unit norm.**

```
quat = [q0, q1, q2, q3]^T,   q0^2 + q1^2 + q2^2 + q3^2 = 1
```

Identity attitude (level, nose north) is `[1, 0, 0, 0]`.

**For plots and reports only: Euler angles**, 3-2-1 sequence
(`psi -> theta -> phi`), converted at the output boundary only. Euler angles are
never integrated, because of the singularity at `theta = +/- 90 deg` which an
agile aircraft will reach.

Renormalise the quaternion after every completed integration step, not inside
RK4 stages.

---

## 3. State vector

```
x = [p_N, p_E, p_D, u, v, w, q0, q1, q2, q3, p, q, r]^T   in R^13
```

| Index | Symbol | Meaning | Unit |
|-------|--------|---------|------|
| 0–2 | `p_N, p_E, p_D` | Position, NED frame | m |
| 3–5 | `u, v, w` | Velocity, body frame | m/s |
| 6–9 | `q0..q3` | Attitude quaternion | — |
| 10–12 | `p, q, r` | Angular rate, body frame | rad/s |

These index ranges are defined once in `src/uav_sim/state.py` as named slices
and are never hard-coded anywhere else.

Naming collision warning: `q` is both pitch rate (index 11) and the quaternion.
In code the quaternion is always `quat`; bare `q` always means pitch rate.

---

## 4. Control inputs and sign conventions

```
u = [delta_a, delta_e, delta_r, delta_t]^T
```

| Input | Positive means | Produces |
|-------|----------------|----------|
| `delta_a` (aileron) | right aileron up, left down | **right-wing-down roll**, positive `L` |
| `delta_e` (elevator) | trailing edge **down** | **nose-down pitch**, negative `M` |
| `delta_r` (rudder) | trailing edge **left** | **nose-right yaw**, positive `N` |
| `delta_t` (throttle) | 0 = idle, 1 = full | thrust along `+x_b` |

Consequences, all checked in the Phase 2 gate:

| Derivative | Expected sign | Reason |
|------------|---------------|--------|
| `C_m_delta_e` | **negative** | positive elevator gives nose-down |
| `C_l_delta_a` | positive | positive aileron gives right roll |
| `C_n_delta_r` | positive | positive rudder gives nose-right |
| `C_m_alpha` | **negative** | statically stable in pitch |
| `C_m_q`, `C_l_p`, `C_n_r` | **negative** | damping opposes motion |
| `C_L_alpha` | positive | more alpha, more lift |
| `C_n_beta` | positive | weathercock stability |

---

## 5. Aircraft parameters — FILL THESE IN

Replace the placeholders and cite a source for each. Using a published airframe
(e.g. a NASA technical report) is worth far more than inventing values, because
it lets you validate against measured behaviour in Phase 3.

| Quantity | Symbol | Value | Unit | Source |
|----------|--------|-------|------|--------|
| Mass | `m` | 13.5 | kg | TODO |
| Wing area | `S` | 0.55 | m^2 | TODO |
| Wing span | `b` | 2.90 | m | TODO |
| Mean chord | `c_bar` | 0.19 | m | TODO |
| Roll inertia | `I_xx` | 0.824 | kg m^2 | TODO |
| Pitch inertia | `I_yy` | 1.135 | kg m^2 | TODO |
| Yaw inertia | `I_zz` | 1.759 | kg m^2 | TODO |
| Product of inertia | `I_xz` | 0.120 | kg m^2 | TODO |
| CG position | `x_cg` | 0.25 | fraction of `c_bar` | TODO |

Derived: `AR = b^2 / S`, wing loading `W/S = m g / S`.

---

## 6. Flight envelope

| Quantity | Min | Nominal | Max |
|----------|-----|---------|-----|
| True airspeed `V` | 15 m/s | 25 m/s | 40 m/s |
| Altitude `h` | 0 m | 100 m | 1000 m |
| Angle of attack `alpha` | -10 deg | ~3 deg | +12 deg |
| Sideslip `beta` | -15 deg | 0 deg | +15 deg |
| Roll angle `phi` | -60 deg | 0 deg | +60 deg |
| Load factor `n` | -1 g | 1 g | +4 g |

Outside these bounds the linear coefficient model is not claimed valid, and
Phase 9 counts an excursion as a constraint violation.

---

## 7. Actuator limits

| Surface | Position limit | Rate limit | Time constant |
|---------|----------------|------------|---------------|
| Aileron | +/- 25 deg | 5.0 rad/s | 0.05 s |
| Elevator | +/- 25 deg | 5.0 rad/s | 0.05 s |
| Rudder | +/- 25 deg | 5.0 rad/s | 0.05 s |
| Throttle | 0 to 1 | 2.0 /s | 0.20 s |

Application order is fixed: **lag, then rate limit, then position saturation.**
Any other order silently disables the rate limit.

---

## 8. Environment models

| Model | Choice |
|-------|--------|
| Gravity | Constant `g = 9.80665 m/s^2` along `+z_n` |
| Atmosphere | ISA troposphere, `T = 288.15 - 0.0065 h`, `rho_0 = 1.225` |
| Steady wind | Constant vector in NED |
| Turbulence | Dryden spectrum via shaping filters on white noise |
| Gusts | 1-cosine discrete gusts, optional |

---

## 9. Numerical integration

| Setting | Value | Justification |
|---------|-------|---------------|
| Method | Classical RK4, fixed step | Adequate accuracy, deterministic cost |
| Step `dt` | 0.002 s (500 Hz) | ~10x faster than the fastest expected mode |
| Quaternion renormalisation | after each completed step | prevents norm drift |
| Convergence check | halve `dt`, require < 0.1% change | Phase 1 acceptance |

Fixed-step is chosen over adaptive so Monte Carlo runs have deterministic cost
and bit-identical reproducibility.

---

## 10. Controller sample rates

| Loop | Rate | Ratio |
|------|------|-------|
| Inner (angular rate) | 200 Hz | 1x |
| Middle (attitude) | 50 Hz | 4x slower |
| Outer (guidance, altitude, airspeed) | 10 Hz | 5x slower |

Each outer loop must be 3–5x slower than the loop inside it, or the loops fight.

---

## 11. Trim definition

`(x*, u*)` such that at commanded `V*`, `gamma*`, `h*`:

```
udot = vdot = wdot = 0
pdot = qdot = rdot = 0
phi = 0,  beta = 0        (wings-level, coordinated)
```

Convergence criterion: `||xdot_dynamic|| < 1e-10`.

---

## 12. Initial verification cases

| # | Case | Acceptance |
|---|------|-----------|
| 1 | Free fall from rest | matches `0.5 g t^2` to 1e-9 m |
| 2 | 45 deg ballistic launch | range matches `V0^2 sin(90deg)/g` to 1e-6 |
| 3 | Constant `omega`, 100 s | quaternion norm within 1e-10 of 1 |
| 4 | Torque-free asymmetric body | inertial angular momentum constant to 1e-8 |
| 5 | Torque-free body | kinetic energy constant to 1e-8 |
| 6 | Trim at 6 envelope points | residual < 1e-10, `alpha` rises as `V` falls |
| 7 | Modes at `V = 25` | five modes present, in expected ranges |
| 8 | Small doublet, linear vs nonlinear | agreement within 2% over 5 s |

---

## 13. Change log

| Date | Version | Change | Author |
|------|---------|--------|--------|
| — | v1 | Initial freeze | — |
