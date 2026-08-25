"""Fixed-step RK4 integration and simulation loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from uav_sim.config import AircraftConfig
from uav_sim.result import SimResult
from uav_sim.rotations import quat_normalise
from uav_sim.state import IDX_QUAT, N_STATES, State

DerivativeFn = Callable[..., np.ndarray]


def rk4_step(
    deriv_fn: DerivativeFn,
    t: float,
    x: np.ndarray,
    dt: float,
    *args: Any,
) -> np.ndarray:
    """Advance one classical RK4 step.

    For ``(13,)`` UAV states, the quaternion is renormalised after the completed
    step and never inside intermediate RK4 stages.
    """
    state = np.asarray(x, dtype=np.float64)
    k1 = np.asarray(deriv_fn(t, state, *args), dtype=np.float64)
    k2 = np.asarray(deriv_fn(t + 0.5 * dt, state + 0.5 * dt * k1, *args))
    k3 = np.asarray(deriv_fn(t + 0.5 * dt, state + 0.5 * dt * k2, *args))
    k4 = np.asarray(deriv_fn(t + dt, state + dt * k3, *args), dtype=np.float64)
    x_next = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if x_next.shape == (N_STATES,):
        x_next = x_next.copy()
        x_next[IDX_QUAT] = quat_normalise(x_next[IDX_QUAT])
    return x_next.astype(np.float64, copy=False)


def simulate(
    x0: np.ndarray,
    controller: Any,
    duration: float,
    cfg: AircraftConfig,
    plant: Callable[..., Any],
    log_every: int = 1,
    rng: np.random.Generator | None = None,
) -> SimResult:
    """Run a deterministic fixed-step simulation.

    Early failures return a ``SimResult`` instead of raising so Monte Carlo
    campaigns can count failed cases.
    """
    if duration < 0.0:
        raise ValueError("duration must be non-negative")
    if log_every < 1:
        raise ValueError("log_every must be at least 1")
    rng = np.random.default_rng(0) if rng is None else rng
    dt = cfg.integration.dt
    n_steps = int(np.ceil(duration / dt))
    log_indices = list(range(0, n_steps + 1, log_every))
    if log_indices[-1] != n_steps:
        log_indices.append(n_steps)

    t_log = np.empty(len(log_indices), dtype=np.float64)
    x_log = np.empty((len(log_indices), N_STATES), dtype=np.float64)
    u_log = np.empty((len(log_indices), 4), dtype=np.float64)
    u_actual_log = np.empty((len(log_indices), 4), dtype=np.float64)

    x = np.asarray(x0, dtype=np.float64).copy()
    t = 0.0
    log_cursor = 0
    terminated = False
    reason: str | None = None

    def log_sample(command: np.ndarray, actual: np.ndarray) -> None:
        nonlocal log_cursor
        t_log[log_cursor] = t
        x_log[log_cursor] = x
        u_log[log_cursor] = command
        u_actual_log[log_cursor] = actual
        log_cursor += 1

    command = np.zeros(4, dtype=np.float64)
    actual = np.zeros(4, dtype=np.float64)
    log_sample(command, actual)

    for step in range(n_steps):
        command = _controller_output(controller, t, x, cfg, rng)
        actual = command

        def wrapped_derivative(
            t_stage: float,
            x_stage: np.ndarray,
            actual_stage: np.ndarray = actual,
        ) -> np.ndarray:
            return _plant_derivative(plant, t_stage, x_stage, actual_stage, cfg, rng)

        try:
            x = rk4_step(wrapped_derivative, t, x, dt)
        except (FloatingPointError, ValueError) as exc:
            terminated = True
            reason = str(exc)
            break

        t += dt
        reason = _termination_reason(x, cfg)
        if reason is not None:
            terminated = True
        if (step + 1) in log_indices:
            log_sample(command, actual)
        if terminated:
            break

    return SimResult(
        t=t_log[:log_cursor].copy(),
        x=x_log[:log_cursor].copy(),
        u=u_log[:log_cursor].copy(),
        u_actual=u_actual_log[:log_cursor].copy(),
        diagnostics={},
        terminated_early=terminated,
        termination_reason=reason,
    )


def _controller_output(
    controller: Any,
    t: float,
    x: np.ndarray,
    cfg: AircraftConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if controller is None:
        return np.zeros(4, dtype=np.float64)
    if hasattr(controller, "update"):
        value = controller.update(t, x.copy(), cfg, rng)
    else:
        value = controller(t, x.copy(), cfg, rng)
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(f"controller output must have shape (4,), got {arr.shape}")
    return arr


def _plant_derivative(
    plant: Callable[..., Any],
    t: float,
    x: np.ndarray,
    u_actual: np.ndarray,
    cfg: AircraftConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    try:
        value = plant(t, x, u_actual, cfg, rng)
    except TypeError:
        value = plant(t, x, u_actual, cfg)
    if isinstance(value, tuple):
        value = value[0]
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (N_STATES,):
        raise ValueError(f"plant derivative must have shape ({N_STATES},)")
    return arr


def _termination_reason(x: np.ndarray, cfg: AircraftConfig) -> str | None:
    if not np.all(np.isfinite(x)):
        return "state contains NaN or infinity"
    if State.from_array(x).altitude < 0.0:
        return "ground impact"
    speed = np.linalg.norm(State.from_array(x).vel_b)
    if speed > 3.0 * cfg.envelope.V_max:
        return "divergent airspeed"
    return None
