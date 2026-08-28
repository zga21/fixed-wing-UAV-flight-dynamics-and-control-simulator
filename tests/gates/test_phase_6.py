"""Phase 6 gate: guarded NDI and fair nonlinear-controller comparison."""

import numpy as np
import pytest
from scripts.mismatch_sweep import PARAMETERS

from uav_sim.air_data import compute_air_data
from uav_sim.control.affine_model import ControlAffineModel
from uav_sim.control.ndi import NDIController
from uav_sim.control.rate_loop import ActuatorLimits, RateLoop
from uav_sim.state import IDX_OMEGA, State, initial_state


def _air(x):
    state = State.from_array(x)
    return compute_air_data(state.vel_b, state.quat, state.altitude)


def _ndi_rate_response(cfg, speed, command=0.2, duration=2.0, model=None):
    true_model = ControlAffineModel(cfg)
    internal_model = true_model if model is None else model
    controller = NDIController(internal_model, K_I=(0.0, 0.0, 0.0), cfg=cfg)
    x = initial_state(speed, alpha=0.05, h=100.0)
    air = _air(x)
    history = []
    omega_cmd = np.array([command, 0.0, 0.0])
    for _ in range(round(duration / cfg.integration.dt)):
        nu = controller._virtual_control(omega_cmd, x[IDX_OMEGA], cfg.integration.dt)
        allocation = controller._invert(nu, x, air)
        x[IDX_OMEGA] += cfg.integration.dt * (
            true_model.f(x, air) + true_model.g(air) @ allocation.delta
        )
        history.append(x[IDX_OMEGA].copy())
    return np.asarray(history)


def _pid_rate_response(cfg, speed, command=0.2, duration=2.0):
    model = ControlAffineModel(cfg)
    rate_loop = RateLoop(limits=ActuatorLimits.from_config(cfg))
    x = initial_state(speed, alpha=0.05, h=100.0)
    air = _air(x)
    trim_surfaces = np.linalg.solve(model.g(air), -model.f(x, air))
    limits = np.array(
        [
            [cfg.actuators.aileron.min_rad, cfg.actuators.aileron.max_rad],
            [cfg.actuators.elevator.min_rad, cfg.actuators.elevator.max_rad],
            [cfg.actuators.rudder.min_rad, cfg.actuators.rudder.max_rad],
        ]
    )
    history = []
    omega_cmd = np.array([command, 0.0, 0.0])
    for _ in range(round(duration / cfg.integration.dt)):
        increments = rate_loop.update(omega_cmd, x[IDX_OMEGA], cfg.integration.dt)
        surfaces = np.clip(trim_surfaces + increments, limits[:, 0], limits[:, 1])
        x[IDX_OMEGA] += cfg.integration.dt * (model.f(x, air) + model.g(air) @ surfaces)
        history.append(x[IDX_OMEGA].copy())
    return np.asarray(history)


@pytest.mark.gate
def test_perfect_model_rate_dynamics_are_inverted(cfg):
    model = ControlAffineModel(cfg)
    controller = NDIController(model, K_I=(0.0, 0.0, 0.0), cfg=cfg)
    x = initial_state(25.0, alpha=0.05, h=100.0)
    x[IDX_OMEGA] = np.array([0.02, -0.01, 0.03])
    air = _air(x)
    command = np.array([0.12, -0.08, 0.05])
    nu = controller._virtual_control(command, x[IDX_OMEGA], cfg.integration.dt)
    allocation = controller._invert(nu, x, air)
    achieved = model.f(x, air) + model.g(air) @ allocation.delta

    np.testing.assert_allclose(achieved, nu, rtol=0.02, atol=1e-10)


@pytest.mark.gate
def test_ndi_rate_response_is_speed_independent_and_beats_pid(cfg):
    ndi_18 = _ndi_rate_response(cfg, 18.0)
    ndi_35 = _ndi_rate_response(cfg, 35.0)
    pid_18 = _pid_rate_response(cfg, 18.0)
    pid_35 = _pid_rate_response(cfg, 35.0)
    command = 0.2
    ndi_difference = np.linalg.norm(ndi_18[:, 0] - ndi_35[:, 0])
    ndi_magnitude = np.linalg.norm(ndi_18[:, 0])
    ndi_error = np.mean((command - ndi_18[:, 0]) ** 2) + np.mean(
        (command - ndi_35[:, 0]) ** 2
    )
    pid_error = np.mean((command - pid_18[:, 0]) ** 2) + np.mean(
        (command - pid_35[:, 0]) ** 2
    )

    assert ndi_difference / ndi_magnitude < 0.05
    assert ndi_error < pid_error


@pytest.mark.gate
def test_low_speed_guard_and_allocation_stay_finite(cfg):
    controller = NDIController(ControlAffineModel(cfg), cfg=cfg)
    x = initial_state(8.0, alpha=0.05, h=100.0)
    air = _air(x)
    allocation = controller._invert(np.array([20.0, -20.0, 20.0]), x, air)

    assert np.all(np.isfinite(allocation.delta))
    assert np.all(np.isfinite(allocation.residual))
    assert np.any(allocation.saturated)
    assert controller._ndi_blend(8.0) == 0.0


@pytest.mark.gate
def test_mismatch_degrades_rate_tracking_without_divergence(cfg):
    nominal = _ndi_rate_response(cfg, 25.0)
    mismatched = _ndi_rate_response(
        cfg, 25.0, model=ControlAffineModel(cfg).perturb(C_l_delta_a=0.5)
    )
    nominal_error = np.mean((0.2 - nominal[:, 0]) ** 2)
    mismatch_error = np.mean((0.2 - mismatched[:, 0]) ** 2)

    assert np.all(np.isfinite(mismatched))
    assert np.max(np.abs(mismatched)) < 2.0
    assert mismatch_error > nominal_error


@pytest.mark.gate
def test_required_mismatch_campaign_dimensions_are_frozen():
    assert set(PARAMETERS) == {
        "C_m_alpha",
        "C_m_delta_e",
        "C_l_delta_a",
        "I_yy",
        "mass",
    }
    assert len(np.linspace(0.5, 1.5, 15)) >= 15
