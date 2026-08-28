"""Guarded nonlinear dynamic-inversion controller tests."""

import numpy as np

from uav_sim.air_data import compute_air_data
from uav_sim.control.affine_model import ControlAffineModel
from uav_sim.control.base import Controller, Reference
from uav_sim.control.ndi import NDIController
from uav_sim.state import IDX_OMEGA, State, initial_state


def _air(x):
    state = State.from_array(x)
    return compute_air_data(state.vel_b, state.quat, state.altitude)


def test_perfect_model_inversion_imposes_virtual_acceleration(cfg):
    model = ControlAffineModel(cfg)
    controller = NDIController(model)
    x = initial_state(25.0, alpha=0.05, h=100.0)
    x[IDX_OMEGA] = np.array([0.03, -0.02, 0.01])
    air = _air(x)
    nu = model.f(x, air) + np.array([0.4, -0.3, 0.2])

    allocation = controller._invert(nu, x, air)
    achieved = model.f(x, air) + model.g(air) @ allocation.delta

    np.testing.assert_allclose(achieved, nu, rtol=0.02, atol=1e-10)
    assert not np.any(allocation.saturated)


def test_low_speed_guard_produces_finite_in_limit_commands(cfg):
    controller = NDIController(ControlAffineModel(cfg))
    x = initial_state(8.0, alpha=0.05, h=100.0)
    controls = controller.update(
        x,
        Reference(altitude=100.0, airspeed=8.0, roll=0.2),
        0.0,
        cfg.integration.dt,
    ).to_array()

    assert np.all(np.isfinite(controls))
    assert cfg.actuators.aileron.min_rad <= controls[0] <= cfg.actuators.aileron.max_rad
    assert (
        cfg.actuators.elevator.min_rad <= controls[1] <= cfg.actuators.elevator.max_rad
    )
    assert cfg.actuators.rudder.min_rad <= controls[2] <= cfg.actuators.rudder.max_rad
    assert controller.diagnostics.q_bar_clamped is False
    assert controller.diagnostics.low_speed_blend == 0.0


def test_low_speed_blend_is_continuous(cfg):
    controller = NDIController(ControlAffineModel(cfg))
    speeds = np.linspace(11.5, 14.5, 301)
    blend = np.array([controller._ndi_blend(speed) for speed in speeds])

    assert np.all(np.diff(blend) >= 0.0)
    assert np.max(np.abs(np.diff(blend))) < 0.02
    assert blend[0] == 0.0
    assert blend[-1] == 1.0


def test_virtual_control_is_limited_and_integrator_stays_finite(cfg):
    controller = NDIController(ControlAffineModel(cfg), nu_limits=(1.0, 2.0, 3.0))
    nu = controller._virtual_control(np.ones(3), np.zeros(3), 0.01)

    np.testing.assert_allclose(nu, [1.0, 2.0, 3.0])
    assert np.all(np.isfinite(controller._rate_error_integral))


def test_ndi_protocol_gains_and_reset_are_deterministic(cfg):
    controller = NDIController(ControlAffineModel(cfg))
    assert isinstance(controller, Controller)
    original = controller.gains.copy()
    controller.gains = original
    assert len(controller.gain_names) == original.size
    assert len(controller.gain_bounds) == original.size

    x = initial_state(25.0, alpha=0.05, h=100.0)
    ref = Reference(altitude=110.0, airspeed=25.0, heading=0.2)
    first = controller.update(x, ref, 0.0, cfg.integration.dt).to_array()
    controller.reset()
    second = controller.update(x, ref, 0.0, cfg.integration.dt).to_array()
    np.testing.assert_array_equal(first, second)


def test_internal_model_mismatch_does_not_change_shared_configuration(cfg):
    perturbed = ControlAffineModel(cfg).perturb(C_m_alpha=1.5, mass=0.5)
    controller = NDIController(perturbed, cfg=cfg)

    assert controller.cfg is cfg
    assert controller.model.cfg.mass.m == 0.5 * cfg.mass.m
    assert controller.model.aero_derivs.C_m_alpha == 1.5 * cfg.aero.C_m_alpha


def test_partial_static_inversion_retains_trim_feedforward(cfg):
    controller = NDIController(
        ControlAffineModel(cfg), static_cancellation=0.2, cfg=cfg
    )
    controller._last_trim_controls[:3] = np.array([0.01, -0.1, 0.02])
    x = initial_state(25.0, alpha=0.05, h=100.0)
    allocation = controller._invert(np.zeros(3), x, _air(x))

    assert np.all(np.isfinite(allocation.delta))
    assert controller.static_cancellation == 0.2


def test_ndi_authority_is_bounded(cfg):
    with np.testing.assert_raises(ValueError):
        NDIController(ControlAffineModel(cfg), ndi_authority=1.1)


def test_pid_retention_is_bounded(cfg):
    with np.testing.assert_raises(ValueError):
        NDIController(ControlAffineModel(cfg), pid_retention=-0.1)
