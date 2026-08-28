"""Controller-owned control-affine model tests."""

from dataclasses import replace

import numpy as np

from uav_sim.air_data import compute_air_data
from uav_sim.control.affine_model import ControlAffineModel
from uav_sim.state import IDX_OMEGA, State, initial_state


def test_affine_model_reproduces_plant_angular_acceleration(cfg, plant):
    x = initial_state(25.0, alpha=0.06, h=120.0)
    x[IDX_OMEGA] = np.array([0.08, -0.04, 0.03])
    controls = np.array([0.03, -0.07, 0.02, 0.6])
    state = State.from_array(x)
    air = compute_air_data(state.vel_b, state.quat, state.altitude)
    model = ControlAffineModel(cfg)

    predicted = model.f(x, air) + model.g(air) @ controls[:3]

    np.testing.assert_allclose(
        predicted, plant.derivatives(0.0, x, controls)[IDX_OMEGA]
    )


def test_effectiveness_scales_with_dynamic_pressure_and_is_invertible(cfg):
    model = ControlAffineModel(cfg)
    x = initial_state(18.0, h=100.0)
    state = State.from_array(x)
    air = compute_air_data(state.vel_b, state.quat, state.altitude)
    doubled = replace(air, q_bar=2.0 * air.q_bar)

    np.testing.assert_allclose(model.g(doubled), 2.0 * model.g(air))
    assert np.linalg.matrix_rank(model.g(air)) == 3
    assert np.isfinite(model.diagnostics(air).condition_number)
    assert model.diagnostics(doubled).minimum_singular_value == (
        2.0 * model.diagnostics(air).minimum_singular_value
    )


def test_perturb_returns_independent_internal_model(cfg):
    model = ControlAffineModel(cfg)
    changed = model.perturb(C_m_alpha=1.5, I_yy=0.8, mass=1.2)

    assert changed is not model
    assert changed.aero_derivs.C_m_alpha == 1.5 * cfg.aero.C_m_alpha
    assert changed.cfg.mass.I_yy == 0.8 * cfg.mass.I_yy
    assert changed.cfg.mass.m == 1.2 * cfg.mass.m
    assert model.aero_derivs.C_m_alpha == cfg.aero.C_m_alpha
    assert model.cfg.mass.I_yy == cfg.mass.I_yy


def test_controller_model_does_not_import_plant():
    source = open("src/uav_sim/control/affine_model.py", encoding="utf-8").read()
    assert "uav_sim.plant" not in source
