"""End-to-end checks that wind reaches every aerodynamic calculation."""

from pathlib import Path

import numpy as np

from uav_sim.environment.wind import DrydenTurbulence, SteadyWind, WindField
from uav_sim.integrate import simulate
from uav_sim.plant import AircraftPlant
from uav_sim.realism import build_plant, degradation_configurations
from uav_sim.state import initial_state


def _trim_controls() -> np.ndarray:
    return np.array([0.0, -0.116, 0.0, 0.329], dtype=np.float64)


def test_headwind_increases_airspeed_and_lift(cfg):
    state = initial_state(25.0, alpha=0.091, h=100.0)
    controls = _trim_controls()
    calm = AircraftPlant(cfg)
    windy = AircraftPlant(cfg, wind_model=WindField(steady=SteadyWind(10.0, 0.0)))
    windy.reset(np.random.default_rng(1), state)
    windy.prepare_step(0.0, state, controls, cfg.integration.dt)
    calm_diagnostics = calm.diagnostics(0.0, state, controls)
    wind_diagnostics = windy.diagnostics(0.0, state, controls)
    assert wind_diagnostics["V"] > calm_diagnostics["V"] + 9.5
    assert abs(wind_diagnostics["F_aero_b"][2]) > 1.7 * abs(
        calm_diagnostics["F_aero_b"][2]
    )


def test_crosswind_produces_sideslip(cfg):
    state = initial_state(25.0, h=100.0)
    plant = AircraftPlant(
        cfg,
        wind_model=WindField(steady=SteadyWind(8.0, np.deg2rad(270.0))),
    )
    plant.reset(np.random.default_rng(2), state)
    plant.prepare_step(0.0, state, _trim_controls(), cfg.integration.dt)
    assert abs(plant.diagnostics(0.0, state, _trim_controls())["beta"]) > 0.1


def test_turbulence_perturbs_alpha_and_pitch_acceleration(cfg):
    state = initial_state(25.0, alpha=0.091, h=100.0)
    turbulence = DrydenTurbulence("moderate", 100.0, np.random.default_rng(3))
    plant = AircraftPlant(cfg, wind_model=WindField(turbulence=turbulence))
    plant.reset(np.random.default_rng(3), state)
    alpha = []
    pitch_acceleration = []
    for index in range(5_000):
        t = index * cfg.integration.dt
        controls = _trim_controls()
        plant.prepare_step(t, state, controls, cfg.integration.dt)
        alpha.append(plant.diagnostics(t, state, controls)["alpha"])
        pitch_acceleration.append(plant.derivatives(t, state, controls)[11])
    assert np.std(alpha) > 0.005
    assert np.std(pitch_acceleration) > 0.01


def test_zero_realism_is_bit_identical_to_default_plant(cfg):
    state = initial_state(25.0, alpha=0.091, h=100.0)
    controls = _trim_controls()

    def controller(_t, _x, _cfg, _rng):
        return controls

    default = simulate(state, controller, 2.0, cfg, AircraftPlant(cfg), log_every=10)
    configured = simulate(
        state,
        controller,
        2.0,
        cfg,
        build_plant(cfg, degradation_configurations()["C0 perfect"]),
        log_every=10,
    )
    np.testing.assert_array_equal(default.x, configured.x)
    np.testing.assert_array_equal(default.u_actual, configured.u_actual)


def test_aerodynamic_modules_do_not_read_ground_velocity_directly():
    root = Path("src/uav_sim")
    for name in ("aero.py", "forces.py", "propulsion.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert "IDX_VEL" not in source
