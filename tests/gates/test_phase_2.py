"""Phase 2 sign, monotonicity, and open-loop plausibility gates."""

import numpy as np
import pytest

from uav_sim.aero import LinearAeroModel
from uav_sim.air_data import AirData
from uav_sim.environment.atmosphere import isa
from uav_sim.forces import aero_forces_moments
from uav_sim.integrate import simulate
from uav_sim.plant import AircraftPlant
from uav_sim.propulsion import propeller_thrust
from uav_sim.state import IDX_VEL, initial_state


def _air(V=25.0, alpha=0.0, beta=0.0):
    return AirData(
        V=V,
        alpha=alpha,
        beta=beta,
        q_bar=0.5 * 1.225 * V**2,
        v_air_b=np.array([V, 0.0, 0.0]),
        rho=1.225,
    )


def _near_trim_state():
    return initial_state(25.0, alpha=0.091, h=100.0)


def _near_trim_controls():
    return np.array([0.0, -0.116, 0.0, 0.329], dtype=np.float64)


@pytest.mark.gate
def test_lift_increases_with_alpha(cfg):
    """Catches sign error in C_L_alpha or alpha computed with wrong sign."""
    model = LinearAeroModel(cfg.aero)
    alphas = np.deg2rad(np.linspace(-5.0, 10.0, 50))
    values = [
        model.coefficients(
            _air(alpha=alpha),
            np.zeros(3),
            np.zeros(4),
            cfg.geometry,
        ).C_L
        for alpha in alphas
    ]

    assert np.all(np.diff(values) > 0.0)


@pytest.mark.gate
def test_static_pitch_stability_and_damping_derivatives(cfg):
    """Catches divergent pitch or energy-creating damping signs."""
    assert cfg.aero.C_m_alpha < 0.0
    assert cfg.aero.C_m_q < 0.0
    assert cfg.aero.C_l_p < 0.0
    assert cfg.aero.C_n_r < 0.0


@pytest.mark.gate
def test_lateral_symmetry_exact(cfg):
    """Catches index swaps and stray lateral coupling in symmetric flight."""
    model = LinearAeroModel(cfg.aero)
    coeffs = model.coefficients(
        air=_air(alpha=0.05, beta=0.0),
        omega_b=np.array([0.0, 0.1, 0.0]),
        controls=np.array([0.0, -0.05, 0.0, 0.5]),
        geom=cfg.geometry,
    )

    assert coeffs.C_Y == 0.0
    assert coeffs.C_l == 0.0
    assert coeffs.C_n == 0.0


@pytest.mark.gate
def test_control_and_force_signs_match_convention(cfg):
    """Catches inverted elevator, aileron, rudder, lift, or drag signs."""
    model = LinearAeroModel(cfg.aero)
    elevator = model.coefficients(
        _air(),
        np.zeros(3),
        np.array([0.0, 0.1, 0.0, 0.5]),
        cfg.geometry,
    )
    aileron = model.coefficients(
        _air(),
        np.zeros(3),
        np.array([0.1, 0.0, 0.0, 0.5]),
        cfg.geometry,
    )
    rudder = model.coefficients(
        _air(),
        np.zeros(3),
        np.array([0.0, 0.0, 0.1, 0.5]),
        cfg.geometry,
    )
    force_b, _moment_b = aero_forces_moments(_air(), elevator, cfg.geometry)

    assert elevator.C_m < cfg.aero.C_m_0
    assert aileron.C_l > 0.0
    assert rudder.C_n > 0.0
    assert force_b[0] < 0.0
    assert force_b[2] < 0.0


@pytest.mark.gate
def test_propulsion_and_atmosphere_reference_values(cfg):
    """Catches density formula errors and implausible thrust scaling."""
    sea_level = isa(0.0)
    assert sea_level.rho == pytest.approx(1.225, rel=1e-4)
    assert isa(1000.0).rho == pytest.approx(1.112, rel=1e-3)
    assert propeller_thrust(0.0, 25.0, sea_level.rho, cfg.propulsion) == 0.0
    assert propeller_thrust(1.0, 0.0, sea_level.rho, cfg.propulsion) > 0.0


@pytest.mark.gate
def test_open_loop_flight_is_plausible(cfg):
    """Catches force/moment wiring errors that make the aircraft diverge."""
    plant = AircraftPlant(cfg)
    controls = _near_trim_controls()

    def controller(_t, _x, _cfg, _rng):
        return controls

    result = simulate(_near_trim_state(), controller, 60.0, cfg, plant, log_every=100)
    altitude = -result.x[:, 2]
    airspeed = np.linalg.norm(result.x[:, IDX_VEL], axis=1)

    assert not result.terminated_early
    assert np.all(np.isfinite(result.x))
    assert abs(altitude[-1] - altitude[0]) < 100.0
    assert airspeed.min() > 10.0
    assert airspeed.max() < 50.0


@pytest.mark.gate
def test_force_scaling_and_zero_speed_behaviour(cfg):
    """Catches q_bar unit mistakes and divide-by-zero force assembly bugs."""
    model = LinearAeroModel(cfg.aero)
    coeffs_slow = model.coefficients(
        _air(V=20.0),
        np.zeros(3),
        np.zeros(4),
        cfg.geometry,
    )
    coeffs_fast = model.coefficients(
        _air(V=40.0),
        np.zeros(3),
        np.zeros(4),
        cfg.geometry,
    )
    F_slow, _ = aero_forces_moments(_air(V=20.0), coeffs_slow, cfg.geometry)
    F_fast, _ = aero_forces_moments(_air(V=40.0), coeffs_fast, cfg.geometry)
    F_zero, M_zero = aero_forces_moments(_air(V=0.0), coeffs_slow, cfg.geometry)

    np.testing.assert_allclose(F_fast, 4.0 * F_slow)
    np.testing.assert_array_equal(F_zero, np.zeros(3))
    np.testing.assert_array_equal(M_zero, np.zeros(3))
