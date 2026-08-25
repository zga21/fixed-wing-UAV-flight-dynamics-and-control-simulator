"""Atmosphere model tests."""

import numpy as np
import pytest

from uav_sim.environment.atmosphere import density_ratio, isa


def test_isa_sea_level_reference_values():
    sea_level = isa(0.0)

    assert sea_level.rho == pytest.approx(1.225, rel=1e-4)
    assert sea_level.temperature == pytest.approx(288.15, rel=1e-6)
    assert sea_level.pressure == pytest.approx(101325.0, rel=1e-4)
    assert sea_level.sound_speed == pytest.approx(340.3, rel=1e-3)


def test_density_decreases_monotonically_to_11km():
    densities = [isa(altitude).rho for altitude in np.linspace(0.0, 11000.0, 50)]

    assert np.all(np.diff(densities) < 0.0)


def test_isa_1000m_reference_density():
    """Checks the standard-atmosphere 1000 m density, about 1.112 kg/m^3."""
    assert isa(1000.0).rho == pytest.approx(1.112, rel=1e-3)


def test_hot_day_reduces_density():
    assert isa(1000.0, delta_isa=15.0).rho < isa(1000.0).rho


def test_negative_altitude_is_clamped():
    assert isa(-100.0) == isa(0.0)


def test_density_ratio_at_sea_level():
    assert density_ratio(0.0) == pytest.approx(1.0, rel=1e-4)
