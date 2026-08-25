"""Phase 0 package smoke tests."""

import uav_sim
from uav_sim import constants


def test_version_is_exported():
    assert uav_sim.__version__ == "0.1.0"


def test_standard_gravity_constant():
    assert constants.G0 == 9.80665
