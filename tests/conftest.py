"""Shared pytest configuration."""

import pytest

from uav_sim.config import load_aircraft


def pytest_configure(config):
    config.addinivalue_line("markers", "gate: phase exit criteria")
    config.addinivalue_line("markers", "slow: long-running campaign tests")


@pytest.fixture(scope="session")
def cfg():
    return load_aircraft("config/aircraft_v1.yaml")
