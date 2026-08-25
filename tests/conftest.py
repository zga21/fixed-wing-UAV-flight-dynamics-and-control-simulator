"""Shared pytest configuration."""


def pytest_configure(config):
    config.addinivalue_line("markers", "gate: phase exit criteria")
    config.addinivalue_line("markers", "slow: long-running campaign tests")
