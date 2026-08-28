"""Multiaxis control-allocation tests."""

import numpy as np

from uav_sim.control.allocation import (
    pseudo_inverse_allocation,
    redistributed_allocation,
)


def test_unsaturated_allocation_matches_solve():
    G = np.array([[2.0, 0.0, 0.3], [0.0, -3.0, 0.0], [-0.2, 0.0, 1.5]])
    demand = np.array([0.2, -0.3, 0.1])
    limits = np.tile([-1.0, 1.0], (3, 1))

    result = redistributed_allocation(G, demand, limits)

    np.testing.assert_allclose(result.delta, np.linalg.solve(G, demand), atol=1e-10)
    np.testing.assert_allclose(result.residual, 0.0, atol=1e-10)
    assert not np.any(result.saturated)


def test_saturation_is_clamped_and_residual_is_reported():
    G = np.array([[2.0, 0.0, 0.3], [0.0, -3.0, 0.0], [-0.2, 0.0, 1.5]])
    demand = np.array([2.0, -0.3, 0.1])
    limits = np.array([[-0.2, 0.2], [-0.5, 0.5], [-0.4, 0.4]])

    result = redistributed_allocation(G, demand, limits)

    assert result.delta[0] == limits[0, 1]
    assert result.saturated[0]
    assert np.linalg.norm(result.residual) > 0.0
    assert result.iterations <= 5
    np.testing.assert_allclose((G @ result.delta)[1], demand[1], atol=1e-12)


def test_fully_saturated_case_terminates():
    result = redistributed_allocation(
        np.eye(3), np.full(3, 10.0), np.tile([-0.1, 0.1], (3, 1)), max_iter=3
    )
    np.testing.assert_allclose(result.delta, 0.1)
    assert np.all(result.saturated)
    assert result.iterations <= 3


def test_weighting_shifts_redundant_surface_effort():
    G = np.array([[1.0, 1.0], [0.0, 1.0]])
    demand = np.array([1.0, 0.2])
    unweighted = pseudo_inverse_allocation(G, demand)
    weighted = pseudo_inverse_allocation(G, demand, np.diag([10.0, 1.0]))

    np.testing.assert_allclose(G @ weighted, demand)
    assert weighted[0] <= unweighted[0] + 1e-12
