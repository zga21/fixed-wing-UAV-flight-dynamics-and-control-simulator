"""Saturation-aware multiaxis control allocation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AllocationResult:
    """Allocated surface commands and any unachievable virtual control."""

    delta: np.ndarray
    residual: np.ndarray
    saturated: np.ndarray
    iterations: int


def pseudo_inverse_allocation(
    G: np.ndarray,
    nu: np.ndarray,
    W: np.ndarray | None = None,
) -> np.ndarray:
    """Return the weighted least-squares surface allocation."""
    matrix, demand = _validated_inputs(G, nu)
    weights = np.eye(matrix.shape[1]) if W is None else np.asarray(W, dtype=float)
    if weights.shape != (matrix.shape[1], matrix.shape[1]):
        raise ValueError("W must be square with one row per control surface")
    if not np.allclose(weights, weights.T) or np.any(np.linalg.eigvalsh(weights) <= 0):
        raise ValueError("W must be symmetric positive-definite")
    weighted_transpose = np.linalg.solve(weights, matrix.T)
    return weighted_transpose @ np.linalg.pinv(matrix @ weighted_transpose) @ demand


def redistributed_allocation(
    G: np.ndarray,
    nu: np.ndarray,
    limits: np.ndarray,
    max_iter: int = 5,
    W: np.ndarray | None = None,
) -> AllocationResult:
    """Allocate, clamp saturated surfaces, and redistribute the remainder."""
    matrix, demand = _validated_inputs(G, nu)
    bounds = np.asarray(limits, dtype=np.float64)
    n_controls = matrix.shape[1]
    if bounds.shape != (n_controls, 2):
        raise ValueError(f"limits must have shape ({n_controls}, 2)")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("control lower limits must not exceed upper limits")
    if max_iter < 1:
        raise ValueError("max_iter must be at least one")

    delta = np.zeros(n_controls, dtype=np.float64)
    saturated = np.zeros(n_controls, dtype=bool)
    iterations = 0
    for iteration in range(1, max_iter + 1):
        iterations = iteration
        free = ~saturated
        if not np.any(free):
            break
        remainder = demand - matrix[:, saturated] @ delta[saturated]
        free_weights = None
        if W is not None:
            weight_matrix = np.asarray(W, dtype=np.float64)
            free_weights = weight_matrix[np.ix_(free, free)]
        proposal = pseudo_inverse_allocation(matrix[:, free], remainder, free_weights)
        free_indices = np.flatnonzero(free)
        below = proposal < bounds[free, 0]
        above = proposal > bounds[free, 1]
        newly_saturated = below | above
        delta[free] = np.clip(proposal, bounds[free, 0], bounds[free, 1])
        if not np.any(newly_saturated):
            break
        saturated[free_indices[newly_saturated]] = True

    residual = demand - matrix @ delta
    saturated |= np.isclose(delta, bounds[:, 0], atol=1e-12)
    saturated |= np.isclose(delta, bounds[:, 1], atol=1e-12)
    return AllocationResult(delta, residual, saturated, iterations)


def _validated_inputs(G: np.ndarray, nu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(G, dtype=np.float64)
    demand = np.asarray(nu, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("G must be a two-dimensional matrix")
    if demand.shape != (matrix.shape[0],):
        raise ValueError(f"nu must have shape ({matrix.shape[0]},)")
    return matrix, demand
