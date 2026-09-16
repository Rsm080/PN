"""Proximity normalization and potential functions from the paper."""

from __future__ import annotations

import numpy as np


def proximity(value: np.ndarray, boundary: float | np.ndarray, reference: float | np.ndarray) -> np.ndarray:
    """H = (b - f) / D0, paper Eq. (11)."""
    value = np.asarray(value, dtype=float)
    boundary = np.asarray(boundary, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if np.any(reference <= 0):
        raise ValueError("proximity reference D0 must be positive")
    return (boundary - value) / reference


def obstacle_feasibility(distance_to_boundary: np.ndarray) -> np.ndarray:
    """Binary feasible/obstacle classifier f_OF, paper Eq. (10)."""
    return (np.asarray(distance_to_boundary) > 0).astype(float)


def obstacle_risk_state(h: np.ndarray) -> np.ndarray:
    """Safe/risk/obstacle classifier f_ORS, corrected paper Eq. (12)."""
    h = np.asarray(h, dtype=float)
    return np.where(h > 1.0, 1.0, np.where(h > 0.0, 0.5, 0.0))


def hard_potential(h: np.ndarray, stiffness: float = 1.0) -> np.ndarray:
    """Hard potential sigma_h, paper Eq. (15)."""
    if stiffness <= 0:
        raise ValueError("stiffness must be positive")
    h = np.asarray(h, dtype=float)
    result = np.zeros_like(h)
    obstacle = h <= 0.0
    risk = (h > 0.0) & (h <= 1.0)
    result[obstacle] = np.inf
    result[risk] = stiffness * (1.0 - h[risk]) ** 3 / h[risk]
    return result


def hard_potential_first_derivative(h: np.ndarray, stiffness: float = 1.0) -> np.ndarray:
    """First derivative in the risk region, paper Eq. (16)."""
    h = np.asarray(h, dtype=float)
    result = np.zeros_like(h)
    risk = (h > 0.0) & (h <= 1.0)
    hr = h[risk]
    result[h <= 0.0] = -np.inf
    result[risk] = -stiffness * (2.0 * hr + 1.0) * (hr - 1.0) ** 2 / hr**2
    return result


def hard_potential_second_derivative(h: np.ndarray, stiffness: float = 1.0) -> np.ndarray:
    """Correct derivative of Eq. (15).

    The paper prints 2*lambda*(H^3-1)/H^3; direct differentiation gives
    2*lambda*(1-H^3)/H^3.
    """
    h = np.asarray(h, dtype=float)
    result = np.zeros_like(h)
    risk = (h > 0.0) & (h <= 1.0)
    hr = h[risk]
    result[h <= 0.0] = np.inf
    result[risk] = 2.0 * stiffness * (1.0 - hr**3) / hr**3
    return result


def soft_potential(h: np.ndarray, tolerance: float = -1.0, flexibility: float = 1.0) -> np.ndarray:
    """Soft potential sigma_s, paper Eq. (17)."""
    if tolerance >= 1.0:
        raise ValueError("soft-potential tolerance rho must be less than 1")
    if flexibility <= 0:
        raise ValueError("flexibility must be positive")
    h = np.asarray(h, dtype=float)
    result = np.empty_like(h)
    invalid = h <= tolerance
    result[invalid] = np.inf
    valid = ~invalid
    result[valid] = -flexibility * np.log((h[valid] - tolerance) / (1.0 - tolerance))
    return result

