"""Reusable constraint constructors returning PotentialGroup objects."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .network import PotentialGroup


def _as_vector(value: float | np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(size, float(array))
    if array.shape != (size,):
        raise ValueError(f"{name} must be scalar or shape ({size},)")
    return array


def box_group(
    name: str,
    lower: np.ndarray | list[float],
    upper: np.ndarray | list[float],
    risk_distance: float | np.ndarray,
    *,
    weight: float = 1.0,
    stiffness: float = 1.0,
) -> PotentialGroup:
    """Hard lower/upper limits for attitude coordinates."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape or lower.ndim != 1 or np.any(lower >= upper):
        raise ValueError("lower and upper must be same-shape vectors with lower < upper")
    risk = _as_vector(risk_distance, lower.size, "risk_distance")
    if np.any(risk <= 0):
        raise ValueError("risk_distance must be positive")

    def proximity_fn(q: np.ndarray, _p: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        return np.concatenate(((q - lower) / risk, (upper - q) / risk))

    return PotentialGroup(name, "hard", proximity_fn, weight=weight, stiffness=stiffness)


def clearance_group(
    name: str,
    distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    minimum_distance: float | np.ndarray,
    risk_distance: float | np.ndarray,
    *,
    weight: float = 1.0,
    stiffness: float = 1.0,
) -> PotentialGroup:
    """Hard clearance d(q,p) > d_min, compatible with Appendix A.2."""

    def proximity_fn(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        distances = np.asarray(distance_fn(q, p), dtype=float)
        return (distances - np.asarray(minimum_distance)) / np.asarray(risk_distance)

    return PotentialGroup(name, "hard", proximity_fn, weight=weight, stiffness=stiffness)


def upper_bound_group(
    name: str,
    value_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    upper: float | np.ndarray,
    risk_scale: float | np.ndarray,
    *,
    kind: str = "hard",
    weight: float = 1.0,
    stiffness: float = 1.0,
    tolerance: float = -1.0,
    flexibility: float = 1.0,
) -> PotentialGroup:
    """Constraint f(q,p) < b expressed directly as Eq. (11)."""

    def proximity_fn(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        return (np.asarray(upper) - np.asarray(value_fn(q, p), dtype=float)) / np.asarray(risk_scale)

    return PotentialGroup(
        name,
        kind,  # type: ignore[arg-type]
        proximity_fn,
        weight=weight,
        stiffness=stiffness,
        tolerance=tolerance,
        flexibility=flexibility,
    )


def tracking_band_group(
    name: str,
    target_fn: Callable[[np.ndarray], np.ndarray],
    half_width: float | np.ndarray,
    *,
    tolerance: float = -3.0,
    weight: float = 1.0,
    flexibility: float = 1.0,
) -> PotentialGroup:
    """Soft |q-q_ref(p)| < half_width performance constraint."""

    def proximity_fn(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        target = np.asarray(target_fn(p), dtype=float)
        width = _as_vector(half_width, target.size, "half_width")
        return (width - np.abs(np.asarray(q) - target)) / width

    return PotentialGroup(
        name,
        "soft",
        proximity_fn,
        weight=weight,
        tolerance=tolerance,
        flexibility=flexibility,
    )


def moving_sphere_distance(center_fn: Callable[[np.ndarray], np.ndarray], radius: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Distance-to-surface callback for a moving spherical obstacle in q-space."""
    if radius <= 0:
        raise ValueError("radius must be positive")

    def distance_fn(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        center = np.asarray(center_fn(p), dtype=float)
        return np.asarray([np.linalg.norm(np.asarray(q) - center)])

    return distance_fn

