"""Adapters for the paper's robot-specific constraints (Appendix A).

The paper does not publish the calibrated robot/collision/stiffness data, so the
actual models are supplied as callbacks by the user or a robotics package.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .constraints import clearance_group, upper_bound_group
from .network import PotentialGroup


def joint_limit_group(
    ik_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    risk_distance: float | np.ndarray,
    *,
    drive_map: np.ndarray | None = None,
    weight: float = 1.0,
    stiffness: float = 1.0,
) -> PotentialGroup:
    """Joint/drive range constraints corresponding to Eqs. (A-1)--(A-4)."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    risk = np.broadcast_to(np.asarray(risk_distance, dtype=float), lower.shape)
    matrix = None if drive_map is None else np.asarray(drive_map, dtype=float)

    def proximity_fn(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        joints = np.asarray(ik_fn(q, p), dtype=float)
        coordinates = joints if matrix is None else matrix @ joints
        return np.concatenate(((coordinates - lower) / risk, (upper - coordinates) / risk))

    return PotentialGroup("joint/drive limits", "hard", proximity_fn, weight=weight, stiffness=stiffness)


def collision_group(
    distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    minimum_distance: float | np.ndarray,
    risk_distance: float | np.ndarray,
    **kwargs: float,
) -> PotentialGroup:
    """Convex-body clearance constraints, Eqs. (A-5)--(A-9)."""
    return clearance_group("collision clearance", distance_fn, minimum_distance, risk_distance, **kwargs)


def manipulability_group(
    jacobian_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    maximum_condition: float,
    risk_scale: float,
    **kwargs: float,
) -> PotentialGroup:
    """Frobenius condition-number constraint, Eqs. (A-10)--(A-11)."""

    def condition(q: np.ndarray, p: np.ndarray) -> np.ndarray:
        jacobian = np.asarray(jacobian_fn(q, p), dtype=float)
        try:
            inverse = np.linalg.inv(jacobian)
            value = np.linalg.norm(jacobian, "fro") * np.linalg.norm(inverse, "fro")
        except np.linalg.LinAlgError:
            value = np.inf
        return np.asarray([value])

    return upper_bound_group("Jacobian condition", condition, maximum_condition, risk_scale, **kwargs)


def deformation_group(
    deformation_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    maximum_deformation: float,
    risk_scale: float,
    *,
    tolerance: float = -1.0,
    weight: float = 1.0,
    flexibility: float = 1.0,
) -> PotentialGroup:
    """Soft normal-deformation constraint, Eqs. (A-12)--(A-16)."""
    return upper_bound_group(
        "machining deformation",
        deformation_fn,
        maximum_deformation,
        risk_scale,
        kind="soft",
        tolerance=tolerance,
        weight=weight,
        flexibility=flexibility,
    )

