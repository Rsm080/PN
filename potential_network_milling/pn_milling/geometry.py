"""SE(3) operators and the paper's tool-attitude manifolds."""

from __future__ import annotations

import numpy as np


def trans(vector: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    """Homogeneous translation, Appendix B, Eq. (B-1)."""
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError("translation vector must have shape (3,)")
    transform = np.eye(4)
    transform[:3, 3] = vector
    return transform


def rot(axis: str, angle: float) -> np.ndarray:
    """Homogeneous rotation, Appendix B, Eqs. (B-2)--(B-4)."""
    c, s = np.cos(angle), np.sin(angle)
    transform = np.eye(4)
    if axis.lower() == "x":
        transform[:3, :3] = ((1, 0, 0), (0, c, -s), (0, s, c))
    elif axis.lower() == "y":
        transform[:3, :3] = ((c, 0, s), (0, 1, 0), (-s, 0, c))
    elif axis.lower() == "z":
        transform[:3, :3] = ((c, -s, 0), (s, c, 0), (0, 0, 1))
    else:
        raise ValueError("axis must be one of 'x', 'y', or 'z'")
    return transform


def attitude_manifold_3d(q: np.ndarray, radius: float, corner_radius: float) -> np.ndarray:
    """General/corner-radius end-mill manifold, paper Eq. (5).

    q = [alpha, beta, gamma] in radians.  The transform maps TCS into ECS.
    """
    alpha, beta, gamma = np.asarray(q, dtype=float)
    radius = float(radius)
    corner_radius = float(corner_radius)
    if radius <= 0 or not 0 <= corner_radius <= radius:
        raise ValueError("require radius > 0 and 0 <= corner_radius <= radius")
    return (
        trans((0.0, 0.0, corner_radius))
        @ rot("z", alpha)
        @ rot("y", beta)
        @ trans((corner_radius - radius, 0.0, 0.0))
        @ rot("z", gamma)
    )


def flat_end_mill_3d(q: np.ndarray, radius: float) -> np.ndarray:
    """Flat-end-mill manifold, paper Eq. (6)."""
    return attitude_manifold_3d(q, radius=radius, corner_radius=0.0)


def ball_end_mill_3d(q: np.ndarray, radius: float) -> np.ndarray:
    """Ball-end-mill manifold, paper Eq. (7)."""
    return attitude_manifold_3d(q, radius=radius, corner_radius=radius)


def compose_tool_pose(t_wcs_ecs: np.ndarray, t_ecs_tcs: np.ndarray) -> np.ndarray:
    """Compose path pose and attitude-manifold pose, paper Eqs. (1) and (4)."""
    left = np.asarray(t_wcs_ecs, dtype=float)
    right = np.asarray(t_ecs_tcs, dtype=float)
    if left.shape != (4, 4) or right.shape != (4, 4):
        raise ValueError("both transforms must have shape (4, 4)")
    return left @ right

