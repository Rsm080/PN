"""The paper's four-layer potential-network composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np

from .potentials import hard_potential, soft_potential

ProximityFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PotentialGroup:
    """One homogeneous tensor constraint before/after potential transfer."""

    name: str
    kind: Literal["hard", "soft"]
    proximity_fn: ProximityFunction
    weight: float = 1.0
    stiffness: float = 1.0
    tolerance: float = -1.0
    flexibility: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in ("hard", "soft"):
            raise ValueError("kind must be 'hard' or 'soft'")
        if self.weight < 0:
            raise ValueError("group weight must be non-negative")

    def proximities(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        values = np.asarray(self.proximity_fn(q, p), dtype=float)
        if values.size == 0:
            raise ValueError(f"group {self.name!r} returned no constraint elements")
        return values

    def element_potentials(self, q: np.ndarray, p: np.ndarray) -> np.ndarray:
        h = self.proximities(q, p)
        if self.kind == "hard":
            return hard_potential(h, self.stiffness)
        return soft_potential(h, self.tolerance, self.flexibility)

    def value(self, q: np.ndarray, p: np.ndarray) -> float:
        """Homogeneous composition: global mean, paper Eqs. (18)--(19)."""
        return float(np.mean(self.element_potentials(q, p)))


class PotentialNetwork:
    """Heterogeneous and rigid/flexible composition, Eqs. (20)--(21)."""

    def __init__(self, groups: Sequence[PotentialGroup], performance_weight: float = 0.0):
        self.groups = tuple(groups)
        self.performance_weight = float(performance_weight)
        if not self.groups:
            raise ValueError("at least one potential group is required")
        if self.performance_weight < 0:
            raise ValueError("performance weight mu_P must be non-negative")
        if not any(group.kind == "hard" and group.weight > 0 for group in self.groups):
            raise ValueError("at least one positive-weight hard group is required")

    def _weighted_kind_value(self, kind: Literal["hard", "soft"], q: np.ndarray, p: np.ndarray) -> float | None:
        selected = [group for group in self.groups if group.kind == kind and group.weight > 0]
        if not selected:
            return None
        numerator = sum(group.weight * group.value(q, p) for group in selected)
        denominator = sum(group.weight for group in selected)
        return float(numerator / denominator)

    def components(self, q: np.ndarray, p: np.ndarray) -> dict[str, float]:
        hard = self._weighted_kind_value("hard", q, p)
        soft = self._weighted_kind_value("soft", q, p)
        if hard is None:
            raise RuntimeError("network has no active hard group")
        if soft is None or self.performance_weight == 0.0:
            total = hard
        else:
            total = (hard + self.performance_weight * soft) / (1.0 + self.performance_weight)
        return {"hard": hard, "soft": np.nan if soft is None else soft, "total": total}

    def value(self, q: np.ndarray, p: np.ndarray) -> float:
        return self.components(np.asarray(q, dtype=float), np.asarray(p, dtype=float))["total"]

    def minimum_hard_proximity(self, q: np.ndarray, p: np.ndarray) -> float:
        values = [
            float(np.min(group.proximities(q, p)))
            for group in self.groups
            if group.kind == "hard" and group.weight > 0
        ]
        return min(values)

