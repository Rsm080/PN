"""Potential-gradient dynamics and RK4 trajectory integration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import PotentialNetwork


@dataclass(frozen=True)
class PlannerConfig:
    mass: float = 10.0
    damping: float = 5.0
    finite_difference_step: float = 1e-4
    barrier_cap: float = 1e8
    force_limit: float | None = 100.0

    def __post_init__(self) -> None:
        if self.mass <= 0 or self.damping < 0 or self.finite_difference_step <= 0:
            raise ValueError("require mass > 0, damping >= 0, and finite_difference_step > 0")
        if self.barrier_cap <= 0:
            raise ValueError("barrier_cap must be positive")


@dataclass(frozen=True)
class PlanResult:
    time: np.ndarray
    path: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    potential: np.ndarray
    minimum_hard_proximity: np.ndarray

    @property
    def feasible(self) -> bool:
        return bool(np.all(self.minimum_hard_proximity > 0.0))


def path_time(path: np.ndarray, feed_rate: float) -> np.ndarray:
    """Recover timestamps from cutter-contact path, paper Eq. (25)."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[0] < 1:
        raise ValueError("path must have shape (samples, dimensions)")
    if feed_rate <= 0:
        raise ValueError("feed_rate must be positive")
    increments = np.linalg.norm(np.diff(path, axis=0), axis=1) / feed_rate
    return np.concatenate(([0.0], np.cumsum(increments)))


def _finite_value(value: float, cap: float) -> float:
    if np.isnan(value):
        return cap
    if np.isposinf(value):
        return cap
    if np.isneginf(value):
        return -cap
    return float(np.clip(value, -cap, cap))


def potential_force(
    q: np.ndarray,
    p: np.ndarray,
    network: PotentialNetwork,
    config: PlannerConfig,
) -> np.ndarray:
    """F = -grad(P) via central differences, paper Eq. (26)."""
    q = np.asarray(q, dtype=float)
    force = np.empty_like(q)
    delta = config.finite_difference_step
    for index in range(q.size):
        step = np.zeros_like(q)
        step[index] = delta
        plus = _finite_value(network.value(q + step, p), config.barrier_cap)
        minus = _finite_value(network.value(q - step, p), config.barrier_cap)
        force[index] = -(plus - minus) / (2.0 * delta)
    if config.force_limit is not None:
        norm = np.linalg.norm(force)
        if norm > config.force_limit:
            force *= config.force_limit / norm
    return force


def _rhs(
    state: np.ndarray,
    p: np.ndarray,
    network: PotentialNetwork,
    config: PlannerConfig,
) -> np.ndarray:
    dimensions = state.size // 2
    q = state[:dimensions]
    velocity = state[dimensions:]
    acceleration = (potential_force(q, p, network, config) - config.damping * velocity) / config.mass
    return np.concatenate((velocity, acceleration))


def rk4_step(
    state: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    dt: float,
    network: PotentialNetwork,
    config: PlannerConfig,
) -> np.ndarray:
    """One path-interpolated RK4 step, paper Eqs. (27)--(30)."""
    if dt < 0:
        raise ValueError("dt must be non-negative")
    if dt == 0:
        return np.asarray(state, dtype=float).copy()
    midpoint = (np.asarray(p0) + np.asarray(p1)) / 2.0
    k1 = _rhs(state, p0, network, config)
    k2 = _rhs(state + 0.5 * dt * k1, midpoint, network, config)
    k3 = _rhs(state + 0.5 * dt * k2, midpoint, network, config)
    k4 = _rhs(state + dt * k3, p1, network, config)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def plan_path(
    path: np.ndarray,
    feed_rate: float,
    q0: np.ndarray,
    network: PotentialNetwork,
    config: PlannerConfig | None = None,
    qdot0: np.ndarray | None = None,
) -> PlanResult:
    """Plan an attitude trajectory along a discrete cutter-contact path."""
    config = PlannerConfig() if config is None else config
    path = np.asarray(path, dtype=float)
    q0 = np.asarray(q0, dtype=float)
    qdot0 = np.zeros_like(q0) if qdot0 is None else np.asarray(qdot0, dtype=float)
    if qdot0.shape != q0.shape:
        raise ValueError("qdot0 and q0 must have the same shape")
    time = path_time(path, feed_rate)
    states = np.empty((path.shape[0], 2 * q0.size), dtype=float)
    states[0] = np.concatenate((q0, qdot0))
    for index in range(path.shape[0] - 1):
        states[index + 1] = rk4_step(
            states[index], path[index], path[index + 1], time[index + 1] - time[index], network, config
        )
    q = states[:, : q0.size]
    qdot = states[:, q0.size :]
    potential = np.asarray([network.value(qi, pi) for qi, pi in zip(q, path)])
    minimum_h = np.asarray([network.minimum_hard_proximity(qi, pi) for qi, pi in zip(q, path)])
    return PlanResult(time, path, q, qdot, potential, minimum_h)

