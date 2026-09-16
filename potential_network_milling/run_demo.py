"""Self-contained dynamic constrained-space benchmark for the PN reproduction."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pn_milling.constraints import box_group, clearance_group, moving_sphere_distance, tracking_band_group
from pn_milling.network import PotentialNetwork
from pn_milling.planner import PlannerConfig, plan_path
from pn_milling.plotting import save_overview


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def build_path(samples: int = 401) -> np.ndarray:
    """A 1.3 m straight cutter-contact path, matching the paper's benchmark length."""
    x = np.linspace(0.0, 1.3, samples)
    return np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))


def normalized_progress(p: np.ndarray) -> float:
    return float(np.clip(p[0] / 1.3, 0.0, 1.0))


def obstacle_center(p: np.ndarray) -> np.ndarray:
    """A moving forbidden region in attitude space as the cutter advances."""
    s = normalized_progress(p)
    return np.asarray((0.12 * np.sin(2 * np.pi * s), 0.95 - 1.9 * s, 0.08 * np.cos(2 * np.pi * s)))


def preferred_attitude(p: np.ndarray) -> np.ndarray:
    """A smooth performance preference analogous to deformation minimization."""
    s = normalized_progress(p)
    return np.asarray((0.08 * np.sin(2 * np.pi * s), 0.0, 0.12 * np.sin(np.pi * s)))


def build_network(performance_weight: float = 2.0) -> PotentialNetwork:
    attitude_limits = box_group(
        "attitude range",
        lower=np.deg2rad((-42.0, -42.0, -55.0)),
        upper=np.deg2rad((42.0, 42.0, 55.0)),
        risk_distance=np.deg2rad((12.0, 12.0, 15.0)),
        stiffness=3.0,
    )
    moving_obstacle = clearance_group(
        "dynamic restricted space",
        moving_sphere_distance(obstacle_center, radius=0.23),
        minimum_distance=0.23,
        risk_distance=0.20,
        stiffness=8.0,
    )
    performance = tracking_band_group(
        "preferred machining attitude",
        preferred_attitude,
        half_width=np.deg2rad((18.0, 18.0, 22.0)),
        tolerance=-4.0,
        flexibility=0.8,
    )
    return PotentialNetwork(
        (attitude_limits, moving_obstacle, performance),
        performance_weight=performance_weight,
    )


def kinematic_metrics(result) -> dict[str, float]:
    """Aggregate range/SD metrics used by the paper's E1/E2 comparison."""
    velocity = result.qdot
    acceleration = np.gradient(velocity, result.time, axis=0, edge_order=2)
    jerk = np.gradient(acceleration, result.time, axis=0, edge_order=2)
    metrics: dict[str, float] = {}
    for name, values in (("velocity", velocity), ("acceleration", acceleration), ("jerk", jerk)):
        per_axis_range = np.ptp(values, axis=0)
        per_axis_sd = np.std(values, axis=0)
        metrics[f"{name}_mean_range"] = float(np.mean(per_axis_range))
        metrics[f"{name}_mean_sd"] = float(np.mean(per_axis_sd))
    return metrics


def save_result(result, directory: Path) -> dict[str, float]:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / "trajectory.npz",
        time=result.time,
        path=result.path,
        q=result.q,
        qdot=result.qdot,
        potential=result.potential,
        minimum_hard_proximity=result.minimum_hard_proximity,
    )
    with (directory / "trajectory.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "path_x_m", "path_y_m", "path_z_m", "alpha_rad", "beta_rad", "gamma_rad", "alpha_dot", "beta_dot", "gamma_dot", "potential", "min_hard_proximity"))
        for row in zip(result.time, result.path, result.q, result.qdot, result.potential, result.minimum_hard_proximity):
            t, p, q, qdot, potential, minimum_h = row
            writer.writerow((t, *p, *q, *qdot, potential, minimum_h))
    max_speed = float(np.max(np.linalg.norm(result.qdot, axis=1)))
    summary = (
        f"feasible={result.feasible}\n"
        f"samples={len(result.time)}\n"
        f"duration_s={result.time[-1]:.6f}\n"
        f"minimum_hard_proximity={np.min(result.minimum_hard_proximity):.6f}\n"
        f"maximum_attitude_speed_rad_s={max_speed:.6f}\n"
        f"final_q_rad={np.array2string(result.q[-1], precision=6)}\n"
    )
    metrics = kinematic_metrics(result)
    summary += "".join(f"{key}={value:.9g}\n" for key, value in metrics.items())
    (directory / "summary.txt").write_text(summary, encoding="utf-8")
    save_overview(directory / "overview.png", result.time, result.q, result.potential, result.minimum_hard_proximity)
    return metrics


def main() -> None:
    path = build_path()
    config = PlannerConfig(
        mass=10.0,
        damping=5.0,
        finite_difference_step=1e-4,
        barrier_cap=1e7,
        force_limit=80.0,
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    cases = (("E1_mu0", 0.0), ("E2_mu5", 5.0))
    all_metrics: dict[str, dict[str, float]] = {}
    results = {}
    for label, performance_weight in cases:
        result = plan_path(
            path,
            feed_rate=0.10,
            q0=np.asarray((0.0, -0.08, 0.0)),
            network=build_network(performance_weight=performance_weight),
            config=config,
        )
        results[label] = result
        all_metrics[label] = save_result(result, RESULTS / label)

    with (RESULTS / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        metric_names = tuple(all_metrics["E1_mu0"].keys())
        writer.writerow(("case", "mu_P", "feasible", "minimum_H", *metric_names))
        for label, performance_weight in cases:
            result = results[label]
            writer.writerow((label, performance_weight, result.feasible, np.min(result.minimum_hard_proximity), *(all_metrics[label][name] for name in metric_names)))

    lines = ["Paper-style E1/E2 comparison (same M=10, C=5):"]
    for label, performance_weight in cases:
        result = results[label]
        lines.append(f"  {label}: mu_P={performance_weight:g}, feasible={result.feasible}, min_H={np.min(result.minimum_hard_proximity):.6f}")
        for key, value in all_metrics[label].items():
            lines.append(f"    {key}={value:.9g}")
    comparison_text = "\n".join(lines) + "\n"
    (RESULTS / "comparison.txt").write_text(comparison_text, encoding="utf-8")
    print(comparison_text, end="")
    print(f"results={RESULTS}")


if __name__ == "__main__":
    main()
