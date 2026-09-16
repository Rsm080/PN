from __future__ import annotations

import unittest

import numpy as np

from pn_milling.constraints import box_group
from pn_milling.geometry import attitude_manifold_3d, ball_end_mill_3d, flat_end_mill_3d
from pn_milling.network import PotentialGroup, PotentialNetwork
from pn_milling.planner import PlannerConfig, path_time, plan_path
from pn_milling.potentials import (
    hard_potential,
    hard_potential_first_derivative,
    hard_potential_second_derivative,
    obstacle_risk_state,
    soft_potential,
)


class PotentialTests(unittest.TestCase):
    def test_hard_potential_regions(self) -> None:
        values = hard_potential(np.asarray((-0.1, 0.5, 1.0, 2.0)), stiffness=2.0)
        self.assertTrue(np.isinf(values[0]))
        self.assertAlmostEqual(values[1], 0.5)
        self.assertAlmostEqual(values[2], 0.0)
        self.assertAlmostEqual(values[3], 0.0)

    def test_hard_derivatives_match_finite_difference(self) -> None:
        h = 0.63
        eps = 1e-5
        first_numeric = (hard_potential(np.asarray([h + eps]))[0] - hard_potential(np.asarray([h - eps]))[0]) / (2 * eps)
        second_numeric = (hard_potential(np.asarray([h + eps]))[0] - 2 * hard_potential(np.asarray([h]))[0] + hard_potential(np.asarray([h - eps]))[0]) / eps**2
        self.assertAlmostEqual(first_numeric, hard_potential_first_derivative(np.asarray([h]))[0], places=6)
        self.assertAlmostEqual(second_numeric, hard_potential_second_derivative(np.asarray([h]))[0], places=4)

    def test_c2_at_safe_boundary(self) -> None:
        h = np.asarray([1.0])
        self.assertAlmostEqual(hard_potential(h)[0], 0.0)
        self.assertAlmostEqual(hard_potential_first_derivative(h)[0], 0.0)
        self.assertAlmostEqual(hard_potential_second_derivative(h)[0], 0.0)

    def test_soft_potential(self) -> None:
        self.assertAlmostEqual(soft_potential(np.asarray([1.0]), tolerance=-1.0)[0], 0.0)
        self.assertTrue(np.isinf(soft_potential(np.asarray([-1.0]), tolerance=-1.0)[0]))

    def test_ors_regions(self) -> None:
        np.testing.assert_array_equal(obstacle_risk_state(np.asarray((1.1, 0.5, 0.0))), (1.0, 0.5, 0.0))


class GeometryTests(unittest.TestCase):
    def test_special_cases_match_general_manifold(self) -> None:
        q = np.asarray((0.2, -0.3, 0.4))
        radius = 0.0125
        np.testing.assert_allclose(flat_end_mill_3d(q, radius), attitude_manifold_3d(q, radius, 0.0))
        np.testing.assert_allclose(ball_end_mill_3d(q, radius), attitude_manifold_3d(q, radius, radius))

    def test_rotation_is_orthonormal(self) -> None:
        transform = attitude_manifold_3d(np.asarray((0.2, -0.3, 0.4)), 0.0125, 0.005)
        rotation = transform[:3, :3]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)


class NetworkAndPlannerTests(unittest.TestCase):
    def test_homogeneous_mean_and_heterogeneous_weighting(self) -> None:
        group1 = PotentialGroup("g1", "hard", lambda _q, _p: np.asarray((0.5, 1.0)), stiffness=1.0)
        group2 = PotentialGroup("g2", "hard", lambda _q, _p: np.asarray((0.5,)), weight=3.0, stiffness=1.0)
        network = PotentialNetwork((group1, group2))
        expected_group1 = (0.25 + 0.0) / 2.0
        expected_group2 = 0.25
        expected = (expected_group1 + 3.0 * expected_group2) / 4.0
        self.assertAlmostEqual(network.value(np.zeros(1), np.zeros(1)), expected)

    def test_path_time(self) -> None:
        path = np.asarray(((0, 0, 0), (3, 4, 0), (3, 4, 2)), dtype=float)
        np.testing.assert_allclose(path_time(path, 2.0), (0.0, 2.5, 3.5))

    def test_stationary_safe_state_remains_stationary(self) -> None:
        group = box_group("box", (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), 0.2)
        network = PotentialNetwork((group,))
        path = np.column_stack((np.linspace(0, 1, 21), np.zeros((21, 2))))
        result = plan_path(path, 1.0, np.zeros(3), network, PlannerConfig(force_limit=None))
        np.testing.assert_allclose(result.q, 0.0, atol=1e-12)
        self.assertTrue(result.feasible)


if __name__ == "__main__":
    unittest.main()

