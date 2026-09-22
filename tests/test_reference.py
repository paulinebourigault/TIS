from __future__ import annotations

import unittest
import json
from pathlib import Path

import numpy as np

from tis.allocation import (
    design_constant,
    integer_allocation,
    largest_remainder,
    normalized_weights,
    regularized_weights,
)
from tis.engine import compute_influences, solve_probability_form
from tis.environments import (
    CLIFF_PRIMARY_ACTIONS,
    CLIFF_SAFE_ACTIONS,
    cliffwalking_primary,
    cliffwalking_safe,
    controlled_shared,
    controlled_untied_v1,
    frozenlake_8x8,
    inventory_untied_v1,
    rainy_taxi,
)
from tis.validation import validate_environment, validate_population_reference


class AllocationTests(unittest.TestCase):
    def test_largest_remainder(self) -> None:
        weights = np.array([0.51, 0.29, 0.20])
        counts = largest_remainder(17, weights)
        self.assertEqual(int(counts.sum()), 17)
        self.assertTrue(np.all(np.abs(counts - 17 * weights) < 1))

    def test_integer_floor_and_budget(self) -> None:
        counts = integer_allocation(103, np.array([0.1, 0.2, 0.7]), 2)
        self.assertEqual(int(counts.sum()), 103)
        self.assertTrue(np.all(counts >= 2))

    def test_oracle_design(self) -> None:
        scales = np.array([0.2, 1.0, 0.5])
        oracle = normalized_weights(scales)
        uniform = np.full(3, 1 / 3)
        self.assertLessEqual(
            design_constant(scales, oracle), design_constant(scales, uniform)
        )
        regularized = regularized_weights(scales, 0.1)
        self.assertAlmostEqual(float(regularized.sum()), 1.0)
        self.assertTrue(np.all(regularized > 0))


class BellmanTests(unittest.TestCase):
    def test_probability_and_stop_loss_forms_agree(self) -> None:
        model = controlled_shared()
        bundle = compute_influences(model, 0.1)
        eta = solve_probability_form(model)
        np.testing.assert_allclose(
            eta[model.horizon, model.initial_state],
            bundle.solution.probabilities,
            atol=1e-10,
        )

    def test_covariance_identities(self) -> None:
        model = controlled_shared()
        bundle = compute_influences(model, 0.1)
        for index, group in enumerate(bundle.groups):
            component = bundle.tail_components[index]
            probabilities = model.kernels[group].probabilities
            covariance = component.T @ (probabilities[:, None] * component)
            shared = float(np.ones(model.horizon) @ covariance @ np.ones(model.horizon))
            dropped = float(np.trace(covariance))
            self.assertAlmostEqual(shared, bundle.tail_scales[index] ** 2, places=10)
            self.assertAlmostEqual(dropped, bundle.no_cov_scales[index] ** 2, places=10)

    def test_finite_differences(self) -> None:
        for name, alpha in (
            ("controlled_shared", 0.1),
            ("controlled_untied_v1", 0.1),
            ("inventory_untied_v1", 0.1),
            ("frozenlake_8x8", 0.2),
        ):
            report = validate_environment(name, alpha)
            self.assertTrue(report["passed"], report)

    def test_untied_layer_state_factorization(self) -> None:
        for model in (controlled_untied_v1(), inventory_untied_v1()):
            self.assertEqual(model.group_mode, "untied_layer_state")
            bundle = compute_influences(model, 0.1)
            np.testing.assert_allclose(
                bundle.tail_scales,
                bundle.reachability_scales * bundle.local_scales,
                atol=1e-12,
                rtol=0,
            )
            for component in bundle.tail_components:
                self.assertLessEqual(
                    int(np.count_nonzero(np.any(np.abs(component) > 1e-14, axis=0))),
                    1,
                )


class PublicEnvironmentTests(unittest.TestCase):
    def test_frozen_policy_tables_are_literal_and_complete(self) -> None:
        self.assertEqual(len(CLIFF_PRIMARY_ACTIONS), 48)
        self.assertEqual(len(CLIFF_SAFE_ACTIONS), 48)
        self.assertEqual(tuple(CLIFF_PRIMARY_ACTIONS[24:35]), (1,) * 11)
        self.assertEqual(CLIFF_PRIMARY_ACTIONS[35], 2)
        self.assertEqual(tuple(CLIFF_SAFE_ACTIONS[12:23]), (1,) * 11)
        self.assertEqual(CLIFF_SAFE_ACTIONS[23], 2)
        specification = json.loads(
            (Path(__file__).resolve().parents[1] / "specs" / "cliffwalking_policies.json").read_text()
        )
        self.assertEqual(tuple(specification["primary_preferred_actions"]), CLIFF_PRIMARY_ACTIONS)
        self.assertEqual(tuple(specification["safe_preferred_actions"]), CLIFF_SAFE_ACTIONS)

    def test_public_structure(self) -> None:
        frozen = frozenlake_8x8()
        primary = cliffwalking_primary()
        safe = cliffwalking_safe()
        taxi = rainy_taxi()
        self.assertEqual(frozen.group_count, 50)
        self.assertEqual(primary.group_count, 149)
        self.assertEqual(safe.group_count, 149)
        self.assertEqual(taxi.group_count, 37)
        self.assertEqual(taxi.metadata["root_official_state"], 4)
        self.assertEqual(taxi.metadata["maximum_dry_completion_steps"], 18)
        self.assertEqual(len(taxi.metadata["maximum_dry_initial_states"]), 8)

    def test_manuscript_population_constants(self) -> None:
        references = (
            ("frozenlake_8x8", .2, 50, .3147769, .0629554, 3.92589),
            ("cliffwalking_primary", .1, 149, 11.8709354, .0139085, 84.6973),
            ("rainy_taxi", .1, 37, 9.0541409, .0121392, 2.33938),
        )
        for values in references:
            with self.subTest(environment=values[0]):
                report = validate_population_reference(*values)
                self.assertTrue(report["passed"], report)


class ReplacementEnvironmentTests(unittest.TestCase):
    def test_frozen_replacement_structure(self) -> None:
        controlled = controlled_untied_v1()
        inventory = inventory_untied_v1()
        self.assertEqual(controlled.group_count, 21)
        self.assertEqual(inventory.group_count, 41)
        self.assertEqual(controlled.query_groups[0], (5, 0))
        self.assertTrue(
            all(np.all(kernel.probabilities > 0) for kernel in controlled.kernels.values())
        )
        self.assertEqual(
            set(np.unique(np.concatenate([kernel.rewards for kernel in inventory.kernels.values()]))),
            {0.0, 0.5, 1.0},
        )

    def test_replacement_population_constants(self) -> None:
        references = (
            ("controlled_untied_v1", .1, 21, .9086716125743333, .010685396723788224, 1.7233238733483527),
            ("inventory_untied_v1", .1, 41, 1.1512371171695401, .04362019546849408, 1.8235977736549085),
        )
        for values in references:
            with self.subTest(environment=values[0]):
                report = validate_population_reference(
                    *values, tolerance=1e-12, ratio_tolerance=1e-12
                )
                self.assertTrue(report["passed"], report)


if __name__ == "__main__":
    unittest.main()
