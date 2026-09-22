"""Adaptive plug-in Neyman: the exploration floor must survive rounding.

A batched rule that applies largest remainder to each batch
independently can discard the lambda/G exploration fractions afresh in
every batch when one score dominates, leaving groups at their two
initial pulls. The cumulative-quota rule accumulates fractional
entitlements across batches; this test checks the floor survives in a
G=10, lambda=.072 configuration with one dominant score.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from tis.llm_study.runner import _adaptive_plugin_neyman


def _stub_model(groups: int):
    kernels = {}
    for g in range(groups):
        kernels[(0, g)] = SimpleNamespace(
            probabilities=np.array([0.5, 0.5]), size=2)
    return SimpleNamespace(query_groups=tuple((0, g) for g in range(groups)),
                           kernels=kernels)


class AdaptiveNeymanExplorationTests(unittest.TestCase):
    def test_exploration_floor_survives_batched_rounding(self):
        groups, exploration, main_budget = 10, 0.072, 2000
        model = _stub_model(groups)
        # One dominant frozen score (sd .5), the rest constant: weights
        # are about [.9352, .0072 x 9].
        scores = tuple([np.array([0.0, 1.0])] +
                       [np.array([0.0, 0.0])] * (groups - 1))
        pulls, empirical, _ = _adaptive_plugin_neyman(
            model, scores, main_budget, exploration,
            np.random.default_rng(7))
        self.assertEqual(int(pulls.sum()), main_budget)
        # Nominal sustained floor: 2 initial + (lambda/G) x adaptive budget.
        nominal_floor = 2 + (exploration / groups) * (main_budget - 2 * groups)
        self.assertGreaterEqual(
            int(pulls.min()), int(0.8 * nominal_floor),
            f"exploration floor lost: pulls={pulls.tolist()}, "
            f"nominal per-group floor {nominal_floor:.1f}")
        # The dominant group must still dominate.
        self.assertGreater(pulls[0], 0.8 * (1 - exploration) * main_budget)
        # Empirical outcome counts stay consistent with pulls.
        for (group, counts), n in zip(empirical.items(), pulls):
            self.assertEqual(int(counts.sum()), int(n))


if __name__ == "__main__":
    unittest.main()
