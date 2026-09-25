"""CPU-side tests for the generation-audit statistics."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tis.llm_study.audit import _binomial_two_sided_p  # noqa: E402


class BinomialTailTests(unittest.TestCase):
    def test_typical_count_is_not_rejected(self) -> None:
        self.assertGreater(_binomial_two_sided_p(410, 4096, 0.1), 0.5)

    def test_extreme_count_is_rejected(self) -> None:
        self.assertLess(_binomial_two_sided_p(700, 4096, 0.1), 1e-8)
        self.assertLess(_binomial_two_sided_p(0, 4096, 0.05), 1e-8)

    def test_tiny_probability_single_hit_is_not_rejected(self) -> None:
        # One observed count for a 1e-4 cell is unremarkable at n=4096;
        # a normal approximation would wrongly flag it.
        self.assertGreater(_binomial_two_sided_p(1, 4096, 1e-4), 0.05)

    def test_matches_exact_enumeration(self) -> None:
        n, p, k = 30, 0.3, 13
        pmf = [
            math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)
        ]
        upper = sum(pmf[k:])
        expected = min(1.0, 2.0 * upper)
        self.assertAlmostEqual(
            _binomial_two_sided_p(k, n, p), expected, places=12
        )

    def test_invalid_probability_raises(self) -> None:
        with self.assertRaises(ValueError):
            _binomial_two_sided_p(1, 10, 0.0)


if __name__ == "__main__":
    unittest.main()
