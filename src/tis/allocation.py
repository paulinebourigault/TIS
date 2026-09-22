from __future__ import annotations

import numpy as np


def normalized_weights(scales: np.ndarray) -> np.ndarray:
    values = np.asarray(scales, dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.any(values < 0):
        raise ValueError("Scales must be a nonempty nonnegative vector")
    total = float(values.sum())
    if total <= 0:
        return np.full(len(values), 1.0 / len(values))
    return values / total


def regularized_weights(scales: np.ndarray, exploration: float) -> np.ndarray:
    if not (0 <= exploration <= 1):
        raise ValueError("Exploration must lie in [0,1]")
    raw = normalized_weights(scales)
    return (1.0 - exploration) * raw + exploration / len(raw)


def largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
    if total < 0:
        raise ValueError("Rounding total must be nonnegative")
    weights = np.asarray(weights, dtype=float)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Weights must be a probability vector")
    expected = total * weights
    counts = np.floor(expected).astype(int)
    remaining = total - int(counts.sum())
    if remaining:
        order = np.argsort(-(expected - counts), kind="stable")
        counts[order[:remaining]] += 1
    return counts


def integer_allocation(
    total: int,
    weights: np.ndarray,
    minimum_per_group: int = 2,
) -> np.ndarray:
    groups = len(weights)
    residual = total - groups * minimum_per_group
    if residual < 0:
        raise ValueError("Budget cannot satisfy the deterministic group minimum")
    return minimum_per_group + largest_remainder(residual, weights)


def design_constant(scales: np.ndarray, fractions: np.ndarray) -> float:
    scales = np.asarray(scales, dtype=float)
    fractions = np.asarray(fractions, dtype=float)
    if np.any(fractions <= 0):
        raise ValueError("Design fractions must be positive")
    return float(np.sum(scales**2 / fractions))


def pilot_size(
    total_budget: int,
    group_count: int,
    multiplier: float,
    minimum: int,
    minimum_main_per_group: int,
) -> int:
    proposed = max(
        minimum,
        int(np.ceil(multiplier * total_budget ** (2.0 / 3.0) / group_count)),
    )
    maximum = total_budget // group_count - minimum_main_per_group
    if maximum < minimum:
        raise ValueError("Budget is too small for the requested pilot and main floors")
    return min(proposed, maximum)

