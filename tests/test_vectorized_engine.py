"""Agreement tests for the vectorized Bellman engine.

The vectorized stop-loss, adjoint, and influence-target paths must reproduce
the original per-outcome ``np.interp`` reference implementation on both the
LLM question models and a generic tabular model with off-grid rewards.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tis.engine import (  # noqa: E402
    compute_influences,
    solve_adjoint,
    solve_stop_loss,
)
from tis.model import (  # noqa: E402
    FiniteHorizonModel,
    Kernel,
    model_from_counts,
    sample_counts,
)


def _reference_stop_loss(model: FiniteHorizonModel) -> np.ndarray:
    """Original loop implementation of the stop-loss recursion."""
    hmax, states, grid_size = model.horizon, model.state_count, len(model.grid)
    stop_loss = np.zeros((hmax + 1, states, grid_size), dtype=float)
    for state in range(states):
        for weight, _, kernel in model.row_components(1, state):
            for reward, probability in zip(kernel.rewards, kernel.probabilities):
                stop_loss[1, state] += (
                    weight * probability * np.maximum(model.grid - float(reward), 0.0)
                )
    for h in range(2, hmax + 1):
        for state in range(states):
            for weight, _, kernel in model.row_components(h, state):
                for reward, next_state, probability in zip(
                    kernel.rewards, kernel.next_states, kernel.probabilities
                ):
                    shifted = model.grid - float(reward)
                    values = np.interp(
                        shifted,
                        model.grid,
                        stop_loss[h - 1, int(next_state)],
                        left=0.0,
                        right=float(stop_loss[h - 1, int(next_state), -1]),
                    )
                    values[shifted <= 0.0] = 0.0
                    stop_loss[h, state] += weight * float(probability) * values
    return stop_loss


def _reference_adjoint(
    model: FiniteHorizonModel, stop_loss: np.ndarray, quantile_index: int
) -> np.ndarray:
    """Original scatter implementation of the adjoint recursion."""
    adjoint = np.zeros_like(stop_loss)
    adjoint[model.horizon, model.initial_state, quantile_index] = 1.0
    grid = model.grid
    for h in range(model.horizon, 1, -1):
        for state in range(model.state_count):
            weights = adjoint[h, state]
            if not np.any(weights):
                continue
            for row_weight, _, kernel in model.row_components(h, state):
                for reward, next_state, probability in zip(
                    kernel.rewards, kernel.next_states, kernel.probabilities
                ):
                    shifted = grid - float(reward)
                    mass_row = row_weight * float(probability) * weights
                    valid = shifted > 0
                    values = shifted[valid]
                    mass = mass_row[valid]
                    hi = np.searchsorted(grid, values, side="left")
                    hi = np.clip(hi, 1, len(grid) - 1)
                    exact = np.isclose(grid[hi], values, atol=1e-12, rtol=0)
                    destination = adjoint[h - 1, int(next_state)]
                    if np.any(exact):
                        np.add.at(destination, hi[exact], mass[exact])
                    between = ~exact
                    if np.any(between):
                        hidx = hi[between]
                        lidx = hidx - 1
                        vals = values[between]
                        denom = grid[hidx] - grid[lidx]
                        high_weight = (vals - grid[lidx]) / denom
                        np.add.at(destination, lidx, mass[between] * (1.0 - high_weight))
                        np.add.at(destination, hidx, mass[between] * high_weight)
    return adjoint


def _random_tabular_model(seed: int) -> FiniteHorizonModel:
    """Small stochastic-policy model with off-grid and exact-node rewards."""
    rng = np.random.default_rng(seed)
    states, actions, horizon = 4, 2, 3
    grid = np.asarray([0.0, 0.25, 0.61, 1.0, 1.7, 2.3, 3.0])
    policy = np.zeros((horizon + 1, states, actions))
    for h in range(1, horizon + 1):
        raw = rng.random((states, actions)) + 0.1
        policy[h] = raw / raw.sum(axis=1, keepdims=True)
    kernels = {}
    for state in range(states):
        for action in range(actions):
            size = 5
            rewards = rng.choice(
                [0.0, 0.25, 0.3777, 0.61, 1.0], size=size, replace=True
            )
            kernels[(state, action)] = Kernel(
                rewards=rewards,
                next_states=rng.integers(0, states, size=size),
                probabilities=rng.dirichlet(np.ones(size)),
            )
    return FiniteHorizonModel(
        name=f"vector-agreement-{seed}",
        horizon=horizon,
        initial_state=0,
        grid=grid,
        policy=policy,
        kernels=kernels,
    )


def _llm_models() -> list[FiniteHorizonModel]:
    from tis.llm_study.mock import write_mock_kernel_artifact
    from tis.llm_study.models import (
        shared_question_model,
        untied_question_model,
    )
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mock.json"
        write_mock_kernel_artifact(path, questions=1)
        question = json.loads(path.read_text())["questions"][0]
    shared = shared_question_model(question, 4)
    return [shared, untied_question_model(shared)]


class VectorizedEngineAgreementTests(unittest.TestCase):
    def _assert_agreement(self, model: FiniteHorizonModel, alpha: float) -> None:
        solution = solve_stop_loss(model, alpha)
        reference = _reference_stop_loss(model)
        np.testing.assert_allclose(
            solution.stop_loss, reference, rtol=1e-10, atol=1e-12
        )
        adjoint = solve_adjoint(model, solution)
        reference_adjoint = _reference_adjoint(
            model, reference, solution.quantile_index
        )
        np.testing.assert_allclose(
            adjoint, reference_adjoint, rtol=1e-10, atol=1e-12
        )

    def test_llm_shared_and_untied_models_agree(self) -> None:
        for model in _llm_models():
            self._assert_agreement(model, 0.1)

    def test_random_tabular_models_agree(self) -> None:
        for seed in range(3):
            self._assert_agreement(_random_tabular_model(seed), 0.15)

    def test_empirical_llm_model_agreement_and_influences(self) -> None:
        shared = _llm_models()[0]
        rng = np.random.default_rng(7)
        counts = sample_counts(
            shared, {group: 40 for group in shared.query_groups or ()}, rng
        )
        empirical = model_from_counts(shared, counts)
        self._assert_agreement(empirical, 0.1)
        bundle = compute_influences(empirical, 0.1)
        solution = bundle.solution
        adjoint = bundle.adjoint
        for group, tail in zip(bundle.groups, bundle.tail_values):
            kernel = empirical.kernels[group]
            state, action = group
            expected = np.zeros(kernel.size)
            for h in range(1, empirical.horizon + 1):
                weight = float(empirical.policy[h, state, action])
                if weight <= 0:
                    continue
                targets = np.vstack(
                    [
                        np.maximum(empirical.grid - float(reward), 0.0)
                        if h == 1
                        else np.where(
                            empirical.grid - float(reward) <= 0.0,
                            0.0,
                            np.interp(
                                empirical.grid - float(reward),
                                empirical.grid,
                                solution.stop_loss[h - 1, int(next_state)],
                                left=0.0,
                                right=float(
                                    solution.stop_loss[h - 1, int(next_state), -1]
                                ),
                            ),
                        )
                        for reward, next_state in zip(
                            kernel.rewards, kernel.next_states
                        )
                    ]
                )
                centered = targets - np.sum(
                    kernel.probabilities[:, None] * targets, axis=0
                )
                expected += -weight / 0.1 * (centered @ adjoint[h, state])
            np.testing.assert_allclose(tail, expected, rtol=1e-9, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
