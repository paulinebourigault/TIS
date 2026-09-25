"""Finite-horizon model for the FinQA review workflow.

States: 0 (root) and 1 + 9c + k; outcome j = 9c + k moves every state to
1 + j. Rewards (u_j - u_s + 1)/2 telescope to G = (u_final + H)/2, so
the h-step return-to-go from utility u_i has support
{(u_j - u_i + h)/2 : j}. The closed grid is the union of these supports
plus endpoints {0, H} (at most H(C+1)C + 2 atoms); the recursion is
exact and the categorical target equals the exact-law CVaR.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..model import FiniteHorizonModel, Kernel
from . import CANDIDATES, CONFIDENCES, OUTCOMES, STATES
from .protocol import action_for_state, utilities

ACTIONS = ("select", "reconsider", "challenge", "verify")
ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}


def state_utility(state: int, candidate_utilities: Sequence[float]) -> float:
    if state == 0:
        return 0.0
    candidate_index = (state - 1) // CONFIDENCES
    return float(candidate_utilities[candidate_index])


def closed_grid(candidate_utilities: Sequence[float], horizon: int) -> np.ndarray:
    sources = [0.0] + [float(u) for u in candidate_utilities]
    finals = [float(u) for u in candidate_utilities]
    atoms: set[float] = {0.0, float(horizon)}
    for h in range(1, horizon + 1):
        for ui in sources:
            for uj in finals:
                atoms.add((uj - ui + h) / 2.0)
    grid = np.asarray(sorted(atoms), dtype=float)
    keep = [0]
    for i in range(1, len(grid)):
        if grid[i] - grid[keep[-1]] > 1e-12:
            keep.append(i)
    return grid[keep]


def finqa_question_model(
    question: Mapping[str, object],
    bank: Sequence[Mapping[str, object]],
    horizon: int,
) -> FiniteHorizonModel:
    gold = float(question["gold_answer"])
    candidate_utilities = utilities(bank, gold)
    laws = {int(k["state_index"]): np.asarray(k["probabilities"], dtype=float)
            for k in question["kernels"]}
    if sorted(laws) != list(range(STATES)):
        raise ValueError("Question artifact must contain all 73 state kernels")
    next_states = np.arange(1, OUTCOMES + 1, dtype=int)
    outcome_utilities = np.asarray(
        [candidate_utilities[j // CONFIDENCES] for j in range(OUTCOMES)], dtype=float
    )
    policy = np.zeros((horizon + 1, STATES, len(ACTIONS)), dtype=float)
    for h in range(1, horizon + 1):
        for state in range(STATES):
            policy[h, state, ACTION_INDEX[action_for_state(state)]] = 1.0
    kernels: dict[tuple[int, int], Kernel] = {}
    query_groups: list[tuple[int, int]] = []
    for state in range(STATES):
        action = ACTION_INDEX[action_for_state(state)]
        source_utility = state_utility(state, candidate_utilities)
        rewards = (outcome_utilities - source_utility + 1.0) / 2.0
        kernels[(state, action)] = Kernel(
            rewards=rewards,
            next_states=next_states,
            probabilities=laws[state],
        )
        query_groups.append((state, action))
    return FiniteHorizonModel(
        name=f"finqa_q{int(question['panel_index']):02d}_h{horizon}",
        horizon=horizon,
        initial_state=0,
        grid=closed_grid(candidate_utilities, horizon),
        policy=policy,
        kernels=kernels,
        query_groups=tuple(query_groups),
        metadata={
            "study": "finqa_terminal_risk",
            "panel_index": int(question["panel_index"]),
            "item_id": str(question["item_id"]),
            "gold_answer": gold,
            "candidate_utilities": [float(u) for u in candidate_utilities],
            "return_transform": "G = (u_terminal + H)/2",
        },
    )


def terminal_cvar_from_return(return_cvar: float, horizon: int) -> float:
    """Invert the affine transform: u-scale CVaR from return-scale CVaR."""
    return 2.0 * return_cvar - horizon
