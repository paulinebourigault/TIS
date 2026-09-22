from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np

from ..model import FiniteHorizonModel, Kernel
from .protocol import ACTION_INDEX, action_for_state, exact_return_grid, reward_vector


def _validate_question_artifact(question: Mapping[str, object]) -> None:
    kernels = question.get("kernels")
    if not isinstance(kernels, list) or len(kernels) != 91:
        raise ValueError("Each question artifact must contain 91 prompt kernels")
    states = sorted(int(kernel["state_index"]) for kernel in kernels)
    if states != list(range(91)):
        raise ValueError("Question kernels must cover state indices 0,...,90")
    for kernel in kernels:
        probabilities = np.asarray(kernel["probabilities"], dtype=float)
        if probabilities.shape != (90,) or np.any(probabilities <= 0):
            raise ValueError("Every constrained kernel must have 90 positive masses")
        if not np.isclose(probabilities.sum(), 1.0, atol=1e-10):
            raise ValueError("Constrained kernel probabilities do not sum to one")


def shared_question_model(
    question: Mapping[str, object],
    horizon: int,
    closed_grid: bool = False,
    utility: str = "brier",
    policy: str = "default",
) -> FiniteHorizonModel:
    review_policy = policy
    _validate_question_artifact(question)
    correct_index = int(question["correct_index"])
    rewards = reward_vector(correct_index, utility=utility)
    next_states = np.arange(1, 91, dtype=int)
    policy = np.zeros((horizon + 1, 91, 4), dtype=float)
    for h in range(1, horizon + 1):
        for state in range(91):
            policy[h, state, ACTION_INDEX[action_for_state(state, policy=review_policy)]] = 1.0
    kernels: dict[tuple[int, int], Kernel] = {}
    query_groups: list[tuple[int, int]] = []
    for record in question["kernels"]:  # type: ignore[index]
        state = int(record["state_index"])
        action = ACTION_INDEX[action_for_state(state, policy=review_policy)]
        group = (state, action)
        kernels[group] = Kernel(
            rewards=rewards,
            next_states=next_states,
            probabilities=np.asarray(record["probabilities"], dtype=float),
        )
        query_groups.append(group)
    return FiniteHorizonModel(
        name=f"llm_q{int(question['panel_index']):02d}_h{horizon}_shared",
        horizon=horizon,
        initial_state=0,
        grid=exact_return_grid(horizon, closed=closed_grid, utility=utility),
        policy=policy,
        kernels=kernels,
        query_groups=tuple(query_groups),
        metadata={
            "study": "structured_llm_workflow",
            "panel_index": int(question["panel_index"]),
            "item_id": str(question["item_id"]),
            "subject": str(question["subject"]),
            "correct_index": correct_index,
            "shared_prompt_groups": 91,
            "closed_grid": bool(closed_grid),
            "utility": str(utility),
            "review_policy": str(review_policy),
        },
    )


def untied_question_model(shared: FiniteHorizonModel) -> FiniteHorizonModel:
    if shared.group_mode != "shared_state_action" or shared.state_count != 91:
        raise ValueError("Expected a shared 91-state LLM question model")
    kernels: dict[tuple[int, int], Kernel] = {}
    for h in range(1, shared.horizon + 1):
        for state in range(shared.state_count):
            action = int(np.argmax(shared.policy[h, state]))
            kernels[(h, state)] = shared.kernels[(state, action)]
    query_groups = ((shared.horizon, 0),) + tuple(
        (h, state)
        for h in range(1, shared.horizon)
        for state in range(1, 91)
    )
    policy = np.ones((shared.horizon + 1, shared.state_count, 1), dtype=float)
    policy[0] = 0.0
    return FiniteHorizonModel(
        name=shared.name.replace("_shared", "_untied"),
        horizon=shared.horizon,
        initial_state=shared.initial_state,
        grid=shared.grid,
        policy=policy,
        kernels=kernels,
        query_groups=query_groups,
        group_mode="untied_layer_state",
        metadata={
            **dict(shared.metadata or {}),
            "structural_ablation": "separate empirical kernel per reachable layer-state",
            "untied_query_groups": len(query_groups),
        },
    )

