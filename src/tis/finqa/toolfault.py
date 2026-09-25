"""FinQA review workflow with a calculator tool that occasionally faults.

Every review prompt carries the output of an automated calculator that
re-ran the prior candidate's computation. The calculator is correct except
with a fixed fault probability p, when it reports the value scaled by 100
(a decimal/percent slip). The model sees the reported value and responds
with the usual two constrained tokens, so each prompt state keeps an exactly
enumerable 72-outcome law, calibrated from the real model.

States: 0 is the root selection; 1 + j (j = 9c + k) is a review of candidate
c at confidence k with a correct calculator report; 1 + 72 + j is the same
review with a faulted report. After each model call the calculator faults
independently with probability p, so a kernel outcome is (model output j,
fault flag f) with probability law[j] * (p if f else 1 - p), moving to state
1 + j + 72 f. Rewards depend only on the selected candidate, as in the base
workflow.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..model import FiniteHorizonModel, Kernel
from . import CANDIDATE_LABELS, CONFIDENCES, OUTCOMES
from .models import ACTION_INDEX, ACTIONS, closed_grid, state_utility
from .protocol import action_for_state, render_messages, utilities

STATES_TF = 1 + 2 * OUTCOMES
FAULT_SCALE = 100.0
REPLY_MARKER = "Reply with the candidate letter"


def base_state(state: int) -> int:
    return state if state <= OUTCOMES else state - OUTCOMES


def is_faulted(state: int) -> bool:
    return state > OUTCOMES


def action_for_tf_state(state: int) -> str:
    return action_for_state(base_state(state))


def calculator_report(candidate: Mapping[str, object], faulted: bool) -> str:
    if not candidate.get("valid"):
        return "no result (the candidate has no valid final answer)"
    value = float(candidate["answer"]) * (FAULT_SCALE if faulted else 1.0)
    suffix = "%" if candidate.get("is_percent") else ""
    return f"{value:.6g}{suffix}"


def render_tf_messages(
    item: Mapping[str, object],
    bank: Sequence[Mapping[str, object]],
    state: int,
    workflow: str,
) -> list[dict[str, str]]:
    messages = render_messages(item, bank, base_state(state), workflow)
    if state == 0:
        return messages
    candidate_index = (base_state(state) - 1) // CONFIDENCES
    report = calculator_report(bank[candidate_index], is_faulted(state))
    sentence = (
        f"An automated calculator re-ran candidate "
        f"{CANDIDATE_LABELS[candidate_index]}'s computation and returned: "
        f"{report}. "
    )
    user = messages[1]["content"]
    if user.count(REPLY_MARKER) != 1:
        raise ValueError("Review template must contain the reply instruction once")
    messages[1] = {"role": "user", "content": user.replace(REPLY_MARKER, sentence + REPLY_MARKER)}
    return messages


def toolfault_question_model(
    question: Mapping[str, object],
    bank: Sequence[Mapping[str, object]],
    horizon: int,
    fault_probability: float,
) -> FiniteHorizonModel:
    if not 0.0 < fault_probability < 1.0:
        raise ValueError("Fault probability must lie strictly between 0 and 1")
    gold = float(question["gold_answer"])
    candidate_utilities = utilities(bank, gold)
    laws = {int(k["state_index"]): np.asarray(k["probabilities"], dtype=float)
            for k in question["kernels"]}
    if sorted(laws) != list(range(STATES_TF)):
        raise ValueError(f"Question artifact must contain all {STATES_TF} state kernels")
    p = float(fault_probability)
    next_states = np.concatenate(
        (np.arange(1, OUTCOMES + 1), np.arange(1 + OUTCOMES, 1 + 2 * OUTCOMES))
    ).astype(int)
    outcome_utilities = np.asarray(
        [candidate_utilities[j // CONFIDENCES] for j in range(OUTCOMES)], dtype=float
    )
    policy = np.zeros((horizon + 1, STATES_TF, len(ACTIONS)), dtype=float)
    for h in range(1, horizon + 1):
        for state in range(STATES_TF):
            policy[h, state, ACTION_INDEX[action_for_tf_state(state)]] = 1.0
    kernels: dict[tuple[int, int], Kernel] = {}
    query_groups: list[tuple[int, int]] = []
    for state in range(STATES_TF):
        action = ACTION_INDEX[action_for_tf_state(state)]
        source_utility = state_utility(base_state(state), candidate_utilities)
        rewards = (outcome_utilities - source_utility + 1.0) / 2.0
        law = laws[state]
        kernels[(state, action)] = Kernel(
            rewards=np.concatenate((rewards, rewards)),
            next_states=next_states,
            probabilities=np.concatenate((law * (1.0 - p), law * p)),
        )
        query_groups.append((state, action))
    return FiniteHorizonModel(
        name=f"finqa_tf_q{int(question['panel_index']):02d}_h{horizon}_p{p:g}",
        horizon=horizon,
        initial_state=0,
        grid=closed_grid(candidate_utilities, horizon),
        policy=policy,
        kernels=kernels,
        query_groups=tuple(query_groups),
        metadata={
            "study": "finqa_tool_fault",
            "panel_index": int(question["panel_index"]),
            "item_id": str(question["item_id"]),
            "gold_answer": gold,
            "fault_probability": p,
            "candidate_utilities": [float(u) for u in candidate_utilities],
            "return_transform": "G = (u_terminal + H)/2",
        },
    )
