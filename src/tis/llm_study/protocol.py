from __future__ import annotations

from fractions import Fraction
from typing import Mapping, Sequence

import numpy as np


ANSWER_LABELS = tuple("ABCDEFGHIJ")
CONFIDENCE_LABELS = tuple(str(index) for index in range(1, 10))
ACTIONS = ("solve", "reconsider", "challenge", "verify")
ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}


def output_index(answer_index: int, confidence_index: int) -> int:
    if not (0 <= answer_index < 10 and 0 <= confidence_index < 9):
        raise ValueError("Output coordinates outside 10-by-9 alphabet")
    return answer_index * 9 + confidence_index


def output_coordinates(index: int) -> tuple[int, int]:
    if not 0 <= index < 90:
        raise ValueError("Output index outside 90-element alphabet")
    return divmod(index, 9)


def confidence_value(confidence_index: int) -> Fraction:
    return Fraction(confidence_index + 1, 10)


# Review policies map a state's confidence band to one of the four frozen
# actions; prompts and labels are shared, so a policy variant changes only
# which prompt template a state receives. "default" is the frozen primary
# protocol. "cautious" tightens both escalation thresholds by one band:
# answers at confidence .4 are reconsidered rather than challenged, and
# answers at .7 are challenged rather than lightly verified.
POLICIES = ("default", "cautious")


def action_for_state(state_index: int, policy: str = "default") -> str:
    if policy not in POLICIES:
        raise ValueError(f"Unknown review policy {policy!r}")
    if state_index == 0:
        return "solve"
    _, confidence_index = output_coordinates(state_index - 1)
    reconsider_top, challenge_top = (2, 5) if policy == "default" else (3, 6)
    if confidence_index <= reconsider_top:
        return "reconsider"
    if confidence_index <= challenge_top:
        return "challenge"
    return "verify"


def brier_utility(
    answer_index: int, confidence_index: int, correct_index: int
) -> Fraction:
    confidence = confidence_value(confidence_index)
    remainder = (1 - confidence) / 9
    probabilities = [remainder] * 10
    probabilities[answer_index] = confidence
    squared = sum(
        (probability - int(index == correct_index)) ** 2
        for index, probability in enumerate(probabilities)
    )
    return Fraction(1) - squared / 2


def confident_error_utility(
    answer_index: int, confidence_index: int, correct_index: int
) -> Fraction:
    """Asymmetric deployment utility for the sensitivity study.

    Correct answers earn u = (1 + c)/2, wrong answers u = (1 - c)^2 / 2:
    a confident error is the one that gets acted on. Bounded in [0, 1]
    and rational on the confidence lattice; same per-response
    monotonicity as the Brier utility but a very different lower tail.
    """
    confidence = confidence_value(confidence_index)
    if answer_index == correct_index:
        return (1 + confidence) / 2
    return (1 - confidence) ** 2 / 2


UTILITIES = {
    "brier": brier_utility,
    "confident_error": confident_error_utility,
}


def reward_vector(correct_index: int, utility: str = "brier") -> np.ndarray:
    stage = UTILITIES[utility]
    return np.asarray(
        [
            float(stage(answer, confidence, correct_index))
            for answer in range(10)
            for confidence in range(9)
        ],
        dtype=float,
    )


def exact_return_grid(
    horizon: int, closed: bool = False, utility: str = "brier"
) -> np.ndarray:
    """Return grid of attainable stage-utility sums plus the endpoints 0, H.

    ``closed=False``: every attainable full-horizon sum; the recursion
    interpolates at depth two and beyond and the target is the categorical
    fixed point on this grid. ``closed=True``: the union of partial-sum
    supports for every h, over which the recursion is interpolation-free
    and the categorical CVaR equals the exact-law CVaR.
    """
    if horizon <= 0:
        raise ValueError("Horizon must be positive")
    stage = UTILITIES[utility]
    stage_rewards = {
        stage(answer, confidence, correct)
        for answer in range(10)
        for confidence in range(9)
        for correct in (0, 1)
    }
    sums = {Fraction(0)}
    union: set[Fraction] = set()
    for _ in range(horizon):
        sums = {partial + reward for partial in sums for reward in stage_rewards}
        if closed:
            union.update(sums)
    result = union if closed else set(sums)
    result.update((Fraction(0), Fraction(horizon)))
    return np.asarray([float(value) for value in sorted(result)], dtype=float)


def format_item(item: Mapping[str, object]) -> str:
    options = item["options"]
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        raise ValueError("Item options must be a sequence")
    lines = ["Question:", str(item["question"]), "", "Options:"]
    lines.extend(
        f"{label}. {option}" for label, option in zip(ANSWER_LABELS, options)
    )
    return "\n".join(lines)


def render_messages(
    item: Mapping[str, object],
    state_index: int,
    prompts: Mapping[str, str],
    policy: str = "default",
) -> list[dict[str, str]]:
    action = action_for_state(state_index, policy=policy)
    item_text = format_item(item)
    prior = ""
    if state_index:
        answer, confidence = output_coordinates(state_index - 1)
        prior = prompts["prior_template"].format(
            answer=ANSWER_LABELS[answer],
            confidence=CONFIDENCE_LABELS[confidence],
            confidence_decimal=f"0.{confidence + 1}",
        )
    user = prompts[f"{action}_template"].format(
        item=item_text,
        prior=prior,
        answer_labels="".join(ANSWER_LABELS),
        confidence_labels="".join(CONFIDENCE_LABELS),
    )
    return [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": user},
    ]

