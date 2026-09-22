"""FinQA study protocol: states, actions, prompts, and severity utility.

State 0 is the root; state 1 + 9*c + k holds candidate c (0-7) at
confidence k (0-8). Each LLM call returns two constrained tokens
(label A-H, confidence 1-9), so every prompt state has an exactly
enumerable 72-outcome law. The workflows differ only in their review
templates: ``ordinary`` is a plain recheck, ``unitcheck`` a unit/sign/
scale audit.

Severity utility: u = 1 - e/(e+s) with e = max(0, |y-y*| - t),
t = TOL_REL*|y*|, s = |y*|; invalid candidates earn 0. Per-step rewards
(u_out - u_src + 1)/2 telescope to (u_final + H)/2, a positive affine
transform of the terminal utility with rewards in [0, 1].
"""
from __future__ import annotations

from typing import Mapping, Sequence

from . import CANDIDATE_LABELS, CANDIDATES, CONFIDENCE_LABELS, CONFIDENCES, WORKFLOWS

TOL_REL = 0.005

SYSTEM_PROMPT = (
    "You are a meticulous financial analyst reviewing candidate solutions "
    "to a numerical question about a financial report. Answer with exactly "
    "two tokens: first the letter of the best candidate (A-H), then a "
    "confidence digit from 1 (very unsure) to 9 (very sure). No other text."
)

SELECT_TEMPLATE = (
    "{context}\n\nQuestion: {question}\n\nCandidate solutions:\n{candidates}\n\n"
    "Select the candidate whose final answer is correct. Reply with the "
    "candidate letter (A-H) and your confidence (1-9)."
)

REVIEW_TEMPLATES = {
    "ordinary": {
        "reconsider": (
            "{context}\n\nQuestion: {question}\n\nCandidate solutions:\n{candidates}\n\n"
            "A previous analyst tentatively selected candidate {prior} with low "
            "confidence {prior_confidence}. Reconsider the problem from scratch "
            "and select the candidate with the correct final answer. Reply with "
            "the candidate letter (A-H) and your confidence (1-9)."
        ),
        "challenge": (
            "{context}\n\nQuestion: {question}\n\nCandidate solutions:\n{candidates}\n\n"
            "A previous analyst selected candidate {prior} with moderate "
            "confidence {prior_confidence}. Recheck the computation in that "
            "candidate and keep it only if its final answer is right. Reply with "
            "the candidate letter (A-H) and your confidence (1-9)."
        ),
        "verify": (
            "{context}\n\nQuestion: {question}\n\nCandidate solutions:\n{candidates}\n\n"
            "A previous analyst selected candidate {prior} with high confidence "
            "{prior_confidence}. Verify the selection briefly. Reply with the "
            "candidate letter (A-H) and your confidence (1-9)."
        ),
    },
    "unitcheck": {
        "reconsider": (
            "{context}\n\nQuestion: {question}\n\nCandidate solutions:\n{candidates}\n\n"
            "A previous analyst tentatively selected candidate {prior} with low "
            "confidence {prior_confidence}. Redo the problem, paying explicit "
            "attention to units (millions versus billions, percentages versus "
            "fractions), signs, and scale. Reply with the candidate letter (A-H) "
            "and your confidence (1-9)."
        ),
        "challenge": (
            "{context}\n\nQuestion: {question}\n\nCandidate solutions:\n{candidates}\n\n"
            "A previous analyst selected candidate {prior} with moderate "
            "confidence {prior_confidence}. Audit that candidate for unit "
            "errors, sign errors, and order-of-magnitude mistakes before keeping "
            "or replacing it. Reply with the candidate letter (A-H) and your "
            "confidence (1-9)."
        ),
        "verify": (
            "{context}\n\nQuestion: {question}\n\nCandidate solutions:\n{candidates}\n\n"
            "A previous analyst selected candidate {prior} with high confidence "
            "{prior_confidence}. Confirm the units, sign, and magnitude of its "
            "final answer. Reply with the candidate letter (A-H) and your "
            "confidence (1-9)."
        ),
    },
    # A non-deferential review: the reviewer recomputes the answer
    # before looking at the prior selection and is instructed to switch
    # whenever the recomputation disagrees, at every confidence band.
    "skeptical": {
        "reconsider": (
            "{context}\n\nQuestion: {question}\n\nCandidate solutions:\n{candidates}\n\n"
            "Compute the answer yourself from the report before reading any "
            "prior work. A previous analyst tentatively picked candidate "
            "{prior} with low confidence {prior_confidence}; treat that pick "
            "as unreliable, and select whichever candidate matches your own "
            "computation, switching without hesitation. Reply with the "
            "candidate letter (A-H) and your confidence (1-9)."
        ),
        "challenge": (
            "{context}\n\nQuestion: {question}\n\nCandidate solutions:\n{candidates}\n\n"
            "Compute the answer yourself from the report before reading any "
            "prior work. A previous analyst picked candidate {prior} with "
            "moderate confidence {prior_confidence}; do not defer to it. "
            "Select whichever candidate matches your own computation, and "
            "switch if that is a different candidate. Reply with the "
            "candidate letter (A-H) and your confidence (1-9)."
        ),
        "verify": (
            "{context}\n\nQuestion: {question}\n\nCandidate solutions:\n{candidates}\n\n"
            "Compute the answer yourself from the report before reading any "
            "prior work. A previous analyst picked candidate {prior} with "
            "high confidence {prior_confidence}; high confidence is not "
            "evidence. Select whichever candidate matches your own "
            "computation, even if that means overturning the selection. "
            "Reply with the candidate letter (A-H) and your confidence (1-9)."
        ),
    },
}


def output_index(candidate_index: int, confidence_index: int) -> int:
    if not (0 <= candidate_index < CANDIDATES and 0 <= confidence_index < CONFIDENCES):
        raise ValueError("Output coordinates outside 8-by-9 alphabet")
    return candidate_index * CONFIDENCES + confidence_index


def output_coordinates(index: int) -> tuple[int, int]:
    if not 0 <= index < CANDIDATES * CONFIDENCES:
        raise ValueError("Output index outside 72-element alphabet")
    return divmod(index, CONFIDENCES)


def action_for_state(state_index: int) -> str:
    if state_index == 0:
        return "select"
    _, confidence_index = output_coordinates(state_index - 1)
    if confidence_index <= 2:
        return "reconsider"
    if confidence_index <= 5:
        return "challenge"
    return "verify"


def candidate_block(bank: Sequence[Mapping[str, object]]) -> str:
    lines = []
    for label, candidate in zip(CANDIDATE_LABELS, bank):
        if candidate.get("valid"):
            suffix = "%" if candidate.get("is_percent") else ""
            lines.append(
                f"{label}. {candidate['derivation']} FINAL ANSWER: {candidate['answer']}{suffix}"
            )
        else:
            lines.append(f"{label}. (no valid final answer was produced)")
    return "\n".join(lines)


def render_messages(
    item: Mapping[str, object],
    bank: Sequence[Mapping[str, object]],
    state_index: int,
    workflow: str,
) -> list[dict[str, str]]:
    if workflow not in WORKFLOWS:
        raise ValueError(f"Unknown workflow {workflow!r}")
    fields = {
        "context": str(item["context"]),
        "question": str(item["question"]),
        "candidates": candidate_block(bank),
    }
    action = action_for_state(state_index)
    if action == "select":
        user = SELECT_TEMPLATE.format(**fields)
    else:
        candidate_index, confidence_index = output_coordinates(state_index - 1)
        user = REVIEW_TEMPLATES[workflow][action].format(
            **fields,
            prior=CANDIDATE_LABELS[candidate_index],
            prior_confidence=f"0.{confidence_index + 1}",
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def numeric_error(answer: float, gold: float) -> float:
    """Convention-invariant absolute error.

    FinQA gold answers mix percent conventions (0.15 versus 15), so the
    error is the minimum over the rescalings {y, y/100, 100y}.
    """
    return min(abs(answer - gold), abs(answer / 100.0 - gold), abs(answer * 100.0 - gold))


def utilities(bank: Sequence[Mapping[str, object]], gold: float) -> list[float]:
    """Severity utility per candidate; invalid candidates earn zero."""
    tolerance = TOL_REL * abs(gold)
    scale = abs(gold)
    values: list[float] = []
    for candidate in bank:
        if not candidate.get("valid"):
            values.append(0.0)
            continue
        error = max(0.0, numeric_error(float(candidate["answer"]), gold) - tolerance)
        values.append(1.0 - error / (error + scale))
    return values
