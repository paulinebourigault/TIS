"""Representation error of the study grid against the exact return law.

The frozen study grid contains every attainable full-horizon return sum
plus the endpoints 0 and H (``exact_return_grid``). Intermediate layers'
return-to-go supports are not all contained in that grid, so the stop-loss
recursion interpolates between grid atoms at depth two and beyond: the
categorical fixed point that defines both the population target and every
empirical estimate is a projected target, not the exact law. All reported
MSE ratios, allocations, and margins are exact statements about that
common categorical target; this script quantifies how far its CVaR sits
from the CVaR of the exact return law.

For every generator, panel question, horizon, and tail level it computes:

1. the study target: ``solve_stop_loss`` on the frozen grid;
2. the closed-grid target: the same solver on the union of all attainable
   partial-return supports for h = 1..H plus endpoints, over which the
   recursion is interpolation-free;
3. the exact CVaR: an exact forward enumeration of the return law over
   (state, rational partial sum) pairs, with the same fractional-atom
   lower-tail CVaR readout.

Agreement between 2 and 3 validates both computations; the difference
between 1 and 3 is the representation error reported in the paper.

Output: results/llm_grid_representation.json
"""
from __future__ import annotations

import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from math import lcm
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tis.engine import solve_stop_loss  # noqa: E402
from tis.llm_study.common import read_json  # noqa: E402
from tis.llm_study.models import shared_question_model  # noqa: E402
from tis.llm_study.protocol import brier_utility  # noqa: E402

GENERATORS = {
    "qwen4b": "frozen/qwen_kernels.json",
    "phi4mini": "frozen/robustness_kernels.json",
    "granite8b": "frozen/robustness2_kernels.json",
    "mistral24b": "frozen/scale1_kernels.json",
    "qwen32b": "frozen/scale2_kernels.json",
    "glm32b": "frozen/scale3_kernels.json",
}
HORIZONS = (2, 4, 6)
ALPHAS = (0.1, 0.2)


def _reward_fractions(correct_index: int) -> list[Fraction]:
    # Outcome j = answer * 9 + confidence, matching protocol.reward_vector.
    return [brier_utility(j // 9, j % 9, correct_index) for j in range(90)]


def _union_grid(rewards: list[Fraction], horizon: int) -> np.ndarray:
    layer = {Fraction(0)}
    union: set[Fraction] = {Fraction(0), Fraction(horizon)}
    for _ in range(horizon):
        layer = {partial + reward for partial in layer for reward in set(rewards)}
        union.update(layer)
    return np.asarray([float(value) for value in sorted(union)], dtype=float)


def _exact_cvar(support: np.ndarray, probabilities: np.ndarray, alpha: float) -> float:
    order = np.argsort(support)
    support = support[order]
    probabilities = probabilities[order]
    cdf = np.cumsum(probabilities)
    index = int(np.searchsorted(cdf, alpha, side="left"))
    quantile = float(support[index])
    below = float(np.sum((quantile - support[:index]) * probabilities[:index]))
    return quantile - below / alpha


def _exact_law(
    question: dict, horizon: int, rewards: list[Fraction]
) -> tuple[np.ndarray, np.ndarray]:
    denominator = lcm(*[reward.denominator for reward in rewards])
    steps = np.asarray(
        [int(reward * denominator) for reward in rewards], dtype=np.int64
    )
    kernel_probs = np.zeros((91, 90), dtype=float)
    for record in question["kernels"]:
        kernel_probs[int(record["state_index"])] = np.asarray(
            record["probabilities"], dtype=float
        )

    layer_sums = [np.asarray([0], dtype=np.int64)]
    current = {0}
    for _ in range(horizon):
        current = {value + int(step) for value in current for step in set(steps)}
        layer_sums.append(np.asarray(sorted(current), dtype=np.int64))

    # dist[s, i]: probability of being in state s with partial sum
    # layer_sums[t][i] after t steps; states are 0 plus j+1 for outcome j.
    dist = np.zeros((91, 1), dtype=float)
    dist[0, 0] = 1.0
    for t in range(horizon):
        old_sums, new_sums = layer_sums[t], layer_sums[t + 1]
        new = np.zeros((91, len(new_sums)), dtype=float)
        mixed = kernel_probs.T @ dist  # (90 outcomes, old sums)
        for outcome in range(90):
            positions = np.searchsorted(new_sums, old_sums + steps[outcome])
            np.add.at(new[outcome + 1], positions, mixed[outcome])
        dist = new
    marginal = dist.sum(axis=0)
    support = layer_sums[horizon].astype(float) / float(denominator)
    return support, marginal


def _one_generator(item: tuple[str, str]) -> tuple[str, dict]:
    name, artifact_path = item
    artifact = read_json(ROOT / artifact_path)
    rows = []
    for question in artifact["questions"]:
        rewards = _reward_fractions(int(question["correct_index"]))
        for horizon in HORIZONS:
            model = shared_question_model(question, horizon)
            union = dataclasses.replace(model, grid=_union_grid(rewards, horizon))
            support, marginal = _exact_law(question, horizon, rewards)
            mass = marginal.sum()
            for alpha in ALPHAS:
                study = solve_stop_loss(model, alpha)
                closed = solve_stop_loss(union, alpha)
                exact = _exact_cvar(support, marginal / mass, alpha)
                rows.append(
                    {
                        "panel_index": int(question["panel_index"]),
                        "horizon": horizon,
                        "alpha": alpha,
                        "cvar_study_grid": study.cvar,
                        "cvar_closed_grid": closed.cvar,
                        "cvar_exact": exact,
                        "study_minus_exact": study.cvar - exact,
                        "closed_minus_exact": closed.cvar - exact,
                        "probability_mass_check": float(mass),
                    }
                )
    return name, {
        "kernel_artifact": artifact_path,
        "cells": rows,
        "grid_atoms_study": {
            horizon: len(shared_question_model(artifact["questions"][0], horizon).grid)
            for horizon in HORIZONS
        },
        "grid_atoms_closed": {
            horizon: len(
                _union_grid(
                    _reward_fractions(int(artifact["questions"][0]["correct_index"])),
                    horizon,
                )
            )
            for horizon in HORIZONS
        },
    }


def main() -> None:
    report: dict[str, object] = {"generators": {}}
    with ProcessPoolExecutor(max_workers=min(6, len(GENERATORS))) as pool:
        for name, entry in pool.map(_one_generator, GENERATORS.items()):
            report["generators"][name] = entry

    summary: dict[str, object] = {}
    for name, entry in report["generators"].items():
        rows = entry["cells"]
        study_gap = np.asarray([row["study_minus_exact"] for row in rows])
        closed_gap = np.asarray([row["closed_minus_exact"] for row in rows])
        exact = np.asarray([row["cvar_exact"] for row in rows])
        relative = np.abs(study_gap) / np.maximum(np.abs(exact), 1e-12)
        summary[name] = {
            "cells": len(rows),
            "max_abs_study_minus_exact": float(np.max(np.abs(study_gap))),
            "median_abs_study_minus_exact": float(np.median(np.abs(study_gap))),
            "max_relative_study_minus_exact": float(np.max(relative)),
            "median_relative_study_minus_exact": float(np.median(relative)),
            "max_abs_closed_minus_exact": float(np.max(np.abs(closed_gap))),
        }
    report["summary"] = summary

    output = ROOT / "results" / "llm_grid_representation.json"
    output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(output)
    for name, entry in summary.items():
        print(
            f"{name}: |study-exact| max {entry['max_abs_study_minus_exact']:.3e} "
            f"median {entry['median_abs_study_minus_exact']:.3e} | relative max "
            f"{entry['max_relative_study_minus_exact']:.3e} | closed-grid check "
            f"max {entry['max_abs_closed_minus_exact']:.3e}"
        )


if __name__ == "__main__":
    main()
