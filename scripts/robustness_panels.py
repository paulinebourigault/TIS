#!/usr/bin/env python3
"""Regenerate the v1 robustness panels for the supplement.

Panel 1: fifty perturbed fully stochastic variants of the frozen
``controlled_untied_v1`` construction (identical concentration rule, Dirichlet
draws from seeds 100 through 149) with exact uniform-to-oracle variance
ratios at alpha in {.05, .1, .2}. No Monte Carlo enters this panel.

Panel 2: inventory pilot-sensitivity grid at 600 queries per block: for pilot
sizes per block {22, 43, 85, 170} (fractions 3.7%--28.3% of the total budget)
and exploration floors N^{-1/6}, N^{-1/4}, N^{-1/3}, the median and 90th
percentile over 300 pilot draws of the conditional design variance
V(w-hat)/V* evaluated with the exact influence scales.

Writes results/robustness_panels/{perturbed_ratios.csv,pilot_grid.csv}.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tis.allocation import regularized_weights  # noqa: E402
from tis.engine import compute_influences  # noqa: E402
from tis.environments import _kernel, make_environment  # noqa: E402
from tis.model import FiniteHorizonModel, model_from_counts, sample_counts  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "robustness_panels"


def controlled_untied_v1_variant(seed: int) -> FiniteHorizonModel:
    """The frozen v1 construction with a different Dirichlet seed."""
    horizon, states = 5, 5
    policy = np.zeros((horizon + 1, states, 1), dtype=float)
    policy[1:, :, 0] = 1.0
    reward_atoms = np.asarray([0.0, 0.5, 1.0])
    rng = np.random.default_rng(seed)
    kernels = {}
    for h in range(1, horizon + 1):
        for state in range(states):
            reward_concentration = np.asarray(
                [0.80 + 0.10 * h, 1.05 + 0.08 * state, 0.90 + 0.06 * (h + state)]
            )
            transition_concentration = np.asarray(
                [0.75 + 0.04 * h + 0.03 * ((state + successor) % states) for successor in range(states)]
            )
            reward_probability = rng.dirichlet(reward_concentration)
            transition_probability = rng.dirichlet(transition_concentration)
            outcomes = [
                (float(reward), successor, float(reward_probability[i] * transition_probability[successor]))
                for i, reward in enumerate(reward_atoms)
                for successor in range(states)
            ]
            kernels[(h, state)] = _kernel(outcomes)
    query_groups = ((horizon, 0),) + tuple((h, s) for h in range(1, horizon) for s in range(states))
    return FiniteHorizonModel(
        name=f"controlled_untied_v1_seed{seed}",
        horizon=horizon,
        initial_state=0,
        grid=np.arange(0.0, horizon + 0.25, 0.5),
        policy=policy,
        kernels=kernels,
        query_groups=query_groups,
        group_mode="untied_layer_state",
    )


def perturbed_panel() -> None:
    rows = []
    for seed in range(100, 150):
        model = controlled_untied_v1_variant(seed)
        for alpha in (0.05, 0.1, 0.2):
            bundle = compute_influences(model, alpha)
            G = model.group_count
            v_unif = G * float((bundle.tail_scales ** 2).sum())
            v_star = float(bundle.tail_scales.sum()) ** 2
            rows.append({
                "seed": seed, "alpha": alpha, "groups": G,
                "margin": bundle.solution.margin, "truth": bundle.solution.cvar,
                "v_unif": v_unif, "v_star": v_star, "ratio": v_unif / v_star,
            })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "perturbed_ratios.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ratios = [r["ratio"] for r in rows]
    margins_ok = all(r["margin"] > 0 for r in rows)
    print("perturbed panel: %d ratios in [%.3f, %.3f], median %.3f, all margins positive: %s" % (
        len(ratios), min(ratios), max(ratios), sorted(ratios)[len(ratios) // 2], margins_ok))


def pilot_grid() -> None:
    model = make_environment("inventory_untied_v1")
    truth = compute_influences(model, 0.1)
    sigma = truth.tail_scales
    v_star = float(sigma.sum()) ** 2
    groups = tuple(model.query_groups or ())
    G = len(groups)
    total = 600 * G
    exponents = {"1/6": 1.0 / 6.0, "1/4": 0.25, "1/3": 1.0 / 3.0}
    pilot_sizes = (22, 43, 85, 170)
    rows = []
    for m in pilot_sizes:
        ratios = {name: [] for name in exponents}
        for replication in range(300):
            rng = np.random.default_rng(np.random.SeedSequence([20260830, m, replication]))
            counts = sample_counts(model, {g: m for g in groups}, rng)
            pilot = compute_influences(model_from_counts(model, counts), 0.1)
            for name, exponent in exponents.items():
                lam = total ** (-exponent)
                w = regularized_weights(pilot.tail_scales, lam)
                v = float(np.sum(sigma ** 2 / w))
                ratios[name].append(v / v_star)
        for name in exponents:
            values = sorted(ratios[name])
            rows.append({
                "pilot_per_block": m, "pilot_fraction": G * m / total,
                "floor": name, "lambda": total ** (-exponents[name]),
                "median": values[len(values) // 2], "q90": values[int(0.9 * len(values))],
            })
            print("m=%d (%.1f%%) floor N^-%s: median %.3f q90 %.3f" % (
                m, 100 * G * m / total, name, values[len(values) // 2], values[int(0.9 * len(values))]))
    with (OUT / "pilot_grid.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    perturbed_panel()
    pilot_grid()
