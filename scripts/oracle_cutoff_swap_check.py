#!/usr/bin/env python3
"""Which kernel moves the oracle's cutoff at H = 2? A one-kernel swap check.

For the two questions that dominate oracle+floor's H = 2 error (Qwen3-4B item
4225 and Qwen3-32B item 9558), replays the oracle+floor replications of
scripts/oracle_cutoff_diagnostic.py with the same seeds and draws. In each
replication it solves the archived estimator and then the same empirical
model with only the starved high-reach kernel replaced by its true law
(a diagnostic: no estimator has the true law). It reports how often the
empirical quantile differs from the population one, and the item's MSE,
before and after the swap.

Usage: python scripts/oracle_cutoff_swap_check.py [budget per kernel, default 400]
Writes results/oracle_cutoff_swap_check.json (budget 400) or results/oracle_cutoff_swap_check_b<budget>.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_cutoff_diagnostic as D  # noqa: E402

from tis.allocation import integer_allocation, pilot_size, regularized_weights  # noqa: E402
from tis.engine import compute_influences, solve_stop_loss  # noqa: E402
from tis.llm_study.models import shared_question_model  # noqa: E402
from tis.llm_study.runner import _anchored_score, _mapping, _occupancy_scores, _rng  # noqa: E402
from tis.model import counts_from_streams, coupled_streams, model_from_counts, sample_counts  # noqa: E402

# (frozen kernels, item, index of the starved high-reach kernel with zero tail influence at the cutoff)
CASES = (("qwen_kernels", "4225", 86), ("scale2_kernels", "9558", 5))


def swap_check(kernels: str, item: str, kernel_index: int, budget: int = D.BUDGET) -> dict:
    questions = json.loads((D.ROOT / "frozen" / f"{kernels}.json").read_text())["questions"]
    question = next(q for q in questions if str(q["item_id"]) == item)
    model = shared_question_model(question, D.HORIZON, closed_grid=True, utility="brier")
    truth = compute_influences(model, D.ALPHA)
    q_index, C = truth.solution.quantile_index, truth.solution.cvar
    groups = model.group_count
    total = groups * budget
    exploration = total ** (-D.EXPONENT)
    m = pilot_size(total, groups, D.PILOT_MULTIPLIER, D.MIN_PILOT, D.MIN_MAIN)
    main = total - groups * m
    sigma = np.asarray(truth.tail_scales)
    qindex = int(question["panel_index"])
    group = model.query_groups[kernel_index]
    moved, moved_swap, err, err_swap = 0, 0, [], []
    for rep in range(D.REPS):
        # Same pilot, allocations and coupled streams as the diagnostic, so the oracle's draws match.
        pilot_counts = sample_counts(model, {g: m for g in model.query_groups},
                                     _rng(D.MASTER_SEED, D.HORIZON, D.ALPHA_INDEX, total, qindex, rep, 0))
        pilot = compute_influences(model_from_counts(model, pilot_counts), D.ALPHA)
        occ = _occupancy_scores(model, pilot)
        counts = {
            "uniform": integer_allocation(total, np.full(groups, 1.0 / groups), D.MIN_MAIN),
            "oracle_tail": integer_allocation(total, regularized_weights(sigma, exploration), D.MIN_MAIN),
            "learned_occupancy": integer_allocation(main, regularized_weights(occ, exploration), D.MIN_MAIN),
            "tis_anchored": integer_allocation(
                main, regularized_weights(_anchored_score(pilot.tail_scales, occ), exploration), D.MIN_MAIN),
        }
        maximum = np.max(np.vstack(list(counts.values())), axis=0)
        streams = coupled_streams(model, _mapping(model, maximum),
                                  _rng(D.MASTER_SEED, D.HORIZON, D.ALPHA_INDEX, total, qindex, rep, 1))
        empirical = model_from_counts(
            model, counts_from_streams(model, streams, _mapping(model, counts["oracle_tail"])))
        solution = solve_stop_loss(empirical, D.ALPHA)
        swapped = empirical.replace_query_probabilities(
            {**{g: empirical.kernels[g].probabilities for g in model.query_groups},
             group: model.kernels[group].probabilities})
        solution_swap = solve_stop_loss(swapped, D.ALPHA)
        moved += int(solution.quantile_index != q_index)
        moved_swap += int(solution_swap.quantile_index != q_index)
        err.append(solution.cvar - C)
        err_swap.append(solution_swap.cvar - C)
    err, err_swap = np.array(err), np.array(err_swap)
    return {"item_id": item, "kernel_index": kernel_index, "budget_per_kernel": budget,
            "oracle_draws": int(counts["oracle_tail"][kernel_index]),
            "reach": float(_occupancy_scores(model, truth)[kernel_index]),
            "tail_scale": float(sigma[kernel_index]), "margin": float(truth.solution.margin),
            "quantile_moved": moved, "quantile_moved_after_swap": moved_swap, "replications": D.REPS,
            "mse": float(np.mean(err ** 2)), "mse_after_swap": float(np.mean(err_swap ** 2))}


def main() -> None:
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else D.BUDGET
    report = [swap_check(*case, budget=budget) for case in CASES]
    for r in report:
        print(f"item {r['item_id']} kernel {r['kernel_index']} (reach {r['reach']:.2f}, tail scale "
              f"{r['tail_scale']:.2g}, {r['oracle_draws']} oracle draws): cutoff moved {r['quantile_moved']} -> "
              f"{r['quantile_moved_after_swap']} of {r['replications']}; MSE {r['mse']:.3e} -> {r['mse_after_swap']:.3e}")
    name = "oracle_cutoff_swap_check.json" if budget == D.BUDGET else f"oracle_cutoff_swap_check_b{budget}.json"
    (D.ROOT / "results" / name).write_text(json.dumps(report, indent=1) + "\n")
    print(f"written results/{name}")


if __name__ == "__main__":
    main()
