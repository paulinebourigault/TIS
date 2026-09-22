#!/usr/bin/env python3
"""Measured token-cost comparison at matched accuracy (FinQA held-out).

Answers the review's cost question with measured quantities only: each
kernel artifact records the real tokenizer input-token count of every
state's prompt, every allocation is replayed deterministically with the
runner's exact seed rule (pilot size, exploration floor, integer
rounding), and complete-rollout trajectories are replayed with their
exact stage-6 seeds. A conditional query at state g costs
input_tokens_g + 2 output tokens; a rollout costs that sum along its
realized trajectory; learned methods are charged their full pilot at
the same per-state token prices; uniform pays no pilot. Frozen candidate
banks are shared setup for every method (uniform included) and are
reported separately, charged to no method.

For each generator x workflow cell the script pairs each budget's mean
measured token cost with the observed panel MSE (committed
panel_replications.csv) and interpolates tokens-to-reach-tolerance on
log-log axes between adjacent budgets (fixed rule; a tolerance below
the smallest achieved MSE is marked unreached). Tolerances are declared
as each cell's uniform panel MSE at budgets 100 and 200: "how many
measured tokens does each method need to match what uniform achieves
there?" Ratios below one are measured token savings at matched accuracy.

Writes results/finqa_token_cost.json.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tis.allocation import integer_allocation, pilot_size, regularized_weights  # noqa: E402
from tis.engine import compute_influences  # noqa: E402
from tis.finqa.models import finqa_question_model  # noqa: E402
from tis.llm_study.runner import _anchored_score, _complete_rollouts, _occupancy_scores, _rng  # noqa: E402
from tis.model import model_from_counts, sample_counts  # noqa: E402

ALPHA = 0.1
H = 3
BUDGETS = (25, 50, 100, 200)
OUTPUT_TOKENS_PER_QUERY = 2
SETTINGS = {
    ("qwen", "ordinary"): ("finqa_kernels_heldout2_ordinary.json", "finqa_heldout_ordinary"),
    ("qwen", "unitcheck"): ("finqa_kernels_heldout2_unitcheck.json", "finqa_heldout_unitcheck"),
    ("phi", "ordinary"): ("finqa_kernels_heldout2_phi_ordinary.json", "finqa_heldout_ordinary_phi"),
    ("phi", "unitcheck"): ("finqa_kernels_heldout2_phi_unitcheck.json", "finqa_heldout_unitcheck_phi"),
}
METHODS = ("uniform", "tis", "learned_occupancy", "tis_anchored", "complete_rollout")


def panel_mse(directory: Path) -> dict[tuple[int, str], float]:
    sums: dict[tuple[int, str], list[float]] = defaultdict(list)
    with (directory / "panel_replications.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            sums[(int(row["budget_per_group"]), row["method"])].append(
                float(row["panel_mse"]))
    return {k: float(np.mean(v)) for k, v in sums.items()}


def main() -> None:
    config = json.loads((ROOT / "configs/finqa_heldout_ordinary.json").read_text())
    replications = int(config["replications"])
    exploration_exp = float(config["exploration_exponent"])
    minimum_main = int(config["minimum_main_per_group"])
    master_seed = int(config["master_seed"])

    report: dict = {"output_tokens_per_query": OUTPUT_TOKENS_PER_QUERY,
                    "cells": {}}
    for (gen, wf), (kfile, rdir) in SETTINGS.items():
        art = json.loads((ROOT / "frozen" / kfile).read_text())
        banks = {int(b["panel_index"]): b["candidates"]
                 for b in json.loads((ROOT / "frozen/finqa_banks_heldout2.json").read_text())["banks"]}
        models, prices = {}, {}
        for q in art["questions"]:
            qi = int(q["panel_index"])
            models[qi] = finqa_question_model(q, banks[qi], H)
            tok = np.zeros(73)
            for k in q["kernels"]:
                tok[int(k["state_index"])] = float(k["input_tokens"])
            # price per query at group (state, action): prompt + 2 generated
            prices[qi] = np.asarray(
                [tok[state] + OUTPUT_TOKENS_PER_QUERY
                 for state, _ in models[qi].query_groups], dtype=float)
        mses = panel_mse(ROOT / "results" / rdir)

        cell = {"budgets": {}, "tolerances": {}}
        mean_tokens: dict[str, dict[int, float]] = {m: {} for m in METHODS}
        for budget in BUDGETS:
            per_method_tok = {m: 0.0 for m in METHODS}
            for qi, model in sorted(models.items()):
                groups = model.group_count
                total = groups * budget
                exploration = total ** (-exploration_exp)
                m_pilot = pilot_size(total, groups, float(config["pilot_multiplier"]),
                                     int(config["minimum_pilot_per_group"]), minimum_main)
                main_total = total - groups * m_pilot
                price = prices[qi]
                pilot_price = float(m_pilot * price.sum())
                uni_counts = integer_allocation(
                    total, np.full(groups, 1.0 / groups), minimum_main)
                per_method_tok["uniform"] += float((uni_counts * price).sum()) * replications
                for rep in range(replications):
                    pilot_counts = sample_counts(
                        model, {g: m_pilot for g in model.query_groups or ()},
                        _rng(master_seed, H, 0, total, qi, rep, 0))
                    pilot = compute_influences(model_from_counts(model, pilot_counts), ALPHA)
                    occ_pilot = _occupancy_scores(model, pilot)
                    scores = {
                        "tis": np.asarray(pilot.tail_scales, dtype=float),
                        "learned_occupancy": occ_pilot,
                        "tis_anchored": _anchored_score(
                            np.asarray(pilot.tail_scales, dtype=float), occ_pilot),
                    }
                    for method, score in scores.items():
                        weights = regularized_weights(score, exploration)
                        counts = integer_allocation(main_total, weights, minimum_main)
                        per_method_tok[method] += float((counts * price).sum()) + pilot_price
                    runs = total // H
                    _, visits = _complete_rollouts(
                        model, H, runs, _rng(master_seed, H, 0, total, qi, rep, 6))
                    # visits are per state over the trajectory steps; groups are
                    # (state, action) in state order, so visit counts map 1:1
                    per_method_tok["complete_rollout"] += float(
                        (np.asarray(visits, dtype=float) * price).sum())
            for method in METHODS:
                mean_tokens[method][budget] = per_method_tok[method] / replications
            cell["budgets"][budget] = {
                method: {"mean_tokens": mean_tokens[method][budget],
                         "panel_mse": mses.get((budget, method))}
                for method in METHODS}

        # tokens to reach declared tolerances via log-log interpolation
        for tol_budget in (100, 200):
            tol = mses.get((tol_budget, "uniform"))
            if tol is None:
                continue
            entry = {}
            for method in METHODS:
                xs = np.array([mean_tokens[method][b] for b in BUDGETS])
                ys = np.array([mses.get((b, method), np.nan) for b in BUDGETS])
                ok = ~np.isnan(ys) & (ys > 0)
                xs, ys = xs[ok], ys[ok]
                order = np.argsort(xs)
                xs, ys = xs[order], ys[order]
                tokens_needed = None
                if (ys <= tol).any():
                    idx = int(np.argmax(ys <= tol))
                    if idx == 0:
                        tokens_needed = float(xs[0])  # already below at smallest budget
                    else:
                        lx0, lx1 = np.log(xs[idx - 1]), np.log(xs[idx])
                        ly0, ly1 = np.log(ys[idx - 1]), np.log(ys[idx])
                        f = (np.log(tol) - ly0) / (ly1 - ly0)
                        tokens_needed = float(np.exp(lx0 + f * (lx1 - lx0)))
                entry[method] = {"tokens_to_tolerance": tokens_needed,
                                 "reached": tokens_needed is not None}
            base = entry["uniform"]["tokens_to_tolerance"]
            for method in METHODS:
                t = entry[method]["tokens_to_tolerance"]
                entry[method]["ratio_to_uniform"] = (t / base) if (t and base) else None
            cell["tolerances"][f"uniform_mse_at_b{tol_budget}"] = {
                "tolerance": tol, "methods": entry}
        report["cells"][f"{gen}|{wf}"] = cell
        print(f"== {gen}|{wf} done")
        for name, tolentry in cell["tolerances"].items():
            row = {m: (round(v["ratio_to_uniform"], 3) if v["ratio_to_uniform"] else "unreached")
                   for m, v in tolentry["methods"].items()}
            print("  ", name, row)

    (ROOT / "results/finqa_token_cost.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(ROOT / "results/finqa_token_cost.json")


if __name__ == "__main__":
    main()
