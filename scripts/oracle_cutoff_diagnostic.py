#!/usr/bin/env python3
"""Why oracle+floor loses at H = 2 on MMLU-Pro: a cutoff diagnostic.

Replays the archived closed-grid MMLU-Pro runs (Brier utility, H = 2,
alpha = .1, 400 queries per kernel, 300 replications) for Qwen3-4B and
Qwen3-32B with the runner's own seeds, so every sampled outcome equals the
archived one. For uniform, oracle+floor, learned occupancy, and anchored TIS
it reports, per question and for the panel:

  full MSE        the archived estimator (empirical root quantile);
  fixed-cutoff    the same samples, with the CVaR read off at the population
                  quantile, C = q* - U_hat(q*)/alpha (a diagnostic only: it
                  uses the population cutoff, which no estimator has);
  first order     sum_g sigma_g^2 / n_g with population scales and the
                  realized counts (Theorem 1's leading term);

and how often the empirical quantile differs from the population one.

Usage: python scripts/oracle_cutoff_diagnostic.py [processes] [budget per kernel, default 400]
Writes results/oracle_cutoff_diagnostic.json (budget 400) or
results/oracle_cutoff_diagnostic_b<budget>.json.
"""
from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tis.allocation import integer_allocation, pilot_size, regularized_weights  # noqa: E402
from tis.engine import compute_influences, solve_stop_loss  # noqa: E402
from tis.llm_study.models import shared_question_model  # noqa: E402
from tis.llm_study.runner import _anchored_score, _mapping, _occupancy_scores, _rng  # noqa: E402
from tis.model import counts_from_streams, coupled_streams, model_from_counts, sample_counts  # noqa: E402

PANELS = {"Qwen3-4B": ("qwen_kernels", "llm_study_closed", "llm_study_anchored"),
          "Qwen3-32B": ("scale2_kernels", "llm_study_closed_scale2", "llm_study_anchored_scale2")}
HORIZON, ALPHA, ALPHA_INDEX, BUDGET, REPS = 2, 0.1, 0, 400, 300
MASTER_SEED, PILOT_MULTIPLIER, MIN_PILOT, MIN_MAIN, EXPONENT = 20260725, 4.0, 8, 2, 0.25
METHODS = ("uniform", "oracle_tail", "learned_occupancy", "tis_anchored")


def question_cell(question: dict, budget: int = BUDGET) -> dict:
    model = shared_question_model(question, HORIZON, closed_grid=True, utility="brier")
    truth = compute_influences(model, ALPHA)
    q_index = truth.solution.quantile_index
    q_value = float(model.grid[q_index])
    groups = model.group_count
    total = groups * budget
    exploration = total ** (-EXPONENT)
    m = pilot_size(total, groups, PILOT_MULTIPLIER, MIN_PILOT, MIN_MAIN)
    main = total - groups * m
    qindex = int(question["panel_index"])
    sigma = np.asarray(truth.tail_scales)
    out = {m_: {"err": [], "fixed_err": [], "first_order": [], "quantile_moved": 0,
                "critical_count": []} for m_ in METHODS}
    critical = int(np.argmax(sigma)) if sigma.sum() > 0 else None
    for rep in range(REPS):
        pilot_counts = sample_counts(model, {g: m for g in model.query_groups},
                                     _rng(MASTER_SEED, HORIZON, ALPHA_INDEX, total, qindex, rep, 0))
        pilot = compute_influences(model_from_counts(model, pilot_counts), ALPHA)
        occ = _occupancy_scores(model, pilot)
        counts = {
            "uniform": integer_allocation(total, np.full(groups, 1.0 / groups), MIN_MAIN),
            "oracle_tail": integer_allocation(total, regularized_weights(sigma, exploration), MIN_MAIN),
            "learned_occupancy": integer_allocation(main, regularized_weights(occ, exploration), MIN_MAIN),
            "tis_anchored": integer_allocation(
                main, regularized_weights(_anchored_score(pilot.tail_scales, occ), exploration), MIN_MAIN),
        }
        maximum = np.max(np.vstack(list(counts.values())), axis=0)
        streams = coupled_streams(model, _mapping(model, maximum),
                                  _rng(MASTER_SEED, HORIZON, ALPHA_INDEX, total, qindex, rep, 1))
        for method, n in counts.items():
            solution = solve_stop_loss(model_from_counts(
                model, counts_from_streams(model, streams, _mapping(model, n))), ALPHA)
            root_u = solution.stop_loss[HORIZON, model.initial_state]
            fixed = q_value - float(root_u[q_index]) / ALPHA
            cell = out[method]
            cell["err"].append(solution.cvar - truth.solution.cvar)
            cell["fixed_err"].append(fixed - truth.solution.cvar)
            cell["first_order"].append(float(np.sum(sigma ** 2 / n)))
            cell["quantile_moved"] += int(solution.quantile_index != q_index)
            if critical is not None:
                cell["critical_count"].append(int(n[critical]))
    summary = {}
    for method, cell in out.items():
        err, fixed = np.array(cell["err"]), np.array(cell["fixed_err"])
        summary[method] = {
            "mse": float(np.mean(err ** 2)), "bias": float(np.mean(err)),
            "fixed_cutoff_mse": float(np.mean(fixed ** 2)),
            "first_order": float(np.mean(cell["first_order"])),
            "quantile_moved_fraction": cell["quantile_moved"] / REPS,
            "critical_kernel_mean_count": float(np.mean(cell["critical_count"])) if cell["critical_count"] else None,
        }
    return {"panel_index": qindex, "item_id": str(question["item_id"]),
            "margin": float(truth.solution.margin), "zero_tail_signal": bool(sigma.sum() == 0),
            "critical_kernel_reach": float(_occupancy_scores(model, truth)[critical]) if critical is not None else None,
            "methods": summary}


def main() -> None:
    processes = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else BUDGET
    report = {"setting": {"horizon": HORIZON, "alpha": ALPHA, "budget_per_kernel": budget,
                          "replications": REPS, "utility": "brier", "grid": "closed"}}
    for name, (kernels, closed_dir, anchored_dir) in PANELS.items():
        questions = json.loads((ROOT / "frozen" / f"{kernels}.json").read_text())["questions"]
        with ProcessPoolExecutor(max_workers=processes) as pool:
            cells = list(pool.map(partial(question_cell, budget=budget), questions))
        panel = {}
        for method in METHODS:
            mse = np.array([c["methods"][method]["mse"] for c in cells])
            panel[method] = {
                "panel_mse": float(mse.mean()),
                "panel_fixed_cutoff_mse": float(np.mean([c["methods"][method]["fixed_cutoff_mse"] for c in cells])),
                "panel_first_order": float(np.mean([c["methods"][method]["first_order"] for c in cells])),
                "panel_squared_bias": float(np.mean([c["methods"][method]["bias"] ** 2 for c in cells])),
                "top_question_shares": sorted(
                    ({"item_id": c["item_id"], "share": float(c["methods"][method]["mse"] / mse.sum())}
                     for c in cells), key=lambda x: -x["share"])[:3],
            }
        archived = {}
        for directory in (closed_dir, anchored_dir):
            for row in csv.DictReader((ROOT / "results" / directory / "panel_summary.csv").open()):
                if (int(row["horizon"]) == HORIZON and float(row["alpha"]) == ALPHA
                        and int(row["budget_per_shared_group"]) == budget and row["method"] in METHODS):
                    archived[row["method"]] = float(row["panel_mse"])
        report[name] = {"panel": panel, "archived_panel_mse": archived, "questions": cells}
        uniform = panel["uniform"]["panel_mse"]
        print(f"== {name}")
        for method in METHODS:
            p = panel[method]
            print(f"  {method:18s} mse/unif {p['panel_mse'] / uniform:.3f} (archived "
                  f"{archived.get(method, float('nan')) / archived.get('uniform', float('nan')):.3f})  fixed-cutoff/unif "
                  f"{p['panel_fixed_cutoff_mse'] / uniform:.3f}  first-order/unif {p['panel_first_order'] / uniform:.3f}"
                  f"  fixed-cutoff reduction {1 - p['panel_fixed_cutoff_mse'] / p['panel_mse']:.1%}")
    name = "oracle_cutoff_diagnostic.json" if budget == BUDGET else f"oracle_cutoff_diagnostic_b{budget}.json"
    (ROOT / "results" / name).write_text(json.dumps(report, indent=1) + "\n")
    print(f"written results/{name}")


if __name__ == "__main__":
    main()
