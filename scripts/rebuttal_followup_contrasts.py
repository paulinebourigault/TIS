#!/usr/bin/env python3
"""Contrasts quoted in the paper that no study script prints.

- alpha = 0.1: full-access designs (inventory raw-profit study) versus
  fitted-model rollouts computed on the same trajectories by
  scripts/rebuttal_checkpoint_access.py.
- Tail levels .1/.05/.01 at 1,200 queries/block: median case ratios of
  oracle+floor and of the occupancy+mean blend to uniform queries.
- Tail levels .05 and .01: per-case anchor/blend ratio against quantile margin
  and tail-mean distance (Spearman), and RMSE-tolerance crossings of
  fitted-model rollouts.
- H=2 diagnostic: cutoff-move rates of the two dominant questions and the
  oracle's squared-bias share at every budget.

Usage: python scripts/rebuttal_followup_contrasts.py
Writes results/rebuttal_followup_contrasts.json.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUDGETS = [150, 300, 600, 1200, 2400]
CASES = range(18)


def load(path, field, arm):
    errors = defaultdict(dict)
    for row in csv.DictReader((ROOT / path).open()):
        errors[(int(row["case"]), int(row["budget"]), row[arm])][int(row["replication"])] = float(row[field])
    return errors


def sq(errors, key):
    return np.square(np.array([errors[key][r] for r in sorted(errors[key])]))


def unpaired_z(a, b):
    return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    out = {}
    full = load("results/rebuttal_inventory_profit/raw.csv", "error_profit", "method")
    ckpt = load("results/rebuttal_checkpoint_access/raw.csv", "error", "arm")
    alpha1 = {}
    for b in BUDGETS:
        for m in ("tis", "tis_anchored", "learned_mean", "learned_occupancy", "occ_plus_mean", "uniform"):
            a = [sq(full, (c, b, m)) for c in CASES]
            r = [sq(ckpt, (c, b, "rollout_model")) for c in CASES]
            zs = [unpaired_z(x, y) for x, y in zip(a, r)]
            alpha1[f"{m}_vs_rollout_model|{b}"] = {
                "family_ratio": float(np.mean([x.mean() for x in a]) / np.mean([y.mean() for y in r])),
                "resolved_better": int(sum(z <= -2 for z in zs)), "resolved_worse": int(sum(z >= 2 for z in zs))}
    out["alpha_0.1_full_access_vs_fitted_rollouts"] = alpha1
    levels = {"0.1": "results/rebuttal_inventory_profit", "0.05": "results/rebuttal_inventory_profit_alpha05",
              "0.01": "results/rebuttal_inventory_profit_alpha01"}
    medians, cases = {}, {}
    for level, directory in levels.items():
        errors = load(f"{directory}/raw.csv", "error_profit", "method")
        mse = {k: float(np.mean(np.square(list(v.values())))) for k, v in errors.items()}
        medians[level] = {f"{m}/uniform|1200": float(np.median([mse[(c, 1200, m)] / mse[(c, 1200, "uniform")]
                                                                 for c in CASES]))
                          for m in ("oracle_tail", "occ_plus_mean", "tis", "tis_anchored")}
        if level != "0.1":
            gate = {c["index"]: c for c in json.loads((ROOT / directory / "gate.json").read_text())["cases"]}
            ratio = [mse[(c, 1200, "tis_anchored")] / mse[(c, 1200, "occ_plus_mean")] for c in CASES]
            cases[level] = {"spearman_ratio_vs_margin": spearman(ratio, [gate[c]["margin"] for c in CASES]),
                            "spearman_ratio_vs_tail_mean_distance":
                                spearman(ratio, [gate[c]["tail_mean_distance"] for c in CASES]),
                            "case_ratios_1200": ratio}
            fitted = load(f"{directory}/fitted_rollouts.csv", "error_profit", "method")
            crossing = defaultdict(int)
            for c in CASES:
                rmse = [np.sqrt(sq(fitted, (c, b, "rollout_model")).mean()) for b in BUDGETS]
                ok = [b for b, x in zip(BUDGETS, rmse) if x <= 0.5]
                crossing[str(ok[0]) if ok else "not reached"] += 1
            cases[level]["fitted_rollouts_first_budget_rmse_le_0.5"] = dict(crossing)
    out["median_case_ratios_to_uniform"] = medians
    out["rare_tail_cases"] = cases
    h2 = {}
    for b, name in ((400, "oracle_cutoff_diagnostic.json"), (1600, "oracle_cutoff_diagnostic_b1600.json"),
                    (6400, "oracle_cutoff_diagnostic_b6400.json")):
        report = json.loads((ROOT / "results" / name).read_text())
        for panel in ("Qwen3-4B", "Qwen3-32B"):
            p = report[panel]
            questions = {q["item_id"]: q for q in p["questions"]}
            entry = {"panel_mean_cutoff_moved": float(np.mean([q["methods"]["oracle_tail"]["quantile_moved_fraction"]
                                                               for q in p["questions"]])),
                     "oracle_squared_bias_share": p["panel"]["oracle_tail"]["panel_squared_bias"]
                     / p["panel"]["oracle_tail"]["panel_mse"]}
            for item in ("4225", "9558"):
                if item in questions:
                    entry[f"item_{item}_cutoff_moved"] = questions[item]["methods"]["oracle_tail"]["quantile_moved_fraction"]
            h2[f"{panel}|{b}"] = entry
    out["h2"] = h2
    (ROOT / "results/rebuttal_followup_contrasts.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rare_tail_cases"}, indent=1)[:4000])
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "case_ratios_1200"} for k, v in cases.items()}, indent=1))


if __name__ == "__main__":
    main()
