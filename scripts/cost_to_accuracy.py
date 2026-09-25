#!/usr/bin/env python3
"""Cost-to-accuracy analysis: RMSE crossing brackets per method.

For CliffWalking, inventory, the controlled replacement task, and the
FinQA Phi ordinary-review cell: absolute RMSE curves on the stated
return scale (never MSE/uniform ratios), one task tolerance
with a sensitivity target, and the observed query crossing bracket per
method - the first tested budget attaining the tolerance and the
neighboring above-target budget. Brackets are observed grid facts, not
confidence intervals for a minimal cost; 'below at smallest budget',
'not reached by largest budget', and 'crossing unresolved' are valid
outcomes. A crossing is labeled resolved when RMSE +/- 2 SE clears the
tolerance on both sides of the bracket; nonmonotone segments are
disclosed. Tolerances are the RMSE uniform attains at its largest and
second-largest measured budgets.

X-axis unit: total charged conditional queries (pilots included, as in
every committed summary); rollout rows charge every sampled transition.
Population oracles are excluded from practical rankings and reported as
references. Writes results/cost_to_accuracy.json and prints the
decision table.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRACTICAL = ("uniform", "tis", "tail_anchored", "tis_anchored",
             "learned_occupancy", "learned_mean", "complete_rollout")
REFERENCE = ("oracle_tail",)


def tabular_curves(name):
    data = json.loads((ROOT / "results/tabular_reconciled.json").read_text())[name]
    groups = {"cliffwalking": 149, "inventory": 41}[name]
    curves = defaultdict(dict)
    for key, cell in data.items():
        b, method = key.split("|")
        total = float(b) * groups
        curves[method][total] = (cell["mse"] ** 0.5,
                                 0.5 * cell["mse_se"] / max(cell["mse"], 1e-300) ** 0.5)
    return curves


def controlled_curves():
    curves = defaultdict(dict)
    for r in csv.DictReader(open(ROOT / "results/replacement_paper/summary.csv")):
        if r["environment"] != "controlled_untied_v1":
            continue
        mse, se = float(r["mse"]), float(r["mse_se"])
        curves[r["method"]][float(r["total_budget"])] = (mse ** 0.5, 0.5 * se / mse ** 0.5)
    for src in ("replacement_anchored", "replacement_meanlearned"):
        for r in csv.DictReader(open(ROOT / f"results/{src}/summary.csv")):
            if r["environment"] != "controlled_untied_v1":
                continue
            mse, se = float(r["mse"]), float(r["mse_se"])
            curves[r["method"]][float(r["total_budget"])] = (mse ** 0.5, 0.5 * se / mse ** 0.5)
    for r in csv.DictReader(open(ROOT / "results/tabular_rollout/summary.csv")):
        if r["environment"] != "controlled_untied_v1":
            continue
        mse, se = float(r["mse"]), float(r["mse_se"])
        curves[r["method"]][float(r["total_budget"])] = (mse ** 0.5, 0.5 * se / mse ** 0.5)
    return curves


def finqa_curves(run="finqa_heldout_ordinary_phi", include_extension=False):
    """FinQA cell: panel RMSE = sqrt(mean panel MSE) per budget; total =
    50 questions x 73 kernels x budget (pilot charged inside). With
    include_extension=True, budgets from results/<run>_b48 (the 400/800
    extension budgets) are appended when present."""
    sums = defaultdict(list)
    paths = [ROOT / f"results/{run}/panel_replications.csv"]
    if include_extension:
        ext = ROOT / f"results/{run}_b48/panel_replications.csv"
        if ext.exists():
            paths.append(ext)
    for path in paths:
        with open(path) as f:
            for r in csv.DictReader(f):
                sums[(int(r["budget_per_group"]), r["method"])].append(float(r["panel_mse"]))
    curves = defaultdict(dict)
    for (b, method), vals in sums.items():
        arr = np.asarray(vals)
        mse, se = arr.mean(), arr.std(ddof=1) / len(arr) ** 0.5
        curves[method][50.0 * 73.0 * b] = (mse ** 0.5, 0.5 * se / mse ** 0.5)
    return curves


def crossings(curves, tol):
    out = {}
    for method, pts in curves.items():
        budgets = sorted(pts)
        rmse = [pts[b][0] for b in budgets]
        ses = [pts[b][1] for b in budgets]
        nonmono = any(rmse[i + 1] > rmse[i] + 2 * (ses[i] + ses[i + 1])
                      for i in range(len(rmse) - 1))
        below = [i for i, v in enumerate(rmse) if v <= tol]
        if not below:
            out[method] = {"result": "not reached by largest budget",
                           "largest_budget": budgets[-1],
                           "rmse_at_largest": rmse[-1], "se": ses[-1],
                           "nonmonotone": nonmono}
            continue
        first = below[0]
        if first == 0:
            entry = {"result": "below at smallest budget",
                     "first_passing_budget": budgets[0]}
        else:
            entry = {"result": "bracket",
                     "last_failing_budget": budgets[first - 1],
                     "first_passing_budget": budgets[first]}
        resolved = (rmse[first] + 2 * ses[first] <= tol) and (
            first == 0 or rmse[first - 1] - 2 * ses[first - 1] >= tol)
        entry.update({"resolved": bool(resolved), "nonmonotone": nonmono,
                      "rmse_at_pass": rmse[first], "se_at_pass": ses[first]})
        out[method] = entry
    return out


def main() -> None:
    report = {"declared": "tolerances are uniform's achieved RMSE at its largest "
              "(primary) and second-largest (sensitivity) measured budgets; "
              "chosen after the runs existed - retrospective benchmark analysis",
              "x_axis": "total charged conditional queries (pilots included); "
              "rollouts charge every sampled transition", "tasks": {}}
    tasks = {
        "cliffwalking": tabular_curves("cliffwalking"),
        "inventory": tabular_curves("inventory"),
        "controlled_untied": controlled_curves(),
        "finqa_phi_ordinary": finqa_curves(),
    }
    for task, curves in tasks.items():
        uni = curves["uniform"]
        budgets = sorted(uni)
        entry = {"curves": {m: {str(int(b)): {"rmse": v[0], "rmse_se": v[1]}
                                for b, v in pts.items()}
                            for m, pts in curves.items()}}
        for label, bidx in (("primary", -1), ("sensitivity", -2)):
            tol = uni[budgets[bidx]][0]
            practical = {m: pts for m, pts in curves.items() if m in PRACTICAL}
            entry[f"tolerance_{label}"] = {
                "value_rmse": tol,
                "definition_budget_total_queries": budgets[bidx],
                "crossings": crossings(practical, tol),
                "references": crossings({m: curves[m] for m in REFERENCE if m in curves}, tol),
            }
        report["tasks"][task] = entry
        print(f"===== {task}: primary tolerance = uniform RMSE at total {budgets[-1]:g} "
              f"= {uni[budgets[-1]][0]:.4g}")
        for m, c in entry["tolerance_primary"]["crossings"].items():
            if c["result"] == "bracket":
                base = budgets[-1]
                frac = c["first_passing_budget"] / base
                print(f"  {m:>18}: reaches it by total {c['first_passing_budget']:g} "
                      f"({frac:.2f}x uniform's budget; bracket from {c['last_failing_budget']:g}; "
                      f"resolved={c['resolved']}, nonmono={c['nonmonotone']})")
            elif c["result"] == "below at smallest budget":
                frac = c["first_passing_budget"] / budgets[-1]
                print(f"  {m:>18}: already below at smallest tested total "
                      f"{c['first_passing_budget']:g} ({frac:.2f}x uniform's budget)")
            else:
                print(f"  {m:>18}: {c['result']} (rmse {c['rmse_at_largest']:.4g})")
    (ROOT / "results/cost_to_accuracy.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n")
    print("written results/cost_to_accuracy.json")


if __name__ == "__main__":
    main()
