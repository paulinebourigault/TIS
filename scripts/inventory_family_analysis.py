#!/usr/bin/env python3
"""Inventory-family analysis: ratios, paired contrasts, tolerance crossings.

From the per-replication errors of every family case: MSE ratios to
uniform per case and budget; paired squared-error contrasts (coupled
streams; rollouts on their independent stream) of plain and anchored TIS
against occupancy, mean influence, both mixture controls, and rollouts;
and, per the declared tolerances (absolute RMSE 0.02/0.01/0.005 on the
return scale), the first tested budget per method reaching each
tolerance, with a +-2 SE resolution flag, no extrapolation. Every case
is reported. Writes results/inventory_family/analysis.json.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "results" / "inventory_family"
DECL = json.loads((ROOT / "configs/inventory_family.json").read_text())
TOLS = [0.02, 0.01, 0.005]
CONTRAST_REFS = ("learned_occupancy", "learned_mean", "occ_plus_uniform",
                 "occ_plus_mean", "complete_rollout")
GROUPS_BY_POLICY = {}


def load_case(i):
    err = defaultdict(dict)
    for r in csv.DictReader((DIR / f"raw_case{i:02d}.csv").open()):
        err[(int(r["budget_per_block"]), r["method"])][int(r["replication"])] = \
            float(r["error"])
    return err


def main() -> None:
    cases = json.loads((DIR / "verification.json").read_text())["cases"]
    budgets = [int(b) for b in DECL["budgets_per_block"]]
    out = {"cases": []}
    wins = defaultdict(lambda: [0, 0, 0])  # method_vs_ref -> [resolved better, resolved worse, unresolved]
    for case in cases:
        i = case["index"]
        err = load_case(i)
        entry = {**{k: case[k] for k in ("index", "p_base", "loss", "policy", "margin",
                                          "uniform_to_oracle")}, "budgets": {}}
        for b in budgets:
            uni = np.array([err[(b, "uniform")][k] for k in sorted(err[(b, "uniform")])])
            u_mse = float((uni ** 2).mean())
            row = {"uniform_mse": u_mse, "ratio": {}, "paired": {}}
            for m in ("tis", "tis_anchored", "learned_occupancy", "learned_mean",
                      "occ_plus_uniform", "occ_plus_mean", "complete_rollout",
                      "oracle_tail"):
                x = np.array([err[(b, m)][k] for k in sorted(err[(b, m)])])
                row["ratio"][m] = float((x ** 2).mean() / u_mse)
            for meth in ("tis", "tis_anchored"):
                for ref in CONTRAST_REFS:
                    a = err[(b, meth)]; r = err[(b, ref)]
                    common = sorted(set(a) & set(r))
                    d = np.array([a[k] ** 2 - r[k] ** 2 for k in common])
                    se = d.std(ddof=1) / np.sqrt(len(d))
                    z = float(d.mean() / se) if se > 0 else 0.0
                    row["paired"][f"{meth}_vs_{ref}"] = {
                        "ratio": row["ratio"][meth] / row["ratio"][ref],
                        "z": z}
                    if b == budgets[-1]:
                        key = f"{meth}_vs_{ref}"
                        if z <= -2: wins[key][0] += 1
                        elif z >= 2: wins[key][1] += 1
                        else: wins[key][2] += 1
            entry["budgets"][str(b)] = row
        # declared-tolerance crossings on RMSE
        entry["tolerance_crossings"] = {}
        for m in ("uniform", "tis", "tis_anchored", "learned_occupancy",
                  "learned_mean", "occ_plus_uniform", "occ_plus_mean",
                  "complete_rollout"):
            rmse = {}
            for b in budgets:
                x = np.array([err[(b, m)][k] for k in sorted(err[(b, m)])])
                sq = x ** 2
                mse = sq.mean()
                mse_se = sq.std(ddof=1) / np.sqrt(len(sq))
                rmse[b] = (float(np.sqrt(mse)), float(0.5 * mse_se / np.sqrt(mse)))
            tol_entry = {}
            for tol in TOLS:
                first = next((b for b in budgets if rmse[b][0] <= tol), None)
                if first is None:
                    tol_entry[str(tol)] = {"first_budget": None}
                else:
                    resolved = rmse[first][0] + 2 * rmse[first][1] <= tol
                    tol_entry[str(tol)] = {"first_budget": first,
                                           "resolved": bool(resolved)}
            entry["tolerance_crossings"][m] = tol_entry
        out["cases"].append(entry)
    out["largest_budget_scoreboard"] = {
        k: {"resolved_better": v[0], "resolved_worse": v[1], "unresolved": v[2]}
        for k, v in sorted(wins.items())}
    (DIR / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print("=== scoreboard at largest budget (18 cases; z threshold 2) ===")
    for k, v in sorted(wins.items()):
        print(f"{k:>40}: better {v[0]:2}  worse {v[1]:2}  unresolved {v[2]:2}")
    print("written", DIR / "analysis.json")


if __name__ == "__main__":
    main()
