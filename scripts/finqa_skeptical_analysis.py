#!/usr/bin/env python3
"""Skeptical-versus-ordinary decision analysis on the Phi held-out panel.

Executes the held-out plan declared in configs/finqa_skeptical_gate.json:
the committed ordinary-Phi run is the comparison arm, the skeptical-Phi
run shares its master seed, so replications pair exactly. For every
method and budget the script computes the per-replication panel-mean
estimates of both workflows from raw.csv, the wrong-selection rate for
the deployment choice (deploy the workflow with the larger estimated
panel value; the population-better workflow comes from the exact
diagnostics), mean regret on the return scale, and discordant-pair z
against uniform. Estimation summaries for the skeptical run come from
its committed panel summary.

Writes results/finqa_skeptical_decision.json.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = {"ordinary": "finqa_heldout_ordinary_phi",
        "skeptical": "finqa_heldout_skeptical_phi"}
METHODS = ("uniform", "tis", "learned_occupancy", "tis_anchored",
           "oracle_tail", "complete_rollout")
BUDGETS = (25, 50, 100, 200)


def panel_estimates(directory: Path) -> tuple[dict, float, int]:
    diag = json.loads((directory / "diagnostics.json").read_text())
    value = float(np.mean([q["truth"] for q in diag]))
    questions = len(diag)
    sums: dict[tuple, float] = defaultdict(float)
    counts: dict[tuple, int] = defaultdict(int)
    with (directory / "raw.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["budget_per_group"]), row["method"], int(row["replication"]))
            sums[key] += float(row["estimate"])
            counts[key] += 1
    est = {k: v / questions for k, v in sums.items() if counts[k] == questions}
    return est, value, questions


def main() -> None:
    est, value, questions = {}, {}, {}
    for wf, rdir in RUNS.items():
        est[wf], value[wf], questions[wf] = panel_estimates(ROOT / "results" / rdir)
    assert questions["ordinary"] == questions["skeptical"]
    gap = value["ordinary"] - value["skeptical"]
    better = "ordinary" if gap > 0 else "skeptical"
    out = {"population_values": value, "signed_gap_ordinary_minus_skeptical": gap,
           "better_workflow": better, "questions": questions["ordinary"],
           "selection_rule": "deploy the workflow with the larger panel-mean "
           "estimate (equal-weight mean of per-question CVaR estimates), "
           "ties to ordinary", "rows": [], "paired_vs_uniform": []}

    wrong_flags: dict[tuple, dict[int, int]] = {}
    for budget in BUDGETS:
        for method in METHODS:
            flags: dict[int, int] = {}
            wrong = n = 0
            for rep in range(1000):
                ka = (budget, method, rep)
                if ka not in est["ordinary"] or ka not in est["skeptical"]:
                    continue
                n += 1
                d = est["ordinary"][ka] - est["skeptical"][ka]
                chosen = "ordinary" if d >= 0 else "skeptical"
                bad = int(chosen != better)
                flags[rep] = bad
                wrong += bad
            if n == 0:
                continue
            wrong_flags[(budget, method)] = flags
            out["rows"].append({"budget": budget, "method": method, "n": n,
                                "wrong_rate": wrong / n,
                                "mean_regret": wrong / n * abs(gap)})
        f_u = wrong_flags.get((budget, "uniform"))
        for method in ("tis_anchored", "learned_occupancy", "complete_rollout", "tis"):
            f_m = wrong_flags.get((budget, method))
            if not f_u or not f_m:
                continue
            reps = sorted(set(f_u) & set(f_m))
            b_ = sum(1 for r in reps if f_u[r] == 1 and f_m[r] == 0)
            c_ = sum(1 for r in reps if f_u[r] == 0 and f_m[r] == 1)
            z = (b_ - c_) / np.sqrt(b_ + c_) if (b_ + c_) > 0 else 0.0
            out["paired_vs_uniform"].append(
                {"budget": budget, "method": method,
                 "uniform_only_wrong": b_, "method_only_wrong": c_, "z": float(z)})

    (ROOT / "results/finqa_skeptical_decision.json").write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n")
    print("gap (ord - skep):", round(gap, 6), "better:", better)
    for r in out["rows"]:
        if r["budget"] in (25, 50):
            print(f"  b{r['budget']:>3} {r['method']:>18} wrong {100*r['wrong_rate']:.1f}%")
    for r in out["paired_vs_uniform"]:
        print(f"  b{r['budget']:>3} z(unif vs {r['method']}): {r['z']:+.1f}")


if __name__ == "__main__":
    main()
