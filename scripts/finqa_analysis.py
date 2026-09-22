#!/usr/bin/env python3
"""Held-out analysis for the FinQA terminal-risk study.

Reads the four held-out simulation runs and writes
results/finqa_paper_numbers.json plus build/finqa_tables.tex:

1. Panel-MSE table: per generator x workflow x budget, each method's
   panel MSE normalized to uniform.
2. Paired resolution: per-replication panel-MSE differences (same
   replication indices share seeds) with normal z, for the declared
   comparisons: anchored-vs-uniform (headline, prospective validation of
   the frozen design), anchored-vs-occupancy, anchored-vs-rollout,
   uniform-vs-TIS.
3. Workflow decision: per generator, the ordinary and unitcheck panel
   values (mean of per-question exact CVaRs) and, when raw.csv is
   available (on the machine where the runs executed), the frequency
   with which each method's estimates select the higher-value workflow
   at each budget, with mean regret. For a generator whose workflows are
   equivalent (population gap below EQUIVALENCE_EPS), the reported
   quantity is the frequency of exceeding that epsilon in estimated gap
   (a false-difference rate).

Usage: python scripts/finqa_analysis.py [--results-root results]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = {
    ("qwen", "ordinary"): "finqa_heldout_ordinary",
    ("qwen", "unitcheck"): "finqa_heldout_unitcheck",
    ("phi", "ordinary"): "finqa_heldout_ordinary_phi",
    ("phi", "unitcheck"): "finqa_heldout_unitcheck_phi",
}
METHODS = ("uniform", "tis", "tis_anchored", "learned_occupancy",
           "oracle_tail", "complete_rollout")
COMPARISONS = (
    ("uniform", "tis_anchored"),
    ("tis_anchored", "learned_occupancy"),
    ("tis_anchored", "complete_rollout"),
    ("uniform", "tis"),
)
EQUIVALENCE_EPS = 1e-4


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def paired(directory: Path):
    """(budget, method) -> replication -> panel_mse."""
    values: dict[tuple, dict[int, float]] = defaultdict(dict)
    for row in read_rows(directory / "panel_replications.csv"):
        values[(row["budget_per_group"], row["method"])][int(row["replication"])] = float(
            row["panel_mse"]
        )
    return values


def z_score(a: dict[int, float], b: dict[int, float]):
    reps = sorted(set(a) & set(b))
    d = np.array([a[r] - b[r] for r in reps])
    se = d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(se), float(d.mean() / se) if se > 0 else float("inf"), len(reps)


def workflow_decision(gen: str, results_root: Path, budgets, replications):
    """Stream raw.csv (if present) for per-replication panel-mean estimates."""
    est: dict[str, dict] = {}
    truth: dict[str, float] = {}
    for wf in ("ordinary", "unitcheck"):
        directory = results_root / RUNS[(gen, wf)]
        if not (directory / "raw.csv").exists():
            return None
        diag = json.loads((directory / "diagnostics.json").read_text())
        truth[wf] = float(np.mean([q["truth"] for q in diag]))
        questions = len(diag)
        sums: dict[tuple, float] = defaultdict(float)
        counts: dict[tuple, int] = defaultdict(int)
        with (directory / "raw.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (row["budget_per_group"], row["method"], int(row["replication"]))
                sums[key] += float(row["estimate"])
                counts[key] += 1
        est[wf] = {k: v / questions for k, v in sums.items() if counts[k] == questions}
    gap = truth["ordinary"] - truth["unitcheck"]
    equivalent = abs(gap) < EQUIVALENCE_EPS
    better = "ordinary" if gap > 0 else "unitcheck"
    out = {"population_values": truth, "population_gap": gap,
           "equivalent_at_eps": equivalent, "epsilon": EQUIVALENCE_EPS,
           "aggregation": "panel value = mean over the 50 panel questions of "
           "the per-question exact CVaR (equal weights); a method selects the "
           "workflow with the larger panel-mean estimate, ties to ordinary",
           "rows": [], "paired_selection": []}
    wrong_flags: dict[tuple, dict[int, int]] = {}
    for budget in budgets:
        for method in METHODS:
            wrong = regret = falsediff = n = 0
            flags: dict[int, int] = {}
            for rep in range(replications):
                ka = (str(budget), method, rep)
                if ka not in est["ordinary"] or ka not in est["unitcheck"]:
                    continue
                n += 1
                d = est["ordinary"][ka] - est["unitcheck"][ka]
                if equivalent:
                    falsediff += int(abs(d) > EQUIVALENCE_EPS)
                else:
                    chosen = "ordinary" if d >= 0 else "unitcheck"
                    flags[rep] = int(chosen != better)
                    if chosen != better:
                        wrong += 1
                        regret += abs(gap)
            if n == 0:
                continue
            wrong_flags[(budget, method)] = flags
            row = {"budget": budget, "method": method, "n": n}
            if equivalent:
                row["false_difference_rate"] = falsediff / n
            else:
                row["wrong_rate"] = wrong / n
                row["mean_regret"] = regret / n
            out["rows"].append(row)
        # McNemar discordant-pair tests on wrong-selection indicators
        for m1, m2 in (("tis_anchored", "complete_rollout"),
                       ("tis_anchored", "learned_occupancy"),
                       ("uniform", "tis_anchored")):
            f1 = wrong_flags.get((budget, m1))
            f2 = wrong_flags.get((budget, m2))
            if not f1 or not f2:
                continue
            reps = sorted(set(f1) & set(f2))
            b_ = sum(1 for r in reps if f1[r] == 1 and f2[r] == 0)
            c_ = sum(1 for r in reps if f1[r] == 0 and f2[r] == 1)
            z = (b_ - c_) / np.sqrt(b_ + c_) if (b_ + c_) > 0 else 0.0
            out["paired_selection"].append(
                {"budget": budget, "comparison": f"{m1}-{m2}",
                 "m1_only_wrong": b_, "m2_only_wrong": c_, "z": float(z)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    report: dict = {"panel": {}, "paired": {}, "workflow_decision": {}}
    budgets_seen = set()
    replications = 0
    for (gen, wf), name in RUNS.items():
        directory = args.results_root / name
        summary = read_rows(directory / "panel_summary.csv")
        table = defaultdict(dict)
        for row in summary:
            table[row["budget_per_group"]][row["method"]] = {
                "panel_mse": float(row["panel_mse"]),
                "normalized_to_uniform": float(row["normalized_to_uniform"]),
                "mcse": float(row["mcse"]),
            }
            budgets_seen.add(int(row["budget_per_group"]))
            replications = max(replications, int(row["replications"]))
        report["panel"][f"{gen}|{wf}"] = dict(table)

        values = paired(directory)
        comps = {}
        for m1, m2 in COMPARISONS:
            for budget in sorted(budgets_seen):
                key = (str(budget), m1)
                if key not in values or (str(budget), m2) not in values:
                    continue
                mean, se, z, n = z_score(values[(str(budget), m1)], values[(str(budget), m2)])
                comps[f"{m1}-{m2}|b{budget}"] = {
                    "mean_difference": mean, "se": se, "z": z, "replications": n,
                }
        report["paired"][f"{gen}|{wf}"] = comps

    missing_decisions = []
    for gen in ("qwen", "phi"):
        decision = workflow_decision(gen, args.results_root, sorted(budgets_seen), replications)
        if decision is None:
            missing_decisions.append(gen)
        else:
            report["workflow_decision"][gen] = decision

    out = ROOT / "results" / "finqa_paper_numbers.json"
    if missing_decisions:
        # Never silently replace the authoritative report with one that
        # lacks the decision section (downstream figure code requires it).
        alt = ROOT / "results" / "finqa_paper_numbers_estimation_only.json"
        alt.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
        raise SystemExit(
            f"workflow_decision unavailable for {missing_decisions} (raw "
            f"replication files absent). The authoritative {out.name} was "
            f"left untouched; estimation-only output written to {alt}.")
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(out)

    lines = ["% Generated by scripts/finqa_analysis.py -- do not edit by hand.",
             "% Held-out panel MSE normalized to uniform; alpha=.1, H=3."]
    for (gen, wf) in RUNS:
        for budget in sorted(budgets_seen):
            cell = report["panel"][f"{gen}|{wf}"].get(str(budget))
            if not cell:
                continue
            entries = [f"{m.replace('_', ' ')} {cell[m]['normalized_to_uniform']:.3f}"
                       for m in METHODS if m in cell and m != "uniform"]
            lines.append(f"% {gen} {wf} b{budget}: " + "; ".join(entries))
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    (build / "finqa_tables.tex").write_text("\n".join(lines) + "\n")

    for key, comps in report["paired"].items():
        for comp, stats in sorted(comps.items()):
            if "b200" in comp or "b100" in comp:
                print(f"{key} {comp}: z={stats['z']:+.1f}")


if __name__ == "__main__":
    main()
