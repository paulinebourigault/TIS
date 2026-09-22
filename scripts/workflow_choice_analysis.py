#!/usr/bin/env python3
"""Fixed-budget workflow-choice experiment from per-replication outputs.

Treats the six generators as candidate workflows sharing the panel, horizon,
and utility. The per-workflow value is C_j = mean over the 50 panel
questions of the per-question categorical CVaR (a mean of conditional
CVaRs, not the CVaR of a pooled distribution). For every replication, each
allocation method selects the workflow with the highest estimated value at
a matched total budget; we report, over all fifteen unordered generator
pairs and the full six-way choice, the frequency of selecting a workflow
below the population best and the mean regret max_j C_j - C_{selected}.

Population values come from each run's diagnostics.json; per-replication
estimates are streamed from raw.csv (available where the runs executed;
raw files are not committed to GitHub). Ties in the population values are
handled by treating any argmax as correct. All pairs are reported.

Reads the closed-grid directories by default (``--original`` switches to
the original-grid runs); the complete-rollout runs are merged in when
present. Writes results/llm_workflow_choice.json and a table body to
build/llm_workflow_choice_table.tex.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL = "--original" in sys.argv[1:]
MAIN = "llm_study_paper" if ORIGINAL else "llm_study_closed"
ROLLOUT = None if ORIGINAL else "llm_study_rollout"
ANCHORED = None if ORIGINAL else "llm_study_anchored"
GENERATORS = [
    ("Qwen3-4B", ""),
    ("Phi-4-mini", "_robustness"),
    ("Granite-4.2-8B", "_robustness2"),
    ("Mistral-24B", "_scale1"),
    ("Qwen3-32B", "_scale2"),
    ("GLM-4-32B", "_scale3"),
]
METHODS = ("uniform", "tis", "oracle_tail", "complete_rollout", "tis_anchored", "learned_occupancy")
QUESTIONS = 50


def stream_panel_means(directory: Path, methods) -> dict:
    """(horizon, alpha, budget, method, replication) -> panel-mean estimate."""
    sums: dict[tuple, float] = defaultdict(float)
    counts: dict[tuple, int] = defaultdict(int)
    with (directory / "raw.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            method = row["method"]
            if method not in methods:
                continue
            key = (
                row["horizon"],
                row["alpha"],
                row["budget_per_shared_group"],
                method,
                int(row["replication"]),
            )
            sums[key] += float(row["estimate"])
            counts[key] += 1
    result = {}
    for key, value in sums.items():
        if counts[key] != QUESTIONS:
            raise ValueError(f"{directory}: cell {key} has {counts[key]} questions")
        result[key] = value / QUESTIONS
    return result


def population_values(directory: Path) -> dict:
    """(horizon, alpha) -> C_j = mean of per-question population CVaRs."""
    diagnostics = json.loads((directory / "diagnostics.json").read_text())
    sums: dict[tuple, list[float]] = defaultdict(list)
    for record in diagnostics:
        sums[(str(record["horizon"]), str(record["alpha"]))].append(float(record["truth"]))
    return {key: sum(values) / len(values) for key, values in sums.items()}


def main() -> None:
    estimates: dict[str, dict] = {}
    truth: dict[str, dict] = {}
    for label, suffix in GENERATORS:
        base = ROOT / "results" / f"{MAIN}{suffix}"
        if not (base / "raw.csv").exists():
            print(f"missing raw.csv under {base}; run where the outputs live")
            return
        cells = stream_panel_means(base, {"uniform", "tis", "oracle_tail"})
        if ROLLOUT is not None:
            rollout_dir = ROOT / "results" / f"{ROLLOUT}{suffix}"
            if (rollout_dir / "raw.csv").exists():
                cells.update(stream_panel_means(rollout_dir, {"complete_rollout"}))
        if ANCHORED is not None:
            anchored_dir = ROOT / "results" / f"{ANCHORED}{suffix}"
            if (anchored_dir / "raw.csv").exists():
                cells.update(
                    stream_panel_means(anchored_dir, {"tis_anchored", "learned_occupancy"})
                )
        estimates[label] = cells
        truth[label] = population_values(base)

    labels = [label for label, _ in GENERATORS]
    settings = sorted(
        {(k[0], k[1], k[2]) for cells in estimates.values() for k in cells}
    )
    replications = sorted({k[4] for cells in estimates.values() for k in cells})
    report: dict[str, object] = {"pairs": {}, "six_way": {}}
    for horizon, alpha, budget in settings:
        for method in METHODS:
            available = [
                label
                for label in labels
                if (horizon, alpha, budget, method, replications[0]) in estimates[label]
            ]
            if len(available) < 2:
                continue
            values = {label: truth[label][(horizon, alpha)] for label in available}
            pair_wrong = pair_regret = pair_total = 0.0
            for j, k in combinations(available, 2):
                best = max(values[j], values[k])
                for rep in replications:
                    ej = estimates[j][(horizon, alpha, budget, method, rep)]
                    ek = estimates[k][(horizon, alpha, budget, method, rep)]
                    chosen = j if ej >= ek else k
                    pair_total += 1
                    if values[chosen] < best - 1e-12:
                        pair_wrong += 1
                        pair_regret += best - values[chosen]
            six_wrong = six_regret = 0.0
            best_all = max(values[label] for label in available)
            for rep in replications:
                chosen = max(
                    available,
                    key=lambda label: estimates[label][(horizon, alpha, budget, method, rep)],
                )
                if values[chosen] < best_all - 1e-12:
                    six_wrong += 1
                    six_regret += best_all - values[chosen]
            key = f"{horizon}|{alpha}|{budget}|{method}"
            report["pairs"][key] = {
                "wrong_rate": pair_wrong / pair_total,
                "mean_regret": pair_regret / pair_total,
                "comparisons": int(pair_total),
                "generators": available,
            }
            report["six_way"][key] = {
                "wrong_rate": six_wrong / len(replications),
                "mean_regret": six_regret / len(replications),
            }

    out = ROOT / "results" / (
        "llm_workflow_choice_original.json" if ORIGINAL else "llm_workflow_choice.json"
    )
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    lines = [
        "% Generated by scripts/workflow_choice_analysis.py -- do not edit by hand.",
        "% Rows: budget x method at alpha=.1; wrong-selection % and mean regret,",
        "% pairwise over all generator pairs / six-way.",
    ]
    for horizon in ("2", "4", "6"):
        for budget in ("50", "100", "200", "400"):
            for method in METHODS:
                key = f"{horizon}|0.1|{budget}|{method}"
                if key not in report["pairs"]:
                    continue
                pair = report["pairs"][key]
                six = report["six_way"][key]
                lines.append(
                    f"$H={horizon}$ & {budget} & {method.replace('_', ' ')} & "
                    f"{100 * pair['wrong_rate']:.1f} & {pair['mean_regret']:.4f} & "
                    f"{100 * six['wrong_rate']:.1f} & {six['mean_regret']:.4f}\\\\"
                )
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    (build / "llm_workflow_choice_table.tex").write_text("\n".join(lines) + "\n")
    print(out)
    for key in sorted(report["pairs"]):
        if "|0.1|400|" in key:
            pair = report["pairs"][key]
            print(f"{key}: wrong {100 * pair['wrong_rate']:.1f}% regret {pair['mean_regret']:.4f}")


if __name__ == "__main__":
    main()
