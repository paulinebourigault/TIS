#!/usr/bin/env python3
"""Workflow-decision pairing sensitivity check.

Inputs: the regenerated results/finqa_heldout_<wf><gen>_rawcheck runs.
Outputs, per generator:
  1. compact per-replication panel-mean estimates
     (results/finqa_heldout_<wf><gen>_rawcheck/panel_estimates.csv);
  2. wrong-selection rates under the original identical-index pairing
     (must reproduce the committed decision rates);
  3. wrong-selection rates under independent pairings: 500 seeded random
     permutations of the unitcheck replication indices per cell, giving
     the permutation-mean rate and its spread, plus one fresh-seed
     single permutation for a point comparison.
Ties in the panel estimates select 'ordinary' (the committed rule).
Writes finqa_decision_paircheck.json in the working directory.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = {("qwen", "ordinary"): "finqa_heldout_ordinary_rawcheck",
        ("qwen", "unitcheck"): "finqa_heldout_unitcheck_rawcheck",
        ("phi", "ordinary"): "finqa_heldout_ordinary_phi_rawcheck",
        ("phi", "unitcheck"): "finqa_heldout_unitcheck_phi_rawcheck"}
PERMS = 500
SEED = 20260927


def panel_estimates(directory: Path):
    diag = json.loads((directory / "diagnostics.json").read_text())
    truth = float(np.mean([q["truth"] for q in diag]))
    questions = len(diag)
    sums = defaultdict(float)
    counts = defaultdict(int)
    with (directory / "raw.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["budget_per_group"]), row["method"],
                   int(row["replication"]))
            sums[key] += float(row["estimate"])
            counts[key] += 1
    est = {k: v / questions for k, v in sums.items() if counts[k] == questions}
    out = directory / "panel_estimates.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["budget_per_group", "method", "replication",
                    "panel_mean_estimate"])
        for (b, m, r), v in sorted(est.items()):
            w.writerow([b, m, r, repr(v)])
    return truth, est


def main() -> None:
    report = {"protocol": {
        "permutations": PERMS, "seed": SEED,
        "tie_rule": "ties select ordinary",
        "note": "identical-index pairing reproduces the committed paired "
                "common-random-number evaluation; permutation pairing "
                "estimates independent-stream selection rates from the "
                "same marginal estimate distributions"}}
    rng = np.random.default_rng(SEED)
    for gen in ("qwen", "phi"):
        tro, eo = panel_estimates(ROOT / "results" / RUNS[(gen, "ordinary")])
        tru, eu = panel_estimates(ROOT / "results" / RUNS[(gen, "unitcheck")])
        gap = tro - tru
        better = "ordinary" if gap >= 0 else "unitcheck"
        cells = sorted({(b, m) for (b, m, _) in eo})
        gen_out = {"truth_ordinary": tro, "truth_unitcheck": tru,
                   "gap": gap, "better": better, "cells": {}}
        for b, m in cells:
            xo = np.array([eo[(b, m, r)] for r in range(300)
                           if (b, m, r) in eo and (b, m, r) in eu])
            xu = np.array([eu[(b, m, r)] for r in range(300)
                           if (b, m, r) in eo and (b, m, r) in eu])
            n = len(xo)
            pick_ord = xo >= xu  # ties -> ordinary
            wrong_paired = float(np.mean(pick_ord != (better == "ordinary")))
            rates = np.empty(PERMS)
            for p in range(PERMS):
                perm = rng.permutation(n)
                pick = xo >= xu[perm]
                rates[p] = np.mean(pick != (better == "ordinary"))
            gen_out["cells"][f"{m}|b{b}"] = {
                "replications": n,
                "wrong_rate_identical_index": wrong_paired,
                "wrong_rate_independent_mean": float(rates.mean()),
                "wrong_rate_independent_sd": float(rates.std(ddof=1)),
                "wrong_rate_independent_p5_p95": [
                    float(np.percentile(rates, 5)),
                    float(np.percentile(rates, 95))],
            }
        report[gen] = gen_out
        for key in ("uniform|b100", "uniform|b200", "tis_anchored|b100",
                    "tis_anchored|b200", "learned_occupancy|b200",
                    "complete_rollout|b200"):
            if key in gen_out["cells"]:
                c = gen_out["cells"][key]
                print(f"{gen} {key}: paired {c['wrong_rate_identical_index']:.3f}"
                      f" independent {c['wrong_rate_independent_mean']:.3f}"
                      f" (sd {c['wrong_rate_independent_sd']:.3f})")
    (ROOT / "results/finqa_decision_paircheck/finqa_decision_paircheck.json").write_text(
        json.dumps(report, indent=1) + "\n")
    print("written finqa_decision_paircheck.json")


if __name__ == "__main__":
    main()
