#!/usr/bin/env python3
"""Analysis of the FinQA tool-fault experiment.

Per generator, fault probability, and budget: panel-MSE ratios to uniform
and paired contrasts (coupled streams; rollouts on their own stream) of
plain and anchored TIS against learned occupancy, learned mean, both
mixture controls, and rollouts. Population diagnostics per cell, computed
from the calibrated kernels: exact CVaR, margins, uniform-to-oracle ratio,
and the share of tail influence versus occupancy carried by the faulted
states. Writes results/finqa_toolfault_analysis.json.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tis.engine import compute_influences  # noqa: E402
from tis.finqa.toolfault import OUTCOMES, toolfault_question_model  # noqa: E402
from tis.llm_study.runner import _occupancy_scores  # noqa: E402

# TF_WORKFLOW=unitcheck analyzes the unit-audit review workflow.
TAG = "" if os.environ.get("TF_WORKFLOW", "ordinary") == "ordinary" else f"{os.environ['TF_WORKFLOW']}_"
GENS = ("qwen", "phi")
PROBS = ("0.01", "0.03", "0.1")
METHODS = ("tis", "tis_anchored", "learned_occupancy", "learned_mean",
           "occ_plus_uniform", "occ_plus_mean", "complete_rollout", "oracle_tail")
REFS = ("learned_occupancy", "learned_mean", "occ_plus_uniform",
        "occ_plus_mean", "complete_rollout")


def population(gen: str, p: float) -> dict:
    art = json.loads((ROOT / f"frozen/finqa_kernels_heldout2_toolfault_{TAG}{gen}.json").read_text())
    banks = {int(b["panel_index"]): b["candidates"] for b in
             json.loads((ROOT / "frozen/finqa_banks_heldout2.json").read_text())["banks"]}
    fault_tail, fault_occ, fault_mean, ratios, margins = [], [], [], [], []
    for q in art["questions"]:
        model = toolfault_question_model(q, banks[int(q["panel_index"])], 3, p)
        truth = compute_influences(model, 0.1)
        sigma = np.asarray(truth.tail_scales)
        mean = np.asarray(truth.mean_scales)
        occ = _occupancy_scores(model, truth)
        faulted = np.array([s > OUTCOMES for s, _ in model.query_groups])
        margins.append(float(truth.solution.margin))
        if sigma.sum() > 0:
            fault_tail.append(float(sigma[faulted].sum() / sigma.sum()))
            ratios.append(model.group_count * float((sigma ** 2).sum()) / float(sigma.sum()) ** 2)
        if mean.sum() > 0:
            fault_mean.append(float(mean[faulted].sum() / mean.sum()))
        fault_occ.append(float(occ[faulted].sum() / occ.sum()))
    return {"faulted_share_tail_influence_median": float(np.median(fault_tail)),
            "faulted_share_mean_influence_median": float(np.median(fault_mean)),
            "faulted_share_occupancy_median": float(np.median(fault_occ)),
            "uniform_to_oracle_median": float(np.median(ratios)),
            "min_margin": float(min(margins)),
            "zero_tail_questions": len(art["questions"]) - len(fault_tail)}


def main() -> None:
    out = {}
    for gen in GENS:
        for p in PROBS:
            cell = f"{gen}|p{p}"
            path = ROOT / f"results/finqa_toolfault_{TAG}{gen}_p{p}/panel_replications.csv"
            if not path.exists():
                print("missing", path)
                continue
            data: dict[tuple[int, str], dict[int, float]] = defaultdict(dict)
            for r in csv.DictReader(path.open()):
                data[(int(r["budget_per_group"]), r["method"])][int(r["replication"])] = \
                    float(r["panel_mse"])
            entry = {"population": population(gen, float(p)), "budgets": {}}
            for b in sorted({k[0] for k in data}):
                uni = np.mean(list(data[(b, "uniform")].values()))
                row = {"ratio_to_uniform": {m: float(np.mean(list(data[(b, m)].values())) / uni)
                                            for m in METHODS}, "paired": {}}
                for meth in ("tis", "tis_anchored"):
                    for ref in REFS:
                        a, r = data[(b, meth)], data[(b, ref)]
                        common = sorted(set(a) & set(r))
                        d = np.array([a[k] - r[k] for k in common])
                        se = d.std(ddof=1) / np.sqrt(len(d))
                        row["paired"][f"{meth}_vs_{ref}"] = {
                            "ratio": float(np.mean([a[k] for k in common]) /
                                           np.mean([r[k] for k in common])),
                            "z": float(d.mean() / se) if se > 0 else 0.0}
                entry["budgets"][str(b)] = row
                pr = row["paired"]
                print(f"{cell} b{b}: anch/occ {pr['tis_anchored_vs_learned_occupancy']['ratio']:.3f}"
                      f" (z {pr['tis_anchored_vs_learned_occupancy']['z']:+.1f})  "
                      f"anch/mix_m {pr['tis_anchored_vs_occ_plus_mean']['ratio']:.3f}"
                      f" (z {pr['tis_anchored_vs_occ_plus_mean']['z']:+.1f})  "
                      f"anch/mix_u {pr['tis_anchored_vs_occ_plus_uniform']['ratio']:.3f}"
                      f" (z {pr['tis_anchored_vs_occ_plus_uniform']['z']:+.1f})  "
                      f"tis/mix_m {pr['tis_vs_occ_plus_mean']['ratio']:.3f}"
                      f" (z {pr['tis_vs_occ_plus_mean']['z']:+.1f})  "
                      f"anch/roll {pr['tis_anchored_vs_complete_rollout']['ratio']:.3f}"
                      f" (z {pr['tis_anchored_vs_complete_rollout']['z']:+.1f})  "
                      f"anch/unif {row['ratio_to_uniform']['tis_anchored']:.3f} oracle/unif {row['ratio_to_uniform']['oracle_tail']:.3f}")
            out[cell] = entry
            print(f"{cell} population: {entry['population']}")
    (ROOT / f"results/finqa_toolfault_{TAG}analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"written results/finqa_toolfault_{TAG}analysis.json")


if __name__ == "__main__":
    main()
