#!/usr/bin/env python3
"""Longer review loops: paired contrasts and the declared predictions.

Compares the MMLU-Pro confident-error runs at H = 8 and 10 with the H = 6
blending runs, and the held-out FinQA runs at H = 6 with the H = 3 blending
runs. For every cell it reports anchored-TIS panel-MSE ratios and paired z
against the occupancy-mean blend, complete rollouts, learned occupancy, and
the occupancy-uniform blend, plus the plain-TIS/anchor ratio. It then scores
the three predictions of frozen/long_horizon_declaration.md:

  P1  divergence rule: distance >= .24 predicts the anchor is resolved better
      than the mean blend in >= 2 of 3 budgets and worse in none; <= .18
      predicts resolved better in <= 1; otherwise no prediction;
  P2  kernel reuse: anchor/rollout ratio at 400 queries is lower at the
      longest new horizon than at the reference horizon;
  P3  pilot risk: plain-TIS/anchor ratio at 400 queries is higher at H = 10
      than at H = 6 (MMLU-Pro).

Distances are those recorded in the declaration (reproduced by
scripts/divergence_statistics.py). "Resolved" means |z| >= 2.

Usage: python scripts/long_horizon_analysis.py
Writes results/long_horizon_analysis.json.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BUDGETS = (100, 200, 400)
REFS = ("occ_plus_mean", "complete_rollout", "learned_occupancy", "occ_plus_uniform", "tis")

MMLU = {"qwen4b": "Qwen3-4B", "phi": "Phi-4-mini", "glm32b": "GLM-4-32B"}
FINQA = {"qwen_ordinary": ("Qwen3-4B, ordinary", "finqa_heldout_ordinary_mix"),
         "qwen_unitcheck": ("Qwen3-4B, unit check", "finqa_heldout_unitcheck_mix"),
         "phi_ordinary": ("Phi-4-mini, ordinary", "finqa_heldout_ordinary_phi_mix"),
         "phi_unitcheck": ("Phi-4-mini, unit check", "finqa_heldout_unitcheck_phi_mix")}
DECLARED_DISTANCE = {"mmlu_qwen4b_h8": .255, "mmlu_qwen4b_h10": .264, "mmlu_phi_h8": .191,
                     "mmlu_phi_h10": .193, "mmlu_glm32b_h8": .355, "mmlu_glm32b_h10": .389,
                     "finqa_qwen_ordinary": .000, "finqa_qwen_unitcheck": .000,
                     "finqa_phi_ordinary": .016, "finqa_phi_unitcheck": .098}


def load(directory: str) -> dict:
    path = ROOT / "results" / directory / "panel_replications.csv"
    data: dict[tuple[int, str], dict[int, float]] = defaultdict(dict)
    for row in csv.DictReader(path.open()):
        budget = int(row.get("budget_per_shared_group") or row["budget_per_group"])
        value = float(row.get("panel_mean_squared_error") or row["panel_mse"])
        data[(budget, row["method"])][int(row["replication"])] = value
    return data


def contrast(data: dict, budget: int, method: str, ref: str) -> dict:
    a, r = data[(budget, method)], data[(budget, ref)]
    keys = sorted(set(a) & set(r))
    diff = np.array([a[k] - r[k] for k in keys])
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    return {"ratio": float(np.mean([a[k] for k in keys]) / np.mean([r[k] for k in keys])),
            "z": float(diff.mean() / se) if se > 0 else 0.0}


def cells(data: dict) -> dict:
    out = {}
    for b in BUDGETS:
        uniform = np.mean(list(data[(b, "uniform")].values()))
        out[str(b)] = {"anchor_over_uniform": float(np.mean(list(data[(b, "tis_anchored")].values())) / uniform),
                       **{f"anchor_vs_{ref}": contrast(data, b, "tis_anchored", ref) for ref in REFS if ref != "tis"},
                       "tis_over_anchor": contrast(data, b, "tis", "tis_anchored")["ratio"]}
    return out


def p1(entry: dict, distance: float) -> dict:
    zs = [entry[str(b)]["anchor_vs_occ_plus_mean"]["z"] for b in BUDGETS]
    wins, losses = sum(z <= -2 for z in zs), sum(z >= 2 for z in zs)
    if distance >= .24:
        prediction, held = "win", wins >= 2 and losses == 0
    elif distance <= .18:
        prediction, held = "no advantage", wins <= 1
    else:
        prediction, held = "none", None
    return {"distance": distance, "prediction": prediction, "wins": wins, "losses": losses, "held": held}


def main() -> None:
    report = {"mmlu": {}, "finqa": {}, "scores": {"P1": {}, "P2": {}, "P3": {}}}
    for key, name in MMLU.items():
        reference = load(f"llm_confident_mix_{key}")
        ref_cells = {str(b): {"anchor_vs_complete_rollout": contrast(reference, b, "tis_anchored", "complete_rollout"),
                              "tis_over_anchor": contrast(reference, b, "tis", "tis_anchored")["ratio"]}
                     for b in BUDGETS}
        report["mmlu"][key] = {"generator": name, "h6": ref_cells}
        for horizon in (8, 10):
            entry = cells(load(f"llm_longh_{key}_h{horizon}"))
            report["mmlu"][key][f"h{horizon}"] = entry
            report["scores"]["P1"][f"mmlu_{key}_h{horizon}"] = p1(entry, DECLARED_DISTANCE[f"mmlu_{key}_h{horizon}"])
        before = ref_cells["400"]["anchor_vs_complete_rollout"]["ratio"]
        after = report["mmlu"][key]["h10"]["400"]["anchor_vs_complete_rollout"]["ratio"]
        report["scores"]["P2"][f"mmlu_{key}"] = {"h6": before, "h10": after, "held": after < before}
        before = ref_cells["400"]["tis_over_anchor"]
        after = report["mmlu"][key]["h10"]["400"]["tis_over_anchor"]
        report["scores"]["P3"][f"mmlu_{key}"] = {"h6": before, "h10": after, "held": after > before}
    for key, (name, reference_dir) in FINQA.items():
        reference = cells(load(reference_dir))
        entry = cells(load(f"finqa_longh_{key}"))
        report["finqa"][key] = {"setting": name, "h3": reference, "h6": entry}
        report["scores"]["P1"][f"finqa_{key}"] = p1(entry, DECLARED_DISTANCE[f"finqa_{key}"])
        before = reference["400"]["anchor_vs_complete_rollout"]["ratio"]
        after = entry["400"]["anchor_vs_complete_rollout"]["ratio"]
        report["scores"]["P2"][f"finqa_{key}"] = {"h3": before, "h6": after, "held": after < before}
    for prediction, scored in report["scores"].items():
        decided = [v["held"] for v in scored.values() if v["held"] is not None]
        print(f"{prediction}: held in {sum(decided)} of {len(decided)} decided settings")
    print("Anchor/rollout ratio (z), FinQA H=3 and H=6 at 100/200/400 queries per kernel:")
    for key, entry in report["finqa"].items():
        row = [entry[h][str(b)]["anchor_vs_complete_rollout"] for h in ("h3", "h6") for b in BUDGETS]
        print(f"  {entry['setting']:24s} " + "  ".join(f"{c['ratio']:.2f} ({c['z']:+.1f})" for c in row))
    print("Anchor/rollout ratio (z) at 400 queries, MMLU-Pro H=6/8/10:")
    for key, entry in report["mmlu"].items():
        row = [entry[h]["400"]["anchor_vs_complete_rollout"] for h in ("h6", "h8", "h10")]
        print(f"  {entry['generator']:12s} " + "  ".join(f"{c['ratio']:.2f} ({c['z']:+.1f})" for c in row))
    (ROOT / "results/long_horizon_analysis.json").write_text(json.dumps(report, indent=1) + "\n")
    print("written results/long_horizon_analysis.json")


if __name__ == "__main__":
    main()
