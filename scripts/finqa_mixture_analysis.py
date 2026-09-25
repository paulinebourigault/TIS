#!/usr/bin/env python3
"""Mixture-control ablation analysis for the FinQA held-out panels.

For each generator, workflow, and budget: panel-MSE ratios with paired z
for anchored TIS against the two equally regularized mixture controls
(0.5u + 0.5w_occ and 0.5w_mean + 0.5w_occ), and against learned
occupancy, learned mean, plain TIS, and rollouts, from the per-replication
panel MSEs of the results/finqa_heldout_<wf><gen>_mix runs. All learned
designs share coupled conditional-query streams within a replication, so
paired differences of squared panel errors give each contrast its own z
(negative favors the anchor). Also reports each mixture against learned
occupancy alone.

Writes results/finqa_mixture_controls.json and prints a summary.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = {("qwen", "ordinary"): "finqa_heldout_ordinary_mix",
        ("phi", "ordinary"): "finqa_heldout_ordinary_phi_mix",
        ("qwen", "unitcheck"): "finqa_heldout_unitcheck_mix",
        ("phi", "unitcheck"): "finqa_heldout_unitcheck_phi_mix"}
CONTRASTS = [("tis_anchored", "occ_plus_uniform"),
             ("tis_anchored", "occ_plus_mean"),
             ("tis_anchored", "learned_occupancy"),
             ("tis_anchored", "learned_mean"),
             ("tis_anchored", "tis"),
             ("tis_anchored", "complete_rollout"),
             ("occ_plus_uniform", "learned_occupancy"),
             ("occ_plus_mean", "learned_occupancy")]


def main() -> None:
    out = {"definition": "ratio = mean panel MSE(method)/mean panel MSE("
                         "reference); z from paired per-replication "
                         "squared-error differences; negative z favors "
                         "the method"}
    for (gen, wf), run in RUNS.items():
        cells: dict[tuple[int, str], dict[int, float]] = defaultdict(dict)
        path = ROOT / "results" / run / "panel_replications.csv"
        for r in csv.DictReader(path.open()):
            cells[(int(r["budget_per_group"]), r["method"])][int(r["replication"])] = \
                float(r["panel_mse"])
        budgets = sorted({b for b, _ in cells})
        block = {}
        for b in budgets:
            row = {}
            for method, ref in CONTRASTS:
                a, r = cells[(b, method)], cells[(b, ref)]
                common = sorted(set(a) & set(r))
                x = np.array([a[k] for k in common])
                y = np.array([r[k] for k in common])
                diff = x - y
                se = diff.std(ddof=1) / np.sqrt(len(diff))
                row[f"{method}_vs_{ref}"] = {
                    "ratio": float(x.mean() / y.mean()),
                    "paired_z": float(diff.mean() / se) if se > 0 else 0.0,
                    "replications": len(common),
                }
            uni = np.mean(list(cells[(b, "uniform")].values()))
            row["normalized_to_uniform"] = {
                m: float(np.mean(list(cells[(b, m)].values())) / uni)
                for m in ("tis_anchored", "occ_plus_uniform", "occ_plus_mean",
                          "learned_occupancy", "learned_mean", "tis",
                          "complete_rollout", "oracle_tail")}
            block[str(b)] = row
        out[f"{gen}|{wf}"] = block
        for b in budgets:
            r = block[str(b)]
            print(f"{gen} {wf} b{b}: "
                  f"anch/mix_u {r['tis_anchored_vs_occ_plus_uniform']['ratio']:.3f} "
                  f"(z {r['tis_anchored_vs_occ_plus_uniform']['paired_z']:+.1f}), "
                  f"anch/mix_m {r['tis_anchored_vs_occ_plus_mean']['ratio']:.3f} "
                  f"(z {r['tis_anchored_vs_occ_plus_mean']['paired_z']:+.1f}), "
                  f"anch/occ {r['tis_anchored_vs_learned_occupancy']['ratio']:.3f} "
                  f"(z {r['tis_anchored_vs_learned_occupancy']['paired_z']:+.1f})")
    (ROOT / "results/finqa_mixture_controls.json").write_text(
        json.dumps(out, indent=1) + "\n")
    print("written results/finqa_mixture_controls.json")


if __name__ == "__main__":
    main()
