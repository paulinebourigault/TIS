#!/usr/bin/env python3
"""Blending-control analysis for the MMLU-Pro studies.

For each result directory results/<prefix>_<generator>/ (confident-error or
Brier runs): panel-MSE ratios to uniform, and paired contrasts over coupled
replications of plain and anchored TIS against learned occupancy, learned
mean, the occupancy-uniform and occupancy-mean blends, and rollouts.
Negative z favors the first method.

Usage: python scripts/llm_mixture_analysis.py llm_confident_mix|llm_brier_mix|llm_divpred
Writes results/<prefix>_analysis.json.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GENERATORS = ("qwen4b", "phi", "granite", "mistral24b", "qwen32b", "glm32b",
              "hs_qwen4b", "hs_phi", "hs_qwen32b",
              "cautious_qwen4b", "cautious_qwen4b_conf")
REFS = ("learned_occupancy", "learned_mean", "occ_plus_uniform",
        "occ_plus_mean", "complete_rollout")
SHOWN = ("tis", "tis_anchored", "learned_occupancy", "learned_mean",
         "occ_plus_uniform", "occ_plus_mean", "complete_rollout", "oracle_tail")


def main() -> None:
    prefix = sys.argv[1]
    out = {}
    for gen in GENERATORS:
        path = ROOT / "results" / f"{prefix}_{gen}" / "panel_replications.csv"
        if not path.exists():
            continue
        data: dict[tuple[int, str], dict[int, float]] = defaultdict(dict)
        for r in csv.DictReader(path.open()):
            data[(int(r["budget_per_shared_group"]), r["method"])][int(r["replication"])] = \
                float(r["panel_mean_squared_error"])
        cell = {}
        for b in sorted({k[0] for k in data}):
            uni = np.mean(list(data[(b, "uniform")].values()))
            row = {"ratio_to_uniform": {m: float(np.mean(list(data[(b, m)].values())) / uni)
                                        for m in SHOWN}, "paired": {}}
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
            cell[str(b)] = row
            pr, rt = row["paired"], row["ratio_to_uniform"]
            print(f"{gen:11s} b{b}: anch/unif {rt['tis_anchored']:.3f} | "
                  f"anch/mix_u {pr['tis_anchored_vs_occ_plus_uniform']['ratio']:.3f}"
                  f" ({pr['tis_anchored_vs_occ_plus_uniform']['z']:+.1f}) | "
                  f"anch/mix_m {pr['tis_anchored_vs_occ_plus_mean']['ratio']:.3f}"
                  f" ({pr['tis_anchored_vs_occ_plus_mean']['z']:+.1f}) | "
                  f"anch/occ {pr['tis_anchored_vs_learned_occupancy']['ratio']:.3f}"
                  f" ({pr['tis_anchored_vs_learned_occupancy']['z']:+.1f}) | "
                  f"anch/roll {pr['tis_anchored_vs_complete_rollout']['ratio']:.3f}"
                  f" ({pr['tis_anchored_vs_complete_rollout']['z']:+.1f}) | "
                  f"tis/mix_m {pr['tis_vs_occ_plus_mean']['ratio']:.3f}"
                  f" ({pr['tis_vs_occ_plus_mean']['z']:+.1f})")
        out[gen] = cell
    (ROOT / "results" / f"{prefix}_analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print("written", ROOT / "results" / f"{prefix}_analysis.json")


if __name__ == "__main__":
    main()
