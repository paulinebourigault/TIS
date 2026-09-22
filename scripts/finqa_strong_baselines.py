#!/usr/bin/env python3
"""Per-workflow strong-baseline comparison for the FinQA held-out study.

For each generator (qwen, phi), workflow (ordinary, unitcheck), and budget:
direct panel-MSE ratios with paired uncertainty for plain TIS and anchored
TIS against learned occupancy and complete rollouts, from the committed
per-replication panel MSEs. Methods within a replication share coupled
sampling streams (the rollout stream is separate but replication-indexed),
so the paired difference of squared panel errors over the 300 common
replications gives each comparison its own z. Ratios are of mean panel
MSEs; z is mean(paired difference)/SE and is negative when the first
method is better.

If extended-budget runs (results/finqa_heldout_<wf><gen>_b48/) are present,
their budgets are included and marked "extension".

Writes results/finqa_strong_baselines.json and prints a summary.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = {("qwen", "ordinary"): "finqa_heldout_ordinary",
        ("phi", "ordinary"): "finqa_heldout_ordinary_phi",
        ("qwen", "unitcheck"): "finqa_heldout_unitcheck",
        ("phi", "unitcheck"): "finqa_heldout_unitcheck_phi"}
COMPARISONS = [("tis", "learned_occupancy"), ("tis", "complete_rollout"),
               ("tis_anchored", "learned_occupancy"),
               ("tis_anchored", "complete_rollout"),
               ("tis_anchored", "tis")]


def load(directory: Path):
    cells: dict[tuple[int, str], dict[int, float]] = defaultdict(dict)
    for r in csv.DictReader((directory / "panel_replications.csv").open()):
        cells[(int(r["budget_per_group"]), r["method"])][int(r["replication"])] = \
            float(r["panel_mse"])
    return cells


def main() -> None:
    out = {"definition": "ratio = mean panel MSE(method)/mean panel MSE("
                         "reference); z from paired per-replication squared-"
                         "error differences over coupled replications; "
                         "negative z favors the method"}
    for (gen, wf), run in RUNS.items():
        cells = load(ROOT / "results" / run)
        ext_dir = ROOT / "results" / f"{run}_b48"
        ext_budgets = set()
        if (ext_dir / "panel_replications.csv").exists():
            ext = load(ext_dir)
            for k, v in ext.items():
                cells[k].update(v)
                ext_budgets.add(k[0])
        budgets = sorted({b for b, _ in cells})
        block = {}
        for b in budgets:
            row = {}
            for method, ref in COMPARISONS:
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
            block[str(b)] = row
        out[f"{gen}|{wf}"] = {"budgets": budgets,
                              "extension_budgets": sorted(ext_budgets),
                              "cells": block}
        for b in budgets:
            tag = " (ext)" if b in ext_budgets else ""
            r = block[str(b)]
            print(f"{gen} {wf} b{b}{tag}: "
                  f"anch/occ {r['tis_anchored_vs_learned_occupancy']['ratio']:.3f} "
                  f"(z {r['tis_anchored_vs_learned_occupancy']['paired_z']:+.1f}), "
                  f"anch/roll {r['tis_anchored_vs_complete_rollout']['ratio']:.3f} "
                  f"(z {r['tis_anchored_vs_complete_rollout']['paired_z']:+.1f}), "
                  f"tis/occ {r['tis_vs_learned_occupancy']['ratio']:.2f} "
                  f"(z {r['tis_vs_learned_occupancy']['paired_z']:+.1f}), "
                  f"tis/roll {r['tis_vs_complete_rollout']['ratio']:.2f} "
                  f"(z {r['tis_vs_complete_rollout']['paired_z']:+.1f})")
    (ROOT / "results/finqa_strong_baselines.json").write_text(
        json.dumps(out, indent=1) + "\n")
    print("written results/finqa_strong_baselines.json")


if __name__ == "__main__":
    main()
