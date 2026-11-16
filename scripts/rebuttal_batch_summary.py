#!/usr/bin/env python3
"""Per-batch MSE for every design of the tabular and inventory studies.

This writes, next to each study's raw.csv, a batch_summary.csv with the MSE and
Monte Carlo SE of every (case, budget, design) cell within each batch, and
the family (equal-case-weight) MSE per batch for the inventory studies.

Usage: python scripts/rebuttal_batch_summary.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STUDIES = (
    ("configs/rebuttal_cliffwalking_attribution.json", "arm", "error", None),
    ("configs/rebuttal_inventory_profit.json", "method", "error_profit", "case"),
    ("configs/rebuttal_inventory_profit_alpha05.json", "method", "error_profit", "case"),
    ("configs/rebuttal_inventory_profit_alpha01.json", "method", "error_profit", "case"),
    ("configs/rebuttal_checkpoint_access.json", "arm", "error", "case"),
)


def batches_for(config: dict, row: dict) -> list:
    if "batches" in config:
        return config["batches"]
    return config[row["setting"]]["batches"]  # checkpoint study: per setting


def main() -> None:
    for config_path, arm_field, error_field, case_field in STUDIES:
        config = json.loads((ROOT / config_path).read_text())
        out = ROOT / config["output_directory"]
        raw = out / "raw.csv"
        if not raw.exists():
            print(f"skip {config_path}: no raw.csv")
            continue
        cells = defaultdict(list)
        for row in csv.DictReader(raw.open()):
            rep = int(row["replication"])
            for lo, hi in batches_for(config, row):
                if lo <= rep < hi:
                    key = (row.get("setting", ""), row[case_field] if case_field else "", int(row["budget"]),
                           row[arm_field], f"{lo}-{hi}")
                    cells[key].append(float(row[error_field]) ** 2)
        rows = [{"setting": k[0], "case": k[1], "budget": k[2], "design": k[3], "batch": k[4],
                 "replications": len(v), "mse": float(np.mean(v)),
                 "mse_se": float(np.std(v, ddof=1) / np.sqrt(len(v)))} for k, v in sorted(cells.items())]
        family = defaultdict(list)
        if case_field:
            for r in rows:
                family[(r["setting"], r["budget"], r["design"], r["batch"])].append(r["mse"])
            rows += [{"setting": k[0], "case": "family", "budget": k[1], "design": k[2], "batch": k[3],
                      "replications": "", "mse": float(np.mean(v)), "mse_se": ""} for k, v in sorted(family.items())]
        with (out / "batch_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out / 'batch_summary.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
