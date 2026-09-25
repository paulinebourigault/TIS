#!/usr/bin/env python3
"""Merge FinQA runs executed separately per budget into one result directory.

Every random stream in the FinQA runner is seeded by (master seed, horizon,
budget, question, replication, stage), so a run restricted to one budget
produces exactly the rows of that budget in the unsplit run. This script
concatenates the per-budget panel_replications.csv files of the unit-check
calculator-fault runs (results/finqa_toolfault_unitcheck_shards/) into
results/finqa_toolfault_unitcheck_<generator>_p<p>/panel_replications.csv.

Usage: python scripts/merge_budget_shards.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARDS = ROOT / "results/finqa_toolfault_unitcheck_shards"


def main() -> None:
    for generator in ("qwen", "phi"):
        for p in ("0.01", "0.03", "0.1"):
            name = f"finqa_toolfault_unitcheck_{generator}_p{p}"
            header, rows = None, []
            for budget in (100, 200, 400):
                with (SHARDS / f"{name}_b{budget}" / "panel_replications.csv").open() as stream:
                    part = list(csv.reader(stream))
                if header is None:
                    header = part[0]
                if part[0] != header:
                    raise ValueError(f"header mismatch in {name}_b{budget}")
                rows.extend(part[1:])
            out = ROOT / "results" / name
            out.mkdir(exist_ok=True)
            with (out / "panel_replications.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(rows)
            print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
