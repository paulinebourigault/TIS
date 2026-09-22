#!/usr/bin/env python3
"""Deterministic builder for results/tabular_reconciled.json.

This script assembles the reconciled table from the committed source
summaries, asserts that every duplicated environment/budget/method cell
agrees across sources to floating-point precision, and writes the JSON
deterministically (sorted keys). Run with --check to verify the committed
file matches a fresh rebuild without writing.

Cell key: "<budget_per_group as float>|<method>"; values carry mse and
mse_se copied verbatim from the source summaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ENVS = {"cliffwalking_primary": "cliffwalking", "inventory_untied_v1": "inventory"}
SOURCES = ["public_paper", "public_anchored", "replacement_paper",
           "replacement_anchored", "replacement_meanlearned", "tabular_rollout"]
RTOL = 1e-12


def build() -> dict:
    table: dict[str, dict[str, dict[str, float]]] = {v: {} for v in ENVS.values()}
    provenance: dict[tuple[str, str], str] = {}
    for source in SOURCES:
        path = ROOT / "results" / source / "summary.csv"
        for row in csv.DictReader(path.open()):
            env = ENVS.get(row["environment"])
            if env is None:
                continue
            key = f"{float(row['budget_per_group'])}|{row['method']}"
            cell = {"mse": float(row["mse"]), "mse_se": float(row["mse_se"])}
            if key in table[env]:
                old = table[env][key]
                for field in ("mse", "mse_se"):
                    if not math.isclose(old[field], cell[field],
                                        rel_tol=RTOL, abs_tol=0.0):
                        raise AssertionError(
                            f"duplicate cell disagrees: {env} {key} {field} "
                            f"{old[field]!r} ({provenance[env, key]}) vs "
                            f"{cell[field]!r} ({source})")
            else:
                table[env][key] = cell
                provenance[env, key] = source
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify the committed file matches; write nothing")
    args = parser.parse_args()
    table = build()
    out = ROOT / "results" / "tabular_reconciled.json"
    if args.check or out.exists():
        committed = json.loads(out.read_text())
        assert set(committed) == set(table), (set(committed), set(table))
        for env in table:
            assert set(committed[env]) == set(table[env]), (
                env, set(committed[env]) ^ set(table[env]))
            for key, cell in table[env].items():
                for field in ("mse", "mse_se"):
                    assert math.isclose(committed[env][key][field], cell[field],
                                        rel_tol=RTOL, abs_tol=0.0), (env, key, field)
        print(f"committed {out.name} matches a deterministic rebuild "
              f"({sum(len(v) for v in table.values())} cells, duplicates consistent)")
        if args.check:
            return
    out.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()
