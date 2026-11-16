#!/usr/bin/env python3
"""Holm adjustment for the checkpoint-access and rarer-tail comparisons.

P3 and P4 (checkpoint access, Appendix F.5) and P5 (alpha = .01, Appendix
F.4) form one family, separate from P1/P2 (whose
adjustment is in results/rebuttal_inventory_profit/analysis.json). Bootstrap
p-values of zero are floored at the bootstrap resolution 1/10,000.

Usage: python scripts/rebuttal_holm.py
Writes results/rebuttal_holm_followup.json.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {"P3": "results/rebuttal_checkpoint_access/analysis.json",
           "P4": "results/rebuttal_checkpoint_access/analysis.json",
           "P5": "results/rebuttal_inventory_profit_alpha01/analysis.json"}
RESOLUTION = 1.0 / 10000


def main() -> None:
    tests = {k: json.loads((ROOT / path).read_text())[k] for k, path in SOURCES.items()}
    raw = {k: max(v["bootstrap_p_two_sided"], RESOLUTION) for k, v in tests.items()}
    adjusted, running = {}, 0.0
    for rank, key in enumerate(sorted(raw, key=raw.get)):
        running = max(running, min(1.0, (len(raw) - rank) * raw[key]))
        adjusted[key] = running
    ratios = {k: v.get("mse_ratio", v.get("family_mse_ratio")) for k, v in tests.items()}
    report = {"raw_p_floor_at_bootstrap_resolution": raw, "holm_adjusted": adjusted, "ratios": ratios,
              "intervals": {k: v["ratio_ci95"] for k, v in tests.items()}}
    (ROOT / "results/rebuttal_holm_followup.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
