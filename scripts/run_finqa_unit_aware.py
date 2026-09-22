#!/usr/bin/env python3
"""Post hoc unit-aware-metric rerun of the FinQA held-out simulations.

Sensitivity rerun: identical frozen
kernels, banks, seeds, budgets, and replications; only the severity
utilities change, to the unit-aware variant defined in
scripts/finqa_metric_audit.py (rescaled error branches only for
percent-flagged answers or percentage-like questions). Outputs land in
results/finqa_heldout_<wf>[_phi]_ua and never replace the frozen-metric
results.

Usage: python scripts/run_finqa_unit_aware.py <ordinary|unitcheck> [phi]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from finqa_metric_audit import bank_utilities, unit_aware_error  # noqa: E402
import tis.finqa.models as fm  # noqa: E402
from tis.finqa.runner import run_finqa_study  # noqa: E402


def main() -> None:
    wf = sys.argv[1]
    phi = len(sys.argv) > 2 and sys.argv[2] == "phi"
    gen = "_phi" if phi else ""
    kernels = ROOT / f"frozen/finqa_kernels_heldout2{'_phi' if phi else ''}_{wf}.json"
    banks_path = ROOT / "frozen/finqa_banks_heldout2.json"
    config = ROOT / f"configs/finqa_heldout_{wf}{gen}.json"

    panel = {json.loads(l)["item_id"]: json.loads(l)
             for l in open(ROOT / "frozen/finqa_panel_heldout2.jsonl")}
    banks = json.load(open(banks_path))["banks"]

    # The runner passes each bank's candidate list (a stable object) into
    # finqa_question_model, which calls the module-level utilities(). Patch
    # it with a lookup keyed by the candidate values themselves.
    table: dict = {}
    for b in banks:
        item = panel[b["item_id"]]
        g = float(item["gold_answer"])
        vals = tuple(
            (None if not c["valid"] else round(float(c["answer"]), 12),
             bool(c["is_percent"]))
            for c in b["candidates"]
        )
        table[(round(g, 12), vals)] = bank_utilities(b, item, unit_aware_error)

    def ua_utilities(bank, gold):
        vals = tuple(
            (None if not c["valid"] else round(float(c["answer"]), 12),
             bool(c["is_percent"]))
            for c in bank
        )
        return list(table[(round(float(gold), 12), vals)])

    fm.utilities = ua_utilities
    out = run_finqa_study(config, kernels, banks_path,
                          ROOT / f"results/finqa_heldout_{wf}{gen}_ua",
                          overwrite=True)
    print(out)


if __name__ == "__main__":
    main()
