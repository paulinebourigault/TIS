#!/usr/bin/env python3
"""Live loop, study 1: references recomputed on the GPU type of each question's design runs (Appendix H.2).

Questions 1-12 ran on an H100 and keep their H100 references; questions 13-20 ran on
A100s and get the references recomputed on an A100 (reference_q12-20_0of1.json). The
validation compares, for every question, the 40-replication mean of the complete-rollout
CVaR with the question's reference (no design under test is involved). The corrected
analysis is the `analyze` step of scripts/rebuttal_live_loop.py run on a directory holding the same design rows
and the hardware-matched references.

Usage: python scripts/rebuttal_live_loop_hwmatched.py
Writes results/rebuttal_live_loop_hwmatched/ (validation.json, analysis.json, summary.csv).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "results/rebuttal_live_loop"
OUT = ROOT / "results/rebuttal_live_loop_hwmatched"


def main() -> None:
    order = json.loads((SOURCE / "screening.json").read_text())["retained_item_ids"]
    h100 = json.loads((SOURCE / "reference_shard0of1.json").read_text())["items"]
    a100 = json.loads((SOURCE / "reference_q12-20_0of1.json").read_text())["items"]
    if sorted(a100) != sorted(order[12:20]):
        raise SystemExit(f"A100 references cover {sorted(a100)}, expected {sorted(order[12:20])}")
    reference = {it: (h100[it] if q < 12 else a100[it]) for q, it in enumerate(order)}
    hardware = {it: ("H100" if q < 12 else "A100") for q, it in enumerate(order)}

    rollouts = defaultdict(list)
    raw = sorted(SOURCE.glob("raw_shard*.csv"))
    for path in raw:
        for row in csv.DictReader(path.open()):
            if row["design"] == "complete_rollout":
                rollouts[row["item_id"]].append(float(row["estimate"]))
    validation = []
    for q, it in enumerate(order):
        e = np.array(rollouts[it])
        se = float(np.sqrt(e.var(ddof=1) / e.size + reference[it]["bootstrap_variance"]))
        z = float((e.mean() - reference[it]["cvar"]) / se) if se > 1e-12 else 0.0
        validation.append({"question": q + 1, "item_id": it, "hardware": hardware[it],
                           "reference_cvar": reference[it]["cvar"],
                           "h100_reference_cvar": h100[it]["cvar"],
                           "rollout_mean": float(e.mean()), "z": z,
                           "error_rate_reference": reference[it]["error_rate"]})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation.json").write_text(json.dumps(validation, indent=1) + "\n")
    for v in validation:
        print(f'q{v["question"]:>2} {v["item_id"]:>6} {v["hardware"]} reference {v["reference_cvar"]:.4f} '
              f'(H100 {v["h100_reference_cvar"]:.4f}) rollout mean {v["rollout_mean"]:.4f} z {v["z"]:+.1f}')

    for name in ("screening.json", "scale.json"):
        shutil.copy(SOURCE / name, OUT / name)
    for path in raw:
        shutil.copy(path, OUT / path.name)
    (OUT / "reference_shard_hwmatched.json").write_text(json.dumps(
        {"items": reference, "hardware": hardware,
         "sources": {"H100": "results/rebuttal_live_loop/reference_shard0of1.json",
                     "A100": "results/rebuttal_live_loop/reference_q12-20_0of1.json"}}, indent=1) + "\n")
    subprocess.run([sys.executable, str(ROOT / "scripts/rebuttal_live_loop.py"), "analyze"], check=True,
                   env={**os.environ, "LIVE_LOOP_OUT": str(OUT)})


if __name__ == "__main__":
    main()
