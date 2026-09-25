#!/usr/bin/env python3
"""Calibrate, audit, and simulate the FinQA tool-fault experiment.

Steps (each skipped when its output already exists, so the run resumes):
  smoke     two development questions per generator: validity checks only
  calibrate all 145 prompt kernels for the 50 held-out questions
  audit     generation audit on sampled normal and faulted states
  simulate  the six (generator, p) cells from configs/finqa_toolfault_*.json

Usage: python scripts/run_finqa_toolfault.py smoke|calibrate|audit|simulate [gen]
Design: frozen/finqa_toolfault_declaration.md.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tis.finqa.toolfault import (  # noqa: E402
    STATES_TF, action_for_tf_state, render_tf_messages,
)

GENERATORS = {"qwen": "configs/finqa_protocol.json",
              "phi": "configs/finqa_protocol_phi.json"}
AUDIT_STATES = (0, 14, 40, 58, 72, 73 + 14, 73 + 40, 73 + 58, 144)
FAULT_PROBS = ("0.01", "0.03", "0.1")
# Review workflow; TF_WORKFLOW=unitcheck selects the unit-audit reviewer.
WORKFLOW = os.environ.get("TF_WORKFLOW", "ordinary")
TAG = "" if WORKFLOW == "ordinary" else f"{WORKFLOW}_"


def kernels_path(gen: str, panel: str) -> Path:
    return ROOT / "frozen" / f"finqa_kernels_{panel}_toolfault_{TAG}{gen}.json"


def calibrate(gen: str, panel: str, questions: int | None = None) -> None:
    from tis.finqa.calibrate import calibrate_kernels
    out = kernels_path(gen, panel)
    if out.exists():
        print("exists:", out)
        return
    panel_file = ROOT / "frozen" / f"finqa_panel_{panel}.jsonl"
    banks_file = ROOT / "frozen" / f"finqa_banks_{panel}.json"
    if questions is not None:
        lines = panel_file.read_text().splitlines()[:questions]
        panel_file = ROOT / "build" / f"finqa_panel_{panel}_first{questions}.jsonl"
        panel_file.parent.mkdir(exist_ok=True)
        panel_file.write_text("\n".join(lines) + "\n")
    calibrate_kernels(ROOT / GENERATORS[gen], panel_file, banks_file, out, WORKFLOW,
                      render=render_tf_messages, states=STATES_TF,
                      study="finqa_tool_fault", action_of=action_for_tf_state)
    print("wrote", out)


def smoke(gen: str) -> None:
    calibrate(gen, "dev2", questions=2)
    art = json.loads(kernels_path(gen, "dev2").read_text())
    for q in art["questions"]:
        laws = {k["state_index"]: np.asarray(k["probabilities"]) for k in q["kernels"]}
        assert sorted(laws) == list(range(STATES_TF))
        assert all(abs(v.sum() - 1) < 1e-9 and np.all(v > 0) for v in laws.values())
        tv = [0.5 * np.abs(laws[s] - laws[s + 72]).sum() for s in range(1, 73)]
        print(f"{gen} dev q{q['panel_index']}: total-variation normal vs faulted "
              f"median {np.median(tv):.3f}, max {np.max(tv):.3f}")


def audit(gen: str) -> None:
    from tis.finqa.calibrate import audit_generations
    out = ROOT / "frozen" / f"finqa_audit_heldout2_toolfault_{TAG}{gen}.json"
    if out.exists():
        print("exists:", out)
        return
    report = audit_generations(kernels_path(gen, "heldout2"), ROOT / GENERATORS[gen],
                               ROOT / "frozen/finqa_panel_heldout2.jsonl",
                               ROOT / "frozen/finqa_banks_heldout2.json", out,
                               state_indices=AUDIT_STATES, render=render_tf_messages)
    print(gen, "audit passed:", report["passed"], "smallest p:", report["smallest_p_value"])


def _simulate_cell(args):
    gen, p = args
    from tis.finqa.runner import run_finqa_study
    config = ROOT / "configs" / f"finqa_toolfault_{TAG}{gen}_p{p}.json"
    out = ROOT / "results" / f"finqa_toolfault_{TAG}{gen}_p{p}"
    if (out / "panel_summary.csv").exists():
        return f"exists {out}"
    run_finqa_study(config, kernels_path(gen, "heldout2"),
                    ROOT / "frozen/finqa_banks_heldout2.json", out, overwrite=True)
    return f"DONE {out}"


def simulate() -> None:
    cells = [(g, p) for g in GENERATORS for p in FAULT_PROBS]
    with ProcessPoolExecutor(max_workers=len(cells)) as pool:
        for message in pool.map(_simulate_cell, cells):
            print(message, flush=True)


if __name__ == "__main__":
    step = sys.argv[1]
    gens = [sys.argv[2]] if len(sys.argv) > 2 else list(GENERATORS)
    if step == "simulate":
        simulate()
    else:
        for g in gens:
            {"smoke": smoke, "audit": audit,
             "calibrate": lambda gen: calibrate(gen, "heldout2")}[step](g)
