#!/usr/bin/env python3
"""Post hoc FinQA metric and decision audit.

This is a sensitivity check: the frozen severity metric,
targets, and results are unchanged and remain the study's evidence.

1. Metric audit: the frozen convention-invariant error takes the minimum
   over {y, y/100, 100y} for every answer, which in principle can score a
   genuine hundredfold error as correct. This script classifies every
   rescaled valid candidate (percent-flagged versus not) and recomputes a
   unit-aware variant that allows the rescaled branches only when the
   candidate printed a percent sign or the question asks for a
   percentage-like quantity (keyword list below).

2. Target sensitivity: exact per-question CVaR truths are recomputed
   under the unit-aware metric for every panel question, workflow, and
   generator, from the frozen kernels (no model queries), together with
   the screen-membership changes.

3. Reviewer diagnostics: per-question truth arrays for both templates on
   both panels (the Qwen equality question), the aggregation formula
   inputs, and the zero-total-influence counts.

Writes results/finqa_metric_audit.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tis.engine import compute_influences  # noqa: E402
from tis.finqa.models import finqa_question_model  # noqa: E402

ALPHA = 0.1
H = 3
TOL_REL = 0.005
SPREAD_THRESHOLD = 0.25
PERCENT_KEYWORDS = ("percent", "percentage", "portion", "proportion", "ratio",
                    " rate", "growth")


def frozen_error(y: float, g: float) -> float:
    return min(abs(y - g), abs(y / 100 - g), abs(y * 100 - g))


def _fraction_scale(value: float, flagged_percent: bool) -> float:
    """Semantic representation of a percentage-like quantity as a fraction.

    A percent-flagged value is a percentage by declaration (divide by 100,
    whatever its magnitude -- '2000%' becomes 20.0, preserving genuine
    scale errors). A bare value in a percentage-like question is read as a
    fraction when |v| <= 1.5 and as percent-scale otherwise. The rule is
    fixed in advance and never chosen to minimize error against the gold.
    """
    if flagged_percent:
        return value / 100.0
    return value if abs(value) <= 1.5 else value / 100.0


def unit_aware_error(y: float, g: float, is_percent: bool, question: str) -> float:
    percent_like = is_percent or any(k in question.lower() for k in PERCENT_KEYWORDS)
    if not percent_like:
        return abs(y - g)
    return abs(_fraction_scale(y, is_percent) - _fraction_scale(g, False))


def utility(e0: float, g: float) -> float:
    e = max(0.0, e0 - TOL_REL * abs(g))
    return 1.0 - e / (e + abs(g))


def scale_for(item) -> float:
    """Utility scale |g| on the same representation the error uses."""
    g = float(item["gold_answer"])
    if any(k in str(item["question"]).lower() for k in PERCENT_KEYWORDS):
        return abs(_fraction_scale(g, False))
    return abs(g)


def bank_utilities(bank, item, error_fn):
    out = []
    s = scale_for(item)
    for c in bank["candidates"]:
        if not c["valid"]:
            out.append(0.0)
            continue
        e0 = error_fn(float(c["answer"]), float(item["gold_answer"]),
                      bool(c["is_percent"]), str(item["question"]))
        e = max(0.0, e0 - TOL_REL * s)
        out.append(1.0 - e / (e + s) if s > 0 else 0.0)
    return out


def main() -> None:
    report: dict = {"percent_keywords": list(PERCENT_KEYWORDS),
                    "candidate_classification": {}, "targets": {},
                    "screen": {}, "truth_arrays": {}, "influence_diagnostics": {}}

    counts = {"valid": 0, "rescued_percent": 0, "rescued_nonpercent": 0,
              "erased_genuine": 0}
    for tag in ("dev2", "heldout2"):
        banks = json.load(open(ROOT / f"frozen/finqa_banks_{tag}.json"))["banks"]
        panel = {json.loads(l)["item_id"]: json.loads(l)
                 for l in open(ROOT / f"frozen/finqa_panel_{tag}.jsonl")}
        for b in banks:
            item = panel[b["item_id"]]
            g = float(item["gold_answer"])
            for c in b["candidates"]:
                if not c["valid"]:
                    continue
                counts["valid"] += 1
                y = float(c["answer"])
                raw, mn = abs(y - g), frozen_error(y, g)
                if mn >= raw - 1e-12:
                    continue
                u_raw, u_min = utility(raw, g), utility(mn, g)
                if u_min - u_raw <= 0.05:
                    continue
                key = "rescued_percent" if c["is_percent"] else "rescued_nonpercent"
                counts[key] += 1
                if not c["is_percent"] and u_raw < 0.5 and u_min > 0.9:
                    counts["erased_genuine"] += 1
    report["candidate_classification"] = counts

    kernel_sets = {
        ("dev2", "qwen", "ordinary"): "finqa_kernels_dev2_ordinary.json",
        ("dev2", "qwen", "unitcheck"): "finqa_kernels_dev2_unitcheck.json",
        ("dev2", "phi", "ordinary"): "finqa_kernels_dev2_phi_ordinary.json",
        ("dev2", "phi", "unitcheck"): "finqa_kernels_dev2_phi_unitcheck.json",
        ("heldout2", "qwen", "ordinary"): "finqa_kernels_heldout2_ordinary.json",
        ("heldout2", "qwen", "unitcheck"): "finqa_kernels_heldout2_unitcheck.json",
        ("heldout2", "phi", "ordinary"): "finqa_kernels_heldout2_phi_ordinary.json",
        ("heldout2", "phi", "unitcheck"): "finqa_kernels_heldout2_phi_unitcheck.json",
    }
    for (tag, gen, wf), fname in kernel_sets.items():
        art = json.load(open(ROOT / "frozen" / fname))
        banks = {int(b["panel_index"]): b
                 for b in json.load(open(ROOT / f"frozen/finqa_banks_{tag}.json"))["banks"]}
        panel = {json.loads(l)["item_id"]: json.loads(l)
                 for l in open(ROOT / f"frozen/finqa_panel_{tag}.jsonl")}
        truths_frozen, truths_ua, zero_inf = [], [], 0
        spread_changes = []
        for q in art["questions"]:
            b = banks[int(q["panel_index"])]
            item = panel[q["item_id"]]
            from tis.finqa.protocol import utilities as frozen_utilities
            u_frozen = frozen_utilities(b["candidates"], float(item["gold_answer"]))
            u_ua = bank_utilities(b, item, unit_aware_error)
            model_f = finqa_question_model(q, b["candidates"], H)
            t_f = compute_influences(model_f, ALPHA)
            truths_frozen.append(float(t_f.solution.cvar))
            if float(np.sum(t_f.tail_scales)) <= 0:
                zero_inf += 1
            # unit-aware target: same kernels, replaced utilities
            bank_ua = {"candidates": [dict(c) for c in b["candidates"]]}
            # finqa_question_model derives utilities via protocol.utilities on
            # the candidate list; emulate by patching answers is intrusive, so
            # rebuild the model manually from its own construction with the
            # unit-aware utilities via metadata override:
            import tis.finqa.models as fm
            orig = fm.utilities
            fm.utilities = lambda bank, gold: u_ua
            try:
                model_u = finqa_question_model(q, b["candidates"], H)
            finally:
                fm.utilities = orig
            t_u = compute_influences(model_u, ALPHA)
            truths_ua.append(float(t_u.solution.cvar))
            if gen == "qwen" and wf == "ordinary":
                s_f = max(u_frozen) - min(u_frozen)
                s_u = max(u_ua) - min(u_ua)
                if (s_f >= SPREAD_THRESHOLD) != (s_u >= SPREAD_THRESHOLD):
                    spread_changes.append(q["item_id"])
        d = np.abs(np.array(truths_frozen) - np.array(truths_ua))
        key = f"{tag}|{gen}|{wf}"
        report["targets"][key] = {
            "questions": len(truths_frozen),
            "max_abs_target_change": float(d.max()),
            "mean_abs_target_change": float(d.mean()),
            "changed_above_1e-3": int((d > 1e-3).sum()),
        }
        report["truth_arrays"][key] = [round(t, 10) for t in truths_frozen]
        report["influence_diagnostics"][key] = {"zero_total_influence": zero_inf}
        if gen == "qwen" and wf == "ordinary":
            report["screen"][tag] = {"membership_changes_under_unit_aware":
                                     spread_changes}

    out = ROOT / "results" / "finqa_metric_audit.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(out)
    print(json.dumps(counts, indent=1))
    for key, t in sorted(report["targets"].items()):
        print(key, "max target change", f"{t['max_abs_target_change']:.4f}",
              "changed>1e-3:", t["changed_above_1e-3"])
    for tag, s in report["screen"].items():
        print(tag, "screen membership changes:", len(s["membership_changes_under_unit_aware"]))


if __name__ == "__main__":
    main()
