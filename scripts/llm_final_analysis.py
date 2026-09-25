#!/usr/bin/env python3
"""One-stop summary of every post-hoc study around the closed-grid campaign.

Reads (where present) the anchored panels (six generators + high-stakes +
deployment-utility variants), the rollout baselines, the breadth panels
(high-stakes x three generators, cautious policy), the utility-sensitivity
panel, the occupancy floor cells, and the extended floor grid, and writes
``results/llm_final_numbers.json`` plus a compact console digest. Every
value is a panel-mean MSE normalized to uniform at alpha = .1 unless
stated; the H = 2,4,6 triple is reported at 400 queries per kernel.

Run on the machine that holds the result directories.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GENS = [
    ("Qwen3-4B", ""),
    ("Phi-4-mini", "_robustness"),
    ("Granite-4.2-8B", "_robustness2"),
    ("Mistral-24B", "_scale1"),
    ("Qwen3-32B", "_scale2"),
    ("GLM-4-32B", "_scale3"),
]
HORIZONS = ("2", "4", "6")


def norms(directory: Path, alpha: str = "0.1", budget: str = "400") -> dict:
    out: dict[tuple[str, str], float] = {}
    path = directory / "panel_summary.csv"
    if not path.exists():
        return out
    for row in csv.DictReader(path.open()):
        if (
            row["alpha"] == alpha
            and row["budget_per_shared_group"] == budget
            and row["normalized_to_uniform"]
        ):
            out[(row["horizon"], row["method"])] = float(row["normalized_to_uniform"])
    return out


def triple(v: dict, method: str) -> list | None:
    vals = [v.get((h, method)) for h in HORIZONS]
    return vals if all(x is not None for x in vals) else None


def main() -> None:
    report: dict[str, object] = {}

    anchored = {}
    for label, suffix in GENS:
        v = norms(ROOT / "results" / f"llm_study_anchored{suffix}")
        if v:
            anchored[label] = {
                m: triple(v, m)
                for m in ("tis", "learned_occupancy", "tis_anchored", "oracle_tail")
            }
    report["anchored_panels"] = anchored

    rollout = {}
    for label, suffix in GENS:
        v = norms(ROOT / "results" / f"llm_study_rollout{suffix}")
        if v:
            rollout[label] = {m: triple(v, m) for m in ("tis", "complete_rollout")}
    report["rollout"] = rollout

    breadth = {}
    for name, directory in (
        ("hs_qwen4b", "llm_study_hs"),
        ("hs_qwen4b_anchored", "llm_study_hs_anchored"),
        ("hs_phi4mini", "llm_study_hs_robustness"),
        ("hs_qwen32b", "llm_study_hs_scale2"),
        ("cautious_qwen4b", "llm_study_cautious_qwen4b"),
        ("confident_qwen4b", "llm_study_confident_qwen4b"),
        ("confident_anchored", "llm_study_confident_anchored"),
    ):
        v = norms(ROOT / "results" / directory)
        if v:
            breadth[name] = {
                m: triple(v, m)
                for m in (
                    "tis",
                    "learned_occupancy",
                    "tis_anchored",
                    "oracle_tail",
                    "occupancy",
                    "mean_influence",
                )
                if triple(v, m)
            }
    report["breadth"] = breadth

    floors: dict[str, dict] = {}
    floor_dirs = {
        "0.072": "llm_study_closed{s}",
        "0.15": "llm_floor_sensitivity_closed_l15/{n}",
        "0.25": "llm_floor_sensitivity_closed/{n}",
        "0.40": "llm_floor_sensitivity_closed_l40/{n}",
        "0.60": "llm_floor_sensitivity_closed_l60/{n}",
        "0.80": "llm_floor_sensitivity_closed_l80/{n}",
    }
    short = {
        "Qwen3-4B": "qwen4b", "Phi-4-mini": "phi4mini", "Granite-4.2-8B": "granite8b",
        "Mistral-24B": "mistral24b", "Qwen3-32B": "qwen32b", "GLM-4-32B": "glm32b",
    }
    for label, suffix in GENS:
        row = {}
        for lam, pattern in floor_dirs.items():
            v = norms(ROOT / "results" / pattern.format(s=suffix, n=short[label]))
            value = v.get(("6", "tis"))
            if value is not None:
                row[lam] = value
        floors[label] = row
    report["floor_curve_h6"] = floors

    occ_floor = {}
    for label in short.values():
        row = {}
        for lam, d in (("0.25", "llm_floor_occ_closed_l25"), ("0.40", "llm_floor_occ_closed_l40")):
            v = norms(ROOT / "results" / d / label)
            value = v.get(("6", "learned_occupancy"))
            if value is not None:
                row[lam] = value
        occ_floor[label] = row
    report["occupancy_floor_h6"] = occ_floor

    for name, d in (
        ("hs_floor_l40", "llm_floor_hs_l40"),
        ("cautious_floor_l40", "llm_floor_cautious_qwen4b_l40"),
        ("confident_floor_l40", "llm_floor_confident_qwen4b_l40"),
    ):
        entry = {}
        base = ROOT / "results" / d
        candidates = [base] if (base / "panel_summary.csv").exists() else sorted(base.glob("*"))
        for sub in candidates:
            v = norms(sub)
            row = {m: v.get(("6", m)) for m in ("tis", "learned_occupancy") if v.get(("6", m))}
            if row:
                entry[sub.name if sub != base else "cell"] = row
        report[name] = entry

    out = ROOT / "results" / "llm_final_numbers.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(out)
    for label in anchored:
        a = anchored[label]
        if a.get("tis_anchored"):
            print(
                f"{label:14s} anchored {['%.3f' % x for x in a['tis_anchored']]} "
                f"| occ {['%.3f' % x for x in a['learned_occupancy']]} "
                f"| tis {['%.3f' % x for x in a['tis']]}"
            )


if __name__ == "__main__":
    main()
