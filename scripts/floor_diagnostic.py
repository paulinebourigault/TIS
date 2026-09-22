#!/usr/bin/env python3
"""Retrospective pilot-computable instability diagnostic (supplement).

For each generator, horizon 6, alpha .1, 400 queries per kernel: per
question, the coefficient of variation of the pilot-estimated influence
scale of the largest estimated-influence group (across replications, which
a practitioner estimates from one pilot by split-halves), and the share of
the estimated scale sum held by that group. Reports pooled AUCs for ranking
questions on which learned TIS has higher MSE than uniform. Requires the
cluster-side allocation_summary.csv files; writes
results/llm_floor_diagnostic.json.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = [
    ("llm_study_paper", "Qwen3-4B"), ("llm_study_paper_robustness", "Phi-4-mini"),
    ("llm_study_paper_robustness2", "Granite-4.2-8B"), ("llm_study_paper_scale1", "Mistral-24B"),
    ("llm_study_paper_scale2", "Qwen3-32B"), ("llm_study_paper_scale3", "GLM-4-32B"),
]


def auc(scores: list[tuple[float, bool]]) -> float:
    pos = [s for s, f in scores if f]
    neg = [s for s, f in scores if not f]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    pooled = []
    per_generator = {}
    for name, label in MODELS:
        base = ROOT / "results" / name
        per_q = defaultdict(list)
        with (base / "allocation_summary.csv").open(newline="", encoding="utf-8") as stream:
            for r in csv.DictReader(stream):
                if (r["horizon"], r["alpha"], r["budget_per_shared_group"], r["method"]) != ("6", "0.1", "400", "tis"):
                    continue
                per_q[int(r["panel_index"])].append((float(r["mean_score"]), float(r["score_sd"])))
        mse = defaultdict(dict)
        for r in csv.DictReader((base / "question_summary.csv").open(newline="", encoding="utf-8")):
            if (r["horizon"], r["alpha"], r["budget_per_shared_group"]) == ("6", "0.1", "400") and r["method"] in ("tis", "uniform"):
                mse[int(r["panel_index"])][r["method"]] = float(r["mse"])
        rows = []
        for qi, groups in per_q.items():
            if mse[qi]["uniform"] <= 0:
                continue
            total = sum(m for m, _ in groups)
            if total <= 0:
                continue
            m_top, s_top = max(groups, key=lambda g: g[0])
            rows.append({
                "question": qi,
                "top_share": m_top / total,
                "top_cv": s_top / m_top if m_top > 0 else 0.0,
                "tis_over_uniform": mse[qi]["tis"] / mse[qi]["uniform"],
            })
            pooled.append((rows[-1], label))
        per_generator[label] = rows
    fails = [(r["top_cv"], r["tis_over_uniform"] > 1) for r, _ in pooled]
    result = {
        "definition": "top_cv is the coefficient of variation, across replications, of the pilot-estimated influence scale of the question's largest estimated-influence group; top_share is that group's share of the estimated scale sum",
        "setting": "H=6, alpha=0.1, 400 queries per kernel, learned TIS",
        "pooled_questions": len(pooled),
        "pooled_failures": sum(1 for r, _ in pooled if r["tis_over_uniform"] > 1),
        "auc_top_cv": auc(fails),
        "auc_top_share": auc([(r["top_share"], r["tis_over_uniform"] > 1) for r, _ in pooled]),
        "auc_share_times_cv": auc([(r["top_share"] * r["top_cv"], r["tis_over_uniform"] > 1) for r, _ in pooled]),
        "per_generator": per_generator,
    }
    out = ROOT / "results" / "llm_floor_diagnostic.json"
    out.write_text(json.dumps(result, indent=1))
    print("pooled=%d failures=%d AUC(top_cv)=%.3f AUC(share)=%.3f" % (
        result["pooled_questions"], result["pooled_failures"], result["auc_top_cv"], result["auc_top_share"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
