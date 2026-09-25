#!/usr/bin/env python3
"""Pilot-estimated tail-mean divergence as a selection rule.

For every blending setting on disk, recompute the runner's own pilots
(replications 0-4, same seed coordinates and pilot size), measure the
total-variation distance between normalized pilot tail and pilot mean
influence shares per question, and apply the declared rule: anchored TIS if
the median over questions is at least .21, otherwise the occupancy-mean
blend. The choice is scored against the recorded paired contrasts.

Usage: python scripts/pilot_divergence_analysis.py [processes]
Design: frozen/pilot_divergence_declaration.md.
Writes results/pilot_divergence_analysis.json.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tis.allocation import pilot_size  # noqa: E402
from tis.engine import compute_influences  # noqa: E402
from tis.finqa.models import finqa_question_model  # noqa: E402
from tis.finqa.toolfault import toolfault_question_model  # noqa: E402
from tis.llm_study.models import shared_question_model  # noqa: E402
from tis.llm_study.runner import _rng  # noqa: E402
from tis.model import model_from_counts, sample_counts  # noqa: E402

THRESHOLD = 0.21
REPS = range(5)
MMLU_KERNELS = {"qwen4b": "qwen", "phi": "robustness", "granite": "robustness2",
                "mistral24b": "scale1", "qwen32b": "scale2", "glm32b": "scale3",
                "hs_qwen4b": "hs_qwen", "hs_phi": "hs_robustness", "hs_qwen32b": "hs_scale2",
                "cautious_qwen4b": "cautious_qwen", "cautious_qwen4b_conf": "cautious_qwen"}


def settings() -> list[dict]:
    out = []
    for prefix in ("llm_brier_mix", "llm_confident_mix", "llm_divpred"):
        for gen, kern in MMLU_KERNELS.items():
            cfg = ROOT / f"configs/{prefix}_{gen}.json"
            if cfg.exists():
                out.append({"name": f"{prefix}_{gen}", "domain": "mmlu", "config": cfg,
                            "kernels": ROOT / f"frozen/{kern}_kernels.json"})
    for gen, kern in (("qwen4b", "qwen"), ("phi", "robustness"), ("glm32b", "scale3")):
        for h in (8, 10):
            cfg = ROOT / f"configs/llm_longh_{gen}_h{h}.json"
            out.append({"name": f"llm_longh_{gen}_h{h}", "domain": "mmlu", "config": cfg,
                        "kernels": ROOT / f"frozen/{kern}_kernels.json"})
    for wf in ("ordinary", "unitcheck"):
        for gen, tag in (("qwen", ""), ("phi", "_phi")):
            out.append({"name": f"finqa_heldout_{wf}{tag}_mix", "domain": "finqa",
                        "config": ROOT / f"configs/finqa_heldout_{wf}{tag}_mix.json",
                        "kernels": ROOT / f"frozen/finqa_kernels_heldout2{tag}_{wf}.json"})
            out.append({"name": f"finqa_longh_{gen}_{wf}", "domain": "finqa",
                        "config": ROOT / f"configs/finqa_longh_{gen}_{wf}.json",
                        "kernels": ROOT / f"frozen/finqa_kernels_heldout2{tag}_{wf}.json"})
    for wf in ("", "unitcheck_"):
        for gen in ("qwen", "phi"):
            for p in ("0.01", "0.03", "0.1"):
                out.append({"name": f"finqa_toolfault_{wf}{gen}_p{p}", "domain": "finqa",
                            "config": ROOT / f"configs/finqa_toolfault_{wf}{gen}_p{p}.json",
                            "kernels": ROOT / f"frozen/finqa_kernels_heldout2_toolfault_{wf}{gen}.json"})
    for s in out:
        cfg = json.loads(s["config"].read_text())
        s["results"] = ROOT / cfg["output_directory"] / "panel_replications.csv"
    return [s for s in out if s["results"].exists()]


def _tv(tail: np.ndarray, mean: np.ndarray) -> float | None:
    if tail.sum() <= 0 or mean.sum() <= 0:
        return None
    return float(0.5 * np.abs(tail / tail.sum() - mean / mean.sum()).sum())


def question_task(args) -> list[tuple]:
    setting, question, banks = args
    cfg = json.loads(Path(setting["config"]).read_text())
    rows = []
    if setting["domain"] == "mmlu":
        alpha = float(cfg["alphas"][0])
        budgets = [int(b) for b in cfg["budgets_per_shared_group"]]
        for horizon in [int(h) for h in cfg["horizons"]]:
            model = shared_question_model(question, horizon,
                                          closed_grid=bool(cfg.get("closed_grid", False)),
                                          utility=str(cfg.get("utility", "brier")),
                                          policy=str(cfg.get("policy", "default")))
            truth = compute_influences(model, alpha)
            pop = _tv(np.asarray(truth.tail_scales), np.asarray(truth.mean_scales))
            for b in budgets:
                total = model.group_count * b
                m = pilot_size(total, model.group_count, float(cfg["pilot_multiplier"]),
                               int(cfg["minimum_pilot_per_group"]), int(cfg["minimum_main_per_group"]))
                for r in REPS:
                    counts = sample_counts(model, {g: m for g in model.query_groups or ()},
                                           _rng(int(cfg["master_seed"]), horizon, 0, total,
                                                int(question["panel_index"]), r, 0))
                    pb = compute_influences(model_from_counts(model, counts), alpha)
                    rows.append((b, r, _tv(np.asarray(pb.tail_scales), np.asarray(pb.mean_scales)), pop))
    else:
        alpha = float(cfg["alpha"])
        horizon = int(cfg["horizon"])
        bank = banks[int(question["panel_index"])]
        if "tool_fault_probability" in cfg:
            model = toolfault_question_model(question, bank, horizon, float(cfg["tool_fault_probability"]))
        else:
            model = finqa_question_model(question, bank, horizon)
        truth = compute_influences(model, alpha)
        pop = _tv(np.asarray(truth.tail_scales), np.asarray(truth.mean_scales))
        for b in [int(v) for v in cfg["budgets_per_group"]]:
            total = model.group_count * b
            m = pilot_size(total, model.group_count, float(cfg["pilot_multiplier"]),
                           int(cfg["minimum_pilot_per_group"]), int(cfg["minimum_main_per_group"]))
            for r in REPS:
                counts = sample_counts(model, {g: m for g in model.query_groups or ()},
                                       _rng(int(cfg["master_seed"]), horizon, 0, total,
                                            int(question["panel_index"]), r, 0))
                pb = compute_influences(model_from_counts(model, counts), alpha)
                rows.append((b, r, _tv(np.asarray(pb.tail_scales), np.asarray(pb.mean_scales)), pop))
    return rows


def contrasts(path: Path) -> dict[int, dict]:
    data: dict[tuple[int, str], dict[int, float]] = defaultdict(dict)
    for r in csv.DictReader(path.open()):
        b = int(r.get("budget_per_shared_group") or r["budget_per_group"])
        v = float(r.get("panel_mean_squared_error") or r["panel_mse"])
        data[(b, r["method"])][int(r["replication"])] = v
    out = {}
    for b in sorted({k[0] for k in data}):
        a, m = data[(b, "tis_anchored")], data[(b, "occ_plus_mean")]
        if not a or not m:
            continue
        k = sorted(set(a) & set(m))
        d = np.array([a[i] - m[i] for i in k])
        se = d.std(ddof=1) / np.sqrt(len(d))
        out[b] = {"anchor": float(np.mean([a[i] for i in k])), "mean_blend": float(np.mean([m[i] for i in k])),
                  "z": float(d.mean() / se) if se > 0 else 0.0}
    return out


def main() -> None:
    processes = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    banks = {int(b["panel_index"]): b["candidates"] for b in
             json.loads((ROOT / "frozen/finqa_banks_heldout2.json").read_text())["banks"]}
    selected = settings()
    tasks, owners = [], []
    for s in selected:
        art = json.loads(Path(s["kernels"]).read_text())
        for q in art["questions"]:
            tasks.append(({k: str(v) for k, v in s.items()}, q, banks))
            owners.append(s["name"])
    per: dict[str, list] = defaultdict(list)
    with ProcessPoolExecutor(max_workers=processes) as pool:
        for name, rows in zip(owners, pool.map(question_task, tasks, chunksize=1)):
            per[name].extend(rows)
    report, bad, total, rel = {}, 0, 0, []
    for s in selected:
        rows = per[s["name"]]
        con = contrasts(s["results"])
        pops = [r[3] for r in rows if r[1] == 0 and r[3] is not None]
        entry = {"population_median": float(np.median(pops)) if pops else None, "budgets": {}}
        for b, c in con.items():
            meds = {}
            for r in REPS:
                vals = [x[2] for x in rows if x[0] == b and x[1] == r and x[2] is not None]
                meds[r] = float(np.median(vals)) if vals else 0.0
            choice = "anchor" if meds[0] >= THRESHOLD else "mean_blend"
            other = "mean_blend" if choice == "anchor" else "anchor"
            resolved_worse = (c["z"] >= 2) if choice == "anchor" else (c["z"] <= -2)
            best = min(c["anchor"], c["mean_blend"])
            stable = sum((meds[r] >= THRESHOLD) == (meds[0] >= THRESHOLD) for r in REPS if r > 0)
            entry["budgets"][str(b)] = {"pilot_median_rep0": meds[0], "pilot_medians": meds,
                                        "choice": choice, "bad_choice": bool(resolved_worse),
                                        "chosen_over_best": c[choice] / best, "z_anchor_vs_mean_blend": c["z"],
                                        "replications_agreeing_of_4": stable}
            total += 1; bad += int(resolved_worse); rel.append(c[choice] / best)
            print(f"{s['name']:38s} b{b:<4d} pop {entry['population_median'] or 0:.3f} pilot {meds[0]:.3f} "
                  f"-> {choice:10s} z {c['z']:+5.1f} {'BAD' if resolved_worse else 'ok '} "
                  f"chosen/best {c[choice] / best:.3f} stable {stable}/4", flush=True)
        report[s["name"]] = entry
    popl = [(v["population_median"], v["budgets"][min(v["budgets"], key=int)]["pilot_median_rep0"])
            for v in report.values() if v["population_median"] is not None and v["budgets"]]
    def _ranks(x):
        return np.argsort(np.argsort(np.asarray(x)))
    rho = (float(np.corrcoef(_ranks([a for a, _ in popl]), _ranks([b for _, b in popl]))[0, 1])
           if len(popl) > 2 else None)
    summary = {"threshold": THRESHOLD, "cells": total, "bad_choices": bad,
               "chosen_over_best_median": float(np.median(rel)), "chosen_over_best_max": float(np.max(rel)),
               "spearman_pilot_vs_population": rho}
    print(json.dumps(summary, indent=1))
    (ROOT / "results/pilot_divergence_analysis.json").write_text(
        json.dumps({"summary": summary, "settings": report}, indent=1) + "\n")


if __name__ == "__main__":
    main()
