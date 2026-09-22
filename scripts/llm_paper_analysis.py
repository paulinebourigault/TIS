#!/usr/bin/env python3
"""Regenerate every language-model-study number and LaTeX table in the paper.

Reads the released result directories and frozen artifacts, writes
``build/llm_paper_numbers.json`` (per-generator summaries, per-question
medians, allocation mechanism, floor sensitivity, and the variance/bias
diagnostics used for the GLM-4-32B analysis) and ``build/llm_paper_tables.tex``
(the main-text table and the five supplement tables). Run from the
repository root after the study directories exist:

    python scripts/llm_paper_analysis.py [--closed]

With ``--closed`` the script reads the closed-grid rerun directories
(``results/llm_study_closed*`` and ``results/llm_floor_sensitivity_closed*``)
and writes ``llm_paper_numbers_closed.json`` / ``llm_paper_tables_closed.tex``.

Per-replication ``raw.csv`` files are needed only for the realized design
constants; when absent (they are not committed to GitHub because of size),
that diagnostic is skipped and everything else is produced.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLOSED = "--closed" in sys.argv[1:]
_DIR = "llm_study_closed" if CLOSED else "llm_study_paper"
FLOOR_BASE = "llm_floor_sensitivity_closed" if CLOSED else "llm_floor_sensitivity"
SUFFIX_OUT = "_closed" if CLOSED else ""
GENERATORS = [
    ("Qwen3-4B", _DIR, "generation_audit.json", "qwen_kernels.json", "qwen4b"),
    ("Phi-4-mini", _DIR + "_robustness", "generation_audit_robustness.json", "robustness_kernels.json", "phi4mini"),
    ("Granite-4.2-8B", _DIR + "_robustness2", "generation_audit_robustness2.json", "robustness2_kernels.json", "granite8b"),
    ("Mistral-24B", _DIR + "_scale1", "generation_audit_scale1.json", "scale1_kernels.json", "mistral24b"),
    ("Qwen3-32B", _DIR + "_scale2", "generation_audit_scale2.json", "scale2_kernels.json", "qwen32b"),
    ("GLM-4-32B", _DIR + "_scale3", "generation_audit_scale3.json", "scale3_kernels.json", "glm32b"),
]
LABELS = {
    "Qwen3-4B": "Qwen3-4B (primary)", "Phi-4-mini": "Phi-4-mini", "Granite-4.2-8B": "Granite-4.2-8B",
    "Mistral-24B": "Mistral-24B$^\\dagger$", "Qwen3-32B": "Qwen3-32B$^\\dagger$", "GLM-4-32B": "GLM-4-32B$^\\dagger$",
}
SHORT = {
    "Qwen3-4B": "Qwen3-4B", "Phi-4-mini": "Phi-4-mini", "Granite-4.2-8B": "Granite-8B",
    "Mistral-24B": "Mistral-24B$^\\dagger$", "Qwen3-32B": "Qwen3-32B$^\\dagger$", "GLM-4-32B": "GLM-4-32B$^\\dagger$",
}
METHODS = [
    ("occupancy", "exact occupancy (adv.)"), ("mean_influence", "exact mean influence (adv.)"),
    ("answer_entropy", "pilot answer entropy"), ("oracle_tail", "oracle tail + floor (adv.)"),
    ("tis", "\\textsc{TIS}"), ("mc_ucb_frozen", "MC-UCB (frozen score)"),
    ("adaptive_neyman", "adaptive plug-in Neyman"), ("untied_tis", "layer-untied \\textsc{TIS}"),
]
BUCKETS = [
    ("root", "root \\texttt{solve}", 1), ("correct_low", "correct, conf.\\ .1--.3", 3),
    ("correct_mid", "correct, conf.\\ .4--.6", 3), ("correct_high", "correct, conf.\\ .7--.9", 3),
    ("wrong_low", "wrong, conf.\\ .1--.3", 27), ("wrong_mid", "wrong, conf.\\ .4--.6", 27),
    ("wrong_high", "wrong, conf.\\ .7--.9", 27),
]
HORIZONS = ("2", "4", "6")
BUDGETS = ("50", "100", "200", "400")


def f2(x: float) -> str:
    return ("%.2f" % x).lstrip("0") if x < 1 else "%.2f" % x


def f3(x: float) -> str:
    return ("%.3f" % x).lstrip("0") if x < 1 else "%.2f" % x


def bucket(state: int, correct: int) -> str:
    if state == 0:
        return "root"
    answer, confidence = divmod(state - 1, 9)
    band = "low" if confidence <= 2 else ("mid" if confidence <= 5 else "high")
    return ("correct" if answer == correct else "wrong") + "_" + band


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def at_400(row: dict[str, str], alpha: str, horizon: str | None = None) -> bool:
    return (
        row["alpha"] == alpha
        and row["budget_per_shared_group"] == "400"
        and (horizon is None or row["horizon"] == horizon)
    )


def summarize_generator(label: str, directory: str, audit_file: str, floor_dir: str, panel_correct: dict[int, int]) -> dict:
    base = ROOT / "results" / directory
    summary = {}
    for r in read_rows(base / "panel_summary.csv"):
        key = (r["horizon"], r["alpha"], r["budget_per_shared_group"], r["method"])
        summary[key] = (float(r["panel_mse"]), float(r["mcse"]), float(r["normalized_to_uniform"]) if r["normalized_to_uniform"] else None)
    paired = {}
    for r in read_rows(base / "paired_panel.csv"):
        paired[(r["horizon"], r["alpha"], r["budget_per_shared_group"], r["comparison"])] = (float(r["mean_difference"]), float(r["se"]))
    diagnostics = json.loads((base / "diagnostics.json").read_text())
    ratios = {a: [d["uniform_to_oracle"] for d in diagnostics if d["alpha"] == a and d["uniform_to_oracle"]] for a in (0.1, 0.2)}
    audit = json.loads((ROOT / "frozen" / audit_file).read_text())
    per_question = defaultdict(lambda: defaultdict(dict))
    for r in read_rows(base / "question_summary.csv"):
        if r["alpha"] == "0.1" and r["budget_per_shared_group"] == "400" and r["method"] in ("tis", "uniform"):
            per_question[r["horizon"]][int(r["panel_index"])][r["method"]] = (float(r["mse"]), float(r["bias"]))
    medians = {}
    bias_share = {}
    for h in HORIZONS:
        live = [i for i in per_question[h] if per_question[h][i]["uniform"][0] > 0]
        rs = sorted(per_question[h][i]["tis"][0] / per_question[h][i]["uniform"][0] for i in live)
        medians[h] = {"median_ratio": rs[len(rs) // 2], "q90_ratio": rs[int(0.9 * len(rs))], "frac_worse": sum(1 for x in rs if x > 1) / len(rs), "n": len(rs)}
        for method in ("tis", "uniform"):
            b2 = sum(per_question[h][i][method][1] ** 2 for i in per_question[h])
            mse = sum(per_question[h][i][method][0] for i in per_question[h])
            bias_share[f"{h}|{method}"] = b2 / mse if mse > 0 else None
    alloc = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    allocation_path = base / "allocation_summary.csv"
    # allocation_summary.csv exceeds GitHub size limits and lives on the cluster;
    # the mechanism table is produced only where it is present.
    with (allocation_path if allocation_path.exists() else Path("/dev/null")).open(newline="", encoding="utf-8") as stream:
        for r in csv.DictReader(stream):
            if not at_400(r, "0.1", "4") or r["method"] not in ("tis", "oracle_tail"):
                continue
            b = bucket(int(r["group_first"]), panel_correct[int(r["panel_index"])])
            alloc[r["method"]][b] += float(r["mean_weight"]) * 91
            if r["method"] == "tis":
                counts[b] += 1
    mechanism = {b: {"tis": alloc["tis"][b] / counts[b], "oracle": alloc["oracle_tail"][b] / counts[b], "groups_per_q": counts[b] // 50} for b in counts} or None
    design = {}
    raw = base / "raw.csv"
    if raw.exists():
        acc = defaultdict(lambda: [0.0, 0])
        with raw.open(newline="", encoding="utf-8") as stream:
            for r in csv.DictReader(stream):
                if at_400(r, "0.1") and r["method"] in ("tis", "uniform"):
                    key = (r["horizon"], r["method"])
                    acc[key][0] += float(r["design_constant"])
                    acc[key][1] += 1
        for h in HORIZONS:
            design[h] = (acc[(h, "tis")][0] / acc[(h, "tis")][1]) / (acc[(h, "uniform")][0] / acc[(h, "uniform")][1])
    floor = None
    fdir = ROOT / "results" / FLOOR_BASE / floor_dir
    grid_dirs = {
        "0.15": ROOT / "results" / (FLOOR_BASE + "_l15") / floor_dir,
        "0.40": ROOT / "results" / (FLOOR_BASE + "_l40") / floor_dir,
    }
    if CLOSED:
        grid_dirs["0.60"] = ROOT / "results" / (FLOOR_BASE + "_l60") / floor_dir
        grid_dirs["0.80"] = ROOT / "results" / (FLOOR_BASE + "_l80") / floor_dir
    if fdir.exists():
        def stats(d: Path, filt) -> dict:
            rows = {r["method"]: float(r["panel_mse"]) for r in read_rows(d / "panel_summary.csv") if filt(r)}
            q = defaultdict(dict)
            for r in read_rows(d / "question_summary.csv"):
                if filt(r):
                    q[int(r["panel_index"])][r["method"]] = float(r["mse"])
            live = [i for i in q if q[i]["uniform"] > 0]
            rs = sorted(q[i]["tis"] / q[i]["uniform"] for i in live)
            z = [float(r["mean_difference"]) / float(r["se"]) for r in read_rows(d / "paired_panel.csv") if filt(r) and r["comparison"] == "uniform-minus-tis"][0]
            return {"mean_ratio": rows["tis"] / rows["uniform"], "median_ratio": rs[len(rs) // 2], "frac_worse": sum(1 for x in rs if x > 1) / len(rs), "z": z}
        six = lambda r: r["horizon"] == "6" and at_400(r, "0.1")
        floor = {"prespecified": stats(base, six), "floor25": stats(fdir, six)}
        for lam, d in grid_dirs.items():
            if d.exists():
                floor[f"floor{lam}"] = stats(d, six)
    return {
        "status": json.loads((base / "manifest.json").read_text())["result_evidence_status"],
        "norm400": {f"{h}|{a}|{m}": summary[(h, a, "400", m)][2] for h in HORIZONS for a in ("0.1", "0.2") for m, _ in METHODS},
        "tis_abs400": {f"{h}|{a}": summary[(h, a, "400", "tis")][:2] for h in HORIZONS for a in ("0.1", "0.2")},
        "tis_budget_a01": {f"{h}|{b}": summary[(h, "0.1", b, "tis")][2] for h in HORIZONS for b in BUDGETS},
        "untied_budget_a01": {f"{h}|{b}": summary[(h, "0.1", b, "untied_tis")][2] for h in HORIZONS for b in BUDGETS},
        "z_uniform_minus_tis": {f"{h}|{a}": paired[(h, a, "400", "uniform-minus-tis")][0] / paired[(h, a, "400", "uniform-minus-tis")][1] for h in HORIZONS for a in ("0.1", "0.2")},
        "headroom": {str(a): (statistics.median(v), min(v), max(v)) for a, v in ratios.items()},
        "margins_all_positive": all(d["positive_margin"] for d in diagnostics),
        "audit_passed": audit["passed"],
        "audit_gaps": sum(1 for r in audit["records"] if r.get("verdict") == "precision_gap"),
        "per_question": medians,
        "bias_share_of_mse": bias_share,
        "design_constant_ratio_tis_over_uniform": design,
        "mechanism": mechanism,
        "floor_sensitivity": floor,
    }


def main_table(d: dict) -> str:
    rows = []
    for m, *_ in GENERATORS:
        v = d[m]
        cells = [f2(v["norm400"][f"{h}|0.1|tis"]) for h in HORIZONS] + [f2(v["norm400"][f"{h}|0.1|untied_tis"]) for h in HORIZONS] + ["%.1f" % v["z_uniform_minus_tis"][f"{h}|0.1"] for h in HORIZONS]
        rows.append(f"{LABELS[m]} & " + " & ".join(cells) + r"\\")
    return "\n".join(rows)


def table_full(d: dict) -> str:
    out = []
    for h in HORIZONS:
        out.append("\\midrule\n\\multicolumn{7}{l}{$H=" + h + "$}\\\\")
        for key, name in METHODS:
            out.append(f"{name} & " + " & ".join(f3(d[m]["norm400"][f"{h}|0.1|{key}"]) for m, *_ in GENERATORS) + r"\\")
        out.append(r"\textsc{TIS} MSE (MCSE) & " + " & ".join("%.1e (%.0e)" % tuple(d[m]["tis_abs400"][f"{h}|0.1"]) for m, *_ in GENERATORS) + r"\\")
    return "\n".join(out)


def table_budget(d: dict) -> str:
    out = []
    for m, *_ in GENERATORS:
        for h in HORIZONS:
            t = [f3(d[m]["tis_budget_a01"][f"{h}|{b}"]) for b in BUDGETS]
            u = [f2(d[m]["untied_budget_a01"][f"{h}|{b}"]) for b in BUDGETS]
            out.append(f"{LABELS[m] if h == '2' else ''} & $H={h}$ & " + " & ".join(t) + " & " + " & ".join(u) + r"\\")
        out.append(r"\midrule")
    return "\n".join(out[:-1])


def table_alpha(d: dict) -> str:
    out = []
    for m, *_ in GENERATORS:
        for h in HORIZONS:
            v = d[m]
            cells = [f3(v["norm400"][f"{h}|0.2|{k}"]) for k in ("tis", "untied_tis", "mc_ucb_frozen", "oracle_tail")]
            out.append(f"{LABELS[m] if h == '2' else ''} & $H={h}$ & " + " & ".join(cells) + " & %.1f" % v["z_uniform_minus_tis"][f"{h}|0.2"] + r"\\")
        out.append(r"\midrule")
    return "\n".join(out[:-1])


def table_mechanism(d: dict) -> str:
    if any(d[m]["mechanism"] is None for m, *_ in GENERATORS if m in d):
        return "% allocation_summary.csv not available here; run on the cluster copy"
    return "\n".join(
        f"{name} & {g} & " + " & ".join("%.1f/%.1f" % (d[m]["mechanism"][key]["tis"], d[m]["mechanism"][key]["oracle"]) for m, *_ in GENERATORS) + r"\\"
        for key, name, g in BUCKETS
    )


def table_floor(d: dict) -> str:
    rows = []
    for m, *_ in GENERATORS:
        fs = d[m]["floor_sensitivity"]
        if fs is None:
            rows.append(f"{LABELS[m]} & \\multicolumn{{8}}{{c}}{{not run}}\\\\")
            continue
        p, q = fs["prespecified"], fs["floor25"]
        rows.append(f"{LABELS[m]} & {f2(p['mean_ratio'])} & {f2(p['median_ratio'])} & {100 * p['frac_worse']:.0f} & {p['z']:.1f} & {f2(q['mean_ratio'])} & {f2(q['median_ratio'])} & {100 * q['frac_worse']:.0f} & {q['z']:.1f}\\\\")
    return "\n".join(rows)


def main() -> None:
    panel_correct = {}
    with (ROOT / "frozen" / "llm_panel.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            panel_correct[int(item["panel_index"])] = int(item["correct_index"])
    numbers = {}
    for label, directory, audit_file, _, floor_dir in GENERATORS:
        if (ROOT / "results" / directory / "manifest.json").exists():
            numbers[label] = summarize_generator(label, directory, audit_file, floor_dir, panel_correct)
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    (build / f"llm_paper_numbers{SUFFIX_OUT}.json").write_text(json.dumps(numbers, indent=1))
    heads = " & ".join(SHORT[m] for m, *_ in GENERATORS)
    tables = f"""% Generated by scripts/llm_paper_analysis.py -- do not edit by hand.
%% main-text table rows (tab:llm-suite)
{main_table(numbers)}

%% tab:supp-llm-full body (columns: {heads})
{table_full(numbers)}

%% tab:supp-llm-budget body
{table_budget(numbers)}

%% tab:supp-llm-alpha body
{table_alpha(numbers)}

%% tab:supp-llm-mechanism body (columns: {heads})
{table_mechanism(numbers)}

%% tab:supp-llm-floor body
{table_floor(numbers)}
"""
    (build / f"llm_paper_tables{SUFFIX_OUT}.tex").write_text(tables)
    for label in numbers:
        v = numbers[label]
        print("%-15s %s | TIS/uniform a=.1 b=400: %s | median q H6: %.2f | audit gaps %d" % (
            label, v["status"], "/".join(f2(v["norm400"][f"{h}|0.1|tis"]) for h in HORIZONS), v["per_question"]["6"]["median_ratio"], v["audit_gaps"]))
    print("wrote", build / f"llm_paper_numbers{SUFFIX_OUT}.json", "and", build / f"llm_paper_tables{SUFFIX_OUT}.tex")


if __name__ == "__main__":
    main()
