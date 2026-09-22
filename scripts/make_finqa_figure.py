#!/usr/bin/env python3
"""FinQA held-out figure: (a,b) panel MSE relative to uniform versus
budget for each generator (workflows averaged; per-workflow values are in
the tables), (c) wrong-workflow-selection rate versus budget on the Phi
near-tie. Budgets 400/800 come from the extension runs. Reads results/finqa_paper_numbers.json,
the *_b48 panel replications, and results/finqa_decision_extension.json;
writes figures/fig_finqa.pdf (+ .png render). Styling matches
scripts/make_figures.py (fig_llm): same font scale, bold left panel
letters, bottom shared legend, dotted gray uniform reference.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

ROOT = Path(__file__).resolve().parent.parent
report = json.loads((ROOT / "results" / "finqa_paper_numbers.json").read_text())

TICKS = 18
LABEL = 20
LEGEND = 18
TITLE = 22

BUDGETS = [25, 50, 100, 200]
EXT_BUDGETS = [400, 800]
ALL_BUDGETS = BUDGETS + EXT_BUDGETS
EXT_RUNS = {("qwen", "ordinary"): "finqa_heldout_ordinary_b48",
            ("qwen", "unitcheck"): "finqa_heldout_unitcheck_b48",
            ("phi", "ordinary"): "finqa_heldout_ordinary_phi_b48",
            ("phi", "unitcheck"): "finqa_heldout_unitcheck_phi_b48"}


def ext_normalized(gen: str, wf: str) -> dict[int, dict[str, float]]:
    """Extension budgets: panel MSE normalized to uniform, per workflow."""
    sums: dict[tuple[int, str], list[float]] = defaultdict(list)
    path = ROOT / "results" / EXT_RUNS[(gen, wf)] / "panel_replications.csv"
    for r in csv.DictReader(path.open()):
        sums[(int(r["budget_per_group"]), r["method"])].append(float(r["panel_mse"]))
    out: dict[int, dict[str, float]] = {}
    for b in EXT_BUDGETS:
        uni = sum(sums[(b, "uniform")]) / len(sums[(b, "uniform")])
        out[b] = {m: (sum(v) / len(v)) / uni
                  for (bb, m), v in sums.items() if bb == b}
    return out


EXT_NORM = {(gen, wf): ext_normalized(gen, wf)
            for gen in ("qwen", "phi") for wf in ("ordinary", "unitcheck")}
EXT_DECISION = json.loads(
    (ROOT / "results" / "finqa_decision_extension.json").read_text())
# (key, label, color, linestyle, linewidth) -- anchored TIS is the paper's
# method and carries the visual weight; oracle is the population diagnostic.
METHODS = [
    ("tis", "plain TIS", "#666666", "--", 2.0),
    ("learned_occupancy", "learned occupancy", "#2ca02c", "-", 2.0),
    ("complete_rollout", "complete rollout", "#ff7f0e", "-", 2.0),
    ("oracle_tail", "oracle + floor", "#000000", ":", 2.0),
    ("tis_anchored", "anchored TIS", "#1f4e9c", "-", 3.2),
]

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))

for ax, gen, letter in ((axes[0], "qwen", "(a) Qwen3-4B"),
                        (axes[1], "phi", "(b) Phi-4-mini")):
    for method, label, color, style, lw in METHODS:
        ys = []
        for b in BUDGETS:
            vals = [
                report["panel"][f"{gen}|{wf}"][str(b)][method]["normalized_to_uniform"]
                for wf in ("ordinary", "unitcheck")
            ]
            ys.append(sum(vals) / len(vals))
        for b in EXT_BUDGETS:
            vals = [EXT_NORM[(gen, wf)][b][method]
                    for wf in ("ordinary", "unitcheck")]
            ys.append(sum(vals) / len(vals))
        ax.plot(ALL_BUDGETS, ys, style, color=color, label=label, marker="o",
                ms=8 if lw < 3 else 9, linewidth=lw)
    ax.axhline(1.0, color="gray", lw=1.2, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ALL_BUDGETS))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([str(b) for b in ALL_BUDGETS])
    ax.tick_params(labelsize=TICKS)
    ax.set_xlabel("queries per kernel", fontsize=LABEL)
    ax.set_title(letter, fontsize=TITLE, fontweight="bold", loc="left")
axes[0].set_ylabel("MSE / uniform MSE", fontsize=LABEL)

ax = axes[2]
decision = report["workflow_decision"]["phi"]
ext_dec = EXT_DECISION["phi"]
for method, label, color, style, lw in METHODS:
    xs, ys = [], []
    for row in decision["rows"]:
        if row["method"] == method and "wrong_rate" in row:
            xs.append(row["budget"])
            ys.append(100 * row["wrong_rate"])
    for b in EXT_BUDGETS:
        xs.append(b)
        ys.append(100 * ext_dec[str(b)][method])
    if xs:
        ax.plot(xs, ys, style, color=color, label=label, marker="o",
                ms=8 if lw < 3 else 9, linewidth=lw)
xs, ys = [], []
for row in decision["rows"]:
    if row["method"] == "uniform":
        xs.append(row["budget"])
        ys.append(100 * row["wrong_rate"])
for b in EXT_BUDGETS:
    xs.append(b)
    ys.append(100 * ext_dec[str(b)]["uniform"])
ax.plot(xs, ys, "-", color="#d62728", label="uniform", marker="o", ms=8,
        linewidth=2.0)
ax.set_xscale("log")
ax.xaxis.set_major_locator(FixedLocator(ALL_BUDGETS))
ax.xaxis.set_minor_locator(FixedLocator([]))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xticklabels([str(b) for b in ALL_BUDGETS])
ax.tick_params(labelsize=TICKS)
ax.set_xlabel("queries per kernel", fontsize=LABEL)
ax.set_ylabel("wrong workflow selected (%)", fontsize=LABEL)
ax.set_title("(c) workflow choice, Phi", fontsize=TITLE, fontweight="bold",
             loc="left")

handles, labels = ax.get_legend_handles_labels()
order = [labels.index(l) for l in
         ["anchored TIS", "learned occupancy", "complete rollout",
          "plain TIS", "oracle + floor", "uniform"]]
fig.legend([handles[i] for i in order], [labels[i] for i in order],
           loc="lower center", ncol=6, fontsize=LEGEND, frameon=False,
           bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=(0, 0.08, 1, 1))
out = ROOT / "figures" / "fig_finqa.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=110, bbox_inches="tight")
print(out)
