#!/usr/bin/env python3
"""Generate the paper figures from the released result CSVs.

Writes figures/fig_main.pdf (three-panel tabular-suite MSE
curves with one shared legend) and figures/fig_supplement.pdf
(perturbed-instance ratios and the inventory pilot-sensitivity grid).
Panel titles are bare letters; the captions in the paper identify them.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"

# One style per method, consistent across every panel.
STYLE = {
    "uniform": ("tab:blue", "o", "uniform"),
    "reachability_only": ("tab:orange", "^", "reachability only"),
    "local_only": ("tab:brown", "v", "local scale only"),
    "occupancy": ("tab:purple", "P", "occupancy"),
    "mean_influence": ("tab:cyan", "X", "mean influence"),
    "mc_ucb_frozen": ("tab:pink", "*", "MC-UCB (frozen)"),
    "oracle_tail": ("tab:red", "s", "oracle + floor"),
    "tis": ("tab:green", "D", "TIS"),
}

TICKS = 18
LABEL = 20
LEGEND = 18
TITLE = 22


def read(path: Path, env: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [r for r in csv.DictReader(stream) if r["environment"] == env]


def panel(ax, rows, budget_field, methods, title, xlab, ticks):
    for key in methods:
        color, marker, _ = STYLE[key]
        pts = sorted(
            (float(r[budget_field]), float(r["mse"]), float(r["mse_se"]))
            for r in rows
            if r["method"] == key
        )
        ax.errorbar(
            [p[0] for p in pts],
            [p[1] for p in pts],
            yerr=[1.96 * p[2] for p in pts],
            color=color,
            marker=marker,
            ms=8,
            lw=2.0,
            capsize=3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=TITLE, fontweight="bold", loc="left")
    ax.set_xlabel(xlab, fontsize=LABEL)
    ax.tick_params(labelsize=TICKS)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([format(int(t), ",") for t in ticks])


# fig_main styling matches fig_finqa (scripts/make_finqa_figure.py) in
# colors; (key, label, color, linestyle, linewidth). In the tabular
# figure plain TIS carries the visual weight (it is the paper's theory
# and the strongest tabular method); the anchor keeps the emphasis in
# the FinQA figure, where it is the frozen prospective design.
MAIN_METHODS = [
    ("uniform", "uniform", "#d62728", "-", 2.0),
    ("learned_occupancy", "learned occupancy", "#2ca02c", "-", 2.0),
    ("learned_mean", "learned mean", "#9467bd", "-", 2.0),
    ("complete_rollout", "complete rollout", "#ff7f0e", "-", 2.0),
    ("tis", "plain TIS", "#444444", "-", 3.2),
    ("tis_anchored", "anchored TIS", "#1f4e9c", "--", 2.0),
    ("oracle_tail", "oracle + floor", "#000000", ":", 2.0),
]


def fig_main() -> None:
    import json

    data = json.loads((ROOT / "results" / "tabular_reconciled.json").read_text())
    key_alias = {"tis_anchored": "tail_anchored"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    tasks = [
        (axes[0], "cliffwalking", "(a)", "queries per kernel", [50, 100, 200, 400]),
        (axes[1], "inventory", "(b)", "queries per block", [150, 300, 600, 1200]),
    ]
    for ax, env, letter, xlab, ticks in tasks:
        cells = data[env]
        for key, label, color, style, lw in MAIN_METHODS:
            source = key_alias.get(key, key)
            pts = sorted(
                (float(k.split("|")[0]), cells[k]["mse"], cells[k]["mse_se"])
                for k in cells
                if k.split("|")[1] == source
            )
            ax.errorbar(
                [p[0] for p in pts],
                [p[1] for p in pts],
                yerr=[1.96 * p[2] for p in pts],
                color=color,
                ls=style,
                marker="o",
                ms=8 if lw < 3 else 9,
                lw=lw,
                capsize=3,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(letter, fontsize=TITLE, fontweight="bold", loc="left")
        ax.set_xlabel(xlab, fontsize=LABEL)
        ax.tick_params(labelsize=TICKS)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticklabels([format(int(t), ",") for t in ticks])
    axes[0].set_ylabel("categorical CVaR MSE", fontsize=LABEL)
    handles = [
        plt.Line2D([], [], color=c, ls=s, marker="o", ms=9, lw=w, label=lab)
        for _, lab, c, s, w in MAIN_METHODS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=LEGEND,
               frameon=False, bbox_to_anchor=(0.5, -0.02), columnspacing=1.4,
               handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(OUT / "fig_main.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "build" / "fig_main_preview.png", dpi=100,
                bbox_inches="tight")


def fig_controlled_mrp() -> None:
    """Appendix figure: controlled H=5 MRP, same method set and styling as
    fig_main, plus the two population-informed ablations of this section."""
    extras = [
        ("reachability_only", "reachability only", "#8c564b", "-.", 1.6),
        ("local_only", "local scale only", "#e377c2", "-.", 1.6),
    ]
    curves: dict[str, list[tuple[float, float, float]]] = {}
    for src in ("replacement_paper", "replacement_anchored",
                "replacement_meanlearned", "tabular_rollout"):
        path = ROOT / "results" / src / "summary.csv"
        for r in csv.DictReader(path.open()):
            if r["environment"] != "controlled_untied_v1":
                continue
            curves.setdefault(r["method"], {})[float(r["total_budget"])] = (
                float(r["mse"]), float(r["mse_se"]))
    key_alias = {"tis_anchored": "tail_anchored"}
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    for key, label, color, style, lw in MAIN_METHODS + extras:
        pts = sorted((b, v[0], v[1])
                     for b, v in curves[key_alias.get(key, key)].items())
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts],
                    yerr=[1.96 * p[2] for p in pts], color=color, ls=style,
                    marker="o", ms=7 if lw < 3 else 8, lw=lw, capsize=3,
                    label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("total queries", fontsize=LABEL)
    ax.set_ylabel("categorical CVaR MSE", fontsize=LABEL)
    ax.tick_params(labelsize=TICKS)
    ticks = [12500, 25000, 50000, 100000]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([format(t, ",") for t in ticks])
    fig.legend(fontsize=LEGEND - 3, frameon=False, ncol=3, columnspacing=1.0,
               handletextpad=0.5, labelspacing=0.4, loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(OUT / "fig_controlled_mrp.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "build" / "fig_controlled_mrp_preview.png",
                dpi=100, bbox_inches="tight")


def fig_supplement() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    panels_dir = ROOT / "results" / "robustness_panels"
    ratios = list(csv.DictReader((panels_dir / "perturbed_ratios.csv").open()))
    markers = {0.05: ("o", "tab:blue"), 0.1: ("s", "tab:red"), 0.2: ("^", "tab:green")}
    for alpha, (marker, color) in markers.items():
        xs = [int(r["seed"]) for r in ratios if float(r["alpha"]) == alpha]
        ys = [float(r["ratio"]) for r in ratios if float(r["alpha"]) == alpha]
        axes[0].scatter(xs, ys, s=42, marker=marker, color=color,
                        label=f"$\\alpha={alpha}$", alpha=0.8)
    axes[0].axhline(1.0, color="gray", lw=1.0, ls=":")
    axes[0].set_xlabel("instance seed", fontsize=LABEL)
    axes[0].set_ylabel("$V_{\\mathrm{unif}}/V^*$ (exact)", fontsize=LABEL)
    axes[0].set_title("(a)", fontsize=TITLE, fontweight="bold", loc="left")
    axes[0].legend(fontsize=LEGEND, frameon=False)
    axes[0].tick_params(labelsize=TICKS)
    grid = list(csv.DictReader((panels_dir / "pilot_grid.csv").open()))
    styles = {"1/6": (":", "tab:blue"), "1/4": ("-", "tab:red"), "1/3": ("--", "tab:green")}
    for floor, (ls, color) in styles.items():
        rows = sorted(
            (float(r["pilot_fraction"]), float(r["median"]), float(r["q90"]))
            for r in grid if r["floor"] == floor
        )
        fr = [100 * a for a, _, _ in rows]
        axes[1].plot(fr, [b for _, b, _ in rows], ls, color=color, marker="o",
                     ms=7, lw=2.0, label=f"$\\lambda=N^{{-{floor}}}$")
        axes[1].plot(fr, [c for _, _, c in rows], ls, color=color, marker="x",
                     ms=9, lw=1.2, alpha=0.55)
    axes[1].axhline(1.0, color="gray", lw=1.0, ls=":")
    axes[1].set_xlabel("pilot fraction (%)", fontsize=LABEL)
    axes[1].set_ylabel("$V(\\widehat w)/V^*$", fontsize=LABEL)
    axes[1].set_title("(b)", fontsize=TITLE, fontweight="bold", loc="left")
    axes[1].legend(fontsize=LEGEND, frameon=False, title="medians (x: 90th pct.)",
                   title_fontsize=LEGEND - 1)
    axes[1].tick_params(labelsize=TICKS)
    fig.tight_layout()
    fig.savefig(OUT / "fig_supplement.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "build" / "fig_supplement_preview.png",
                dpi=100, bbox_inches="tight")




def fig_llm() -> None:
    import json

    # Closed-grid (exact-target) numbers are canonical when present.
    numbers_path = ROOT / "results" / "llm_paper_numbers_closed.json"
    if not numbers_path.exists():
        numbers_path = ROOT / "build" / "llm_paper_numbers.json"
    if not numbers_path.exists():
        numbers_path = ROOT / "results" / "llm_paper_numbers.json"
    d = json.loads(numbers_path.read_text())
    anchored_path = ROOT / "results" / "llm_anchored_budgets.json"
    anchored = json.loads(anchored_path.read_text()) if anchored_path.exists() else {}
    generators = ["Qwen3-4B", "Phi-4-mini", "Granite-4.2-8B", "Mistral-24B",
                  "Qwen3-32B", "GLM-4-32B"]
    colors = {"Qwen3-4B": "tab:blue", "Phi-4-mini": "tab:orange",
              "Granite-4.2-8B": "tab:green", "Mistral-24B": "tab:red",
              "Qwen3-32B": "tab:purple", "GLM-4-32B": "tab:brown"}
    markers = {"Qwen3-4B": "o", "Phi-4-mini": "^", "Granite-4.2-8B": "s",
               "Mistral-24B": "D", "Qwen3-32B": "v", "GLM-4-32B": "P"}
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))

    budgets = [50, 100, 200, 400]
    for g in generators:
        ys = [d[g]["tis_budget_a01"][f"6|{b}"] for b in budgets]
        axes[0].plot(budgets, ys, color=colors[g], marker=markers[g], ms=8,
                     lw=2.0)
        if g in anchored:
            ya = [anchored[g][str(b)] for b in budgets]
            axes[0].plot(budgets, ya, color=colors[g], marker=markers[g],
                         ms=6, lw=1.6, ls="--", alpha=0.85)
    axes[0].axhline(1.0, color="gray", lw=1.2, ls=":")
    if anchored:
        from matplotlib.lines import Line2D
        style_handles = [
            Line2D([0], [0], color="black", lw=2.0, label="TIS"),
            Line2D([0], [0], color="black", lw=1.6, ls="--", label="anchored TIS"),
        ]
        axes[0].legend(handles=style_handles, fontsize=LEGEND - 2,
                       frameon=False, loc="lower left", borderaxespad=0.3,
                       handletextpad=0.5, labelspacing=0.3)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].xaxis.set_major_locator(FixedLocator(budgets))
    axes[0].xaxis.set_minor_formatter(NullFormatter())
    axes[0].set_xticklabels([str(b) for b in budgets])
    axes[0].set_ylim(bottom=0.025)
    axes[0].set_xlabel("queries per kernel", fontsize=LABEL)
    axes[0].set_ylabel("MSE / uniform MSE", fontsize=LABEL)
    axes[0].set_title("(a)", fontsize=TITLE, fontweight="bold", loc="left")
    axes[0].tick_params(labelsize=TICKS)

    buckets = [("root", "root"), ("correct_low", "corr .1--.3"),
               ("correct_mid", "corr .4--.6"), ("correct_high", "corr .7--.9"),
               ("wrong_low", "wrong .1--.3"), ("wrong_mid", "wrong .4--.6"),
               ("wrong_high", "wrong .7--.9")]
    positions = range(len(buckets))
    for g in generators:
        mech = d[g].get("mechanism") or {}
        if not mech:
            continue
        ys = [mech[k]["tis"] for k, _ in buckets]
        axes[1].plot(positions, ys, color=colors[g], marker=markers[g], ms=8,
                     lw=1.6, alpha=0.9)
    axes[1].axhline(1.0, color="gray", lw=1.2, ls=":")
    axes[1].set_yscale("log")
    axes[1].set_xticks(list(positions))
    axes[1].set_xticklabels([lab.replace("--", "-") for _, lab in buckets],
                            rotation=38, ha="right")
    axes[1].set_ylabel("weight / uniform share", fontsize=LABEL)
    axes[1].set_title("(b)", fontsize=TITLE, fontweight="bold", loc="left")
    axes[1].tick_params(labelsize=TICKS - 1)

    floor_keys = [("prespecified", "$\\lambda=.072$"), ("floor0.15", "$\\lambda=.15$"),
                  ("floor25", "$\\lambda=.25$"), ("floor0.40", "$\\lambda=.40$"),
                  ("floor0.60", "$\\lambda=.60$"), ("floor0.80", "$\\lambda=.80$")]
    floor_keys = [(k, lab) for k, lab in floor_keys
                  if all(k in (d[g].get("floor_sensitivity") or {}) for g in generators)]
    shades = ["0.65", "0.45", "tab:olive", "tab:green", "tab:cyan", "tab:blue"]
    x = range(len(generators))
    width = 0.8 / max(len(floor_keys), 1)
    for j, (key, lab) in enumerate(floor_keys):
        vals = [d[g]["floor_sensitivity"][key]["mean_ratio"] for g in generators]
        axes[2].bar([i + (j - (len(floor_keys) - 1) / 2) * width for i in x],
                    vals, width, label=lab, color=shades[j])
    axes[2].axhline(1.0, color="black", lw=1.2, ls=":")
    axes[2].set_yscale("log")
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(["Qwen3-4B", "Phi-4-mini", "Granite-8B",
                             "Mistral-24B", "Qwen3-32B", "GLM-4-32B"],
                            rotation=38, ha="right")
    axes[2].set_ylabel("TIS MSE / uniform MSE", fontsize=LABEL)
    axes[2].set_title("(c)", fontsize=TITLE, fontweight="bold", loc="left")
    axes[2].set_ylim(top=4.5)
    axes[2].legend(fontsize=LEGEND - 3, frameon=False, ncol=3,
                   loc="upper left", borderaxespad=0.2, columnspacing=0.7,
                   handletextpad=0.35, handlelength=1.1, labelspacing=0.35)
    axes[2].tick_params(labelsize=TICKS - 1)

    handles = [plt.Line2D([], [], color=colors[g], marker=markers[g], ms=9,
                          lw=2.0, label=g) for g in generators]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               fontsize=LEGEND - 1, frameon=False,
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.0,
               handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(OUT / "fig_llm.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "build" / "fig_llm_preview.png", dpi=100,
                bbox_inches="tight")



if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "build").mkdir(exist_ok=True)
    fig_main()
    fig_controlled_mrp()
    fig_supplement()
    if (ROOT / "build" / "llm_paper_numbers.json").exists() or (
        ROOT / "results" / "llm_paper_numbers.json").exists():
        fig_llm()
    print("figures written to", OUT)
