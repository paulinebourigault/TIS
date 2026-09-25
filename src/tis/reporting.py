from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarize(raw_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    grouped: dict[tuple[str, float, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        key = (
            str(row["environment"]),
            float(row["alpha"]),
            int(row["total_budget"]),
            str(row["method"]),
        )
        grouped[key].append(row)

    summaries: list[dict[str, object]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        errors = np.asarray([float(row["error"]) for row in rows])
        squared = errors**2
        estimates = np.asarray([float(row["estimate"]) for row in rows])
        design = np.asarray([float(row["design_constant"]) for row in rows])
        summaries.append(
            {
                "environment": key[0],
                "alpha": key[1],
                "budget_per_group": float(rows[0]["budget_per_group"]),
                "budget_unit": str(rows[0].get("budget_unit", "per_group")),
                "budget_value": int(rows[0].get("budget_value", rows[0]["budget_per_group"])),
                "total_budget": key[2],
                "method": key[3],
                "replications": len(rows),
                "truth": float(rows[0]["truth"]),
                "mse": float(squared.mean()),
                "mse_se": float(squared.std(ddof=1) / np.sqrt(len(rows))) if len(rows) > 1 else 0.0,
                "bias": float(errors.mean()),
                "estimate_sd": float(estimates.std(ddof=1)) if len(rows) > 1 else 0.0,
                "mean_design_constant": float(design.mean()),
                "pilot_fraction": float(rows[0]["pilot_total"]) / float(rows[0]["total_budget"]),
            }
        )

    by_rep: dict[tuple[str, float, int], dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    budget_metadata: dict[tuple[str, float, int], dict[str, object]] = {}
    for row in raw_rows:
        key = (
            str(row["environment"]),
            float(row["alpha"]),
            int(row["total_budget"]),
        )
        by_rep[key][int(row["replication"])][str(row["method"])] = float(row["squared_error"])
        budget_metadata[key] = row
    paired: list[dict[str, object]] = []
    for key in sorted(by_rep):
        replications = by_rep[key]
        methods = sorted(next(iter(replications.values())))
        if "tis" not in methods:
            continue
        for method in methods:
            if method == "tis":
                continue
            differences = np.asarray(
                [values[method] - values["tis"] for _, values in sorted(replications.items())]
            )
            se = float(differences.std(ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else 0.0
            paired.append(
                {
                    "environment": key[0],
                    "alpha": key[1],
                    "budget_per_group": float(budget_metadata[key]["budget_per_group"]),
                    "budget_unit": str(budget_metadata[key].get("budget_unit", "per_group")),
                    "budget_value": int(
                        budget_metadata[key].get(
                            "budget_value", budget_metadata[key]["budget_per_group"]
                        )
                    ),
                    "total_budget": key[2],
                    "comparison": f"{method}-minus-tis",
                    "replications": len(differences),
                    "mean_difference": float(differences.mean()),
                    "se": se,
                    "ci95_low": float(differences.mean() - 1.96 * se),
                    "ci95_high": float(differences.mean() + 1.96 * se),
                }
            )
    return summaries, paired


def write_latex_table(path: Path, summaries: list[dict[str, object]]) -> None:
    environments = sorted({str(row["environment"]) for row in summaries})
    preferred_methods = [
        "uniform",
        "reachability_only",
        "local_only",
        "occupancy",
        "mean_influence",
        "mc_ucb_frozen",
        "tis",
        "oracle_tail",
        "tail_no_cov",
        "learned_mean",
        "learned_occupancy",
        "tail_anchored",
        "oracle_no_cov",
    ]
    observed_methods = {str(row["method"]) for row in summaries}
    methods = [method for method in preferred_methods if method in observed_methods]
    largest = {
        env: max(int(row["total_budget"]) for row in summaries if row["environment"] == env)
        for env in environments
    }
    lookup = {
        (str(row["environment"]), int(row["total_budget"]), str(row["method"])): row
        for row in summaries
    }
    lines = [
        "% Generated from results/raw.csv; do not edit by hand.",
        "\\begin{tabular}{l" + "r" * len(environments) + "}",
        "\\toprule",
        "Method & " + " & ".join(env.replace("_", "\\_") for env in environments) + " \\\\",
        "\\midrule",
    ]
    for method in methods:
        cells = []
        for env in environments:
            row = lookup.get((env, largest[env], method))
            cells.append(
                "--" if row is None
                else f"{float(row['mse']):.4g} ({float(row['mse_se']):.2g})"
            )
        lines.append(method.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(path_stem: Path, summaries: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    environments = sorted({str(row["environment"]) for row in summaries})
    preferred_methods = [
        "uniform",
        "reachability_only",
        "local_only",
        "occupancy",
        "mean_influence",
        "mc_ucb_frozen",
        "tis",
        "oracle_tail",
        "tail_no_cov",
        "learned_mean",
        "learned_occupancy",
        "tail_anchored",
        "oracle_no_cov",
    ]
    observed_methods = {str(row["method"]) for row in summaries}
    methods = [method for method in preferred_methods if method in observed_methods]
    labels = {
        "uniform": "Uniform",
        "reachability_only": "Exact reachability only",
        "local_only": "Exact local scale only",
        "occupancy": "Exact occupancy",
        "mean_influence": "Exact mean influence",
        "mc_ucb_frozen": "MC-UCB (frozen score)",
        "learned_mean": "Learned mean influence",
        "tail_no_cov": "Tail influence, covariance dropped",
        "learned_occupancy": "Learned occupancy",
        "tail_anchored": "Anchored TIS",
        "tis": "TIS",
        "oracle_tail": "Oracle tail + floor",
        "oracle_no_cov": "Oracle tail, covariance dropped",
    }
    fig, axes = plt.subplots(1, len(environments), figsize=(5.2 * len(environments), 4.0), squeeze=False)
    for axis, env in zip(axes[0], environments):
        for method in methods:
            rows = sorted(
                [row for row in summaries if row["environment"] == env and row["method"] == method],
                key=lambda row: int(row["total_budget"]),
            )
            x = [int(row["budget_value"]) for row in rows]
            y = [float(row["mse"]) for row in rows]
            error = [1.96 * float(row["mse_se"]) for row in rows]
            axis.errorbar(x, y, yerr=error, marker="o", linewidth=1.5, capsize=2, label=labels[method])
        axis.set_yscale("log")
        units = {str(row["budget_unit"]) for row in summaries if row["environment"] == env}
        axis.set_xlabel(
            "Total queries" if units == {"total"} else "Queries per retained group"
        )
        axis.set_ylabel("Categorical-CVaR MSE")
        axis.set_title(env.replace("_", " "))
        axis.grid(True, which="both", alpha=0.25)
    axes[0, -1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), dpi=180)
    fig.savefig(path_stem.with_suffix(".pdf"))
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
