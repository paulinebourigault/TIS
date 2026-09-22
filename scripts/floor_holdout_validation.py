"""Held-out validation of exploration-floor selection.

The floor grid (lambda in {0.072, 0.15, 0.25, 0.40} at H=6, alpha=.1,
400 queries per kernel, identical seeds) was run after observing the
GLM-4-32B depth failure and is reported in full. This script asks
whether selecting a floor from data would have worked out of sample, using
only the committed per-question summaries; no new simulation is run.

Two prespecified splits:

1. Question-level 5-fold cross-validation within each generator. Folds are
   a deterministic seeded partition of the 50 panel questions. For each
   fold, the floor minimizing the panel MSE ratio (sum of TIS per-question
   MSEs over the development questions divided by the uniform sum) is
   selected on the other four folds and applied to the held-out fold. The
   pooled held-out ratio uses each question's fold-selected floor; its
   standard error is a leave-one-question-out jackknife.

2. Leave-one-generator-out. The floor minimizing the mean panel ratio
   across five generators is applied to the sixth.

Both are compared with the original schedule (lambda = N^{-1/4} = 0.072)
and with the hindsight-best fixed floor per generator.

Output: results/llm_floor_holdout.json (or llm_floor_holdout_closed.json
with --closed, which reads the closed-grid rerun directories).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CELL = {"horizon": "6", "alpha": "0.1", "budget_per_shared_group": "400"}
FOLDS = 5
FOLD_SEED = 20260907

CLOSED = "--closed" in sys.argv[1:]
_MAIN = "llm_study_closed" if CLOSED else "llm_study_paper"
_FLOOR = "llm_floor_sensitivity_closed" if CLOSED else "llm_floor_sensitivity"
GENERATORS = {
    "qwen4b": ("Qwen/Qwen3-4B-Instruct-2507", f"results/{_MAIN}"),
    "phi4mini": ("microsoft/Phi-4-mini-instruct", f"results/{_MAIN}_robustness"),
    "granite8b": ("ibm-granite/granite-4.2-8b", f"results/{_MAIN}_robustness2"),
    "mistral24b": ("mistralai/Mistral-Small-24B-Instruct-2501", f"results/{_MAIN}_scale1"),
    "qwen32b": ("Qwen/Qwen3-32B", f"results/{_MAIN}_scale2"),
    "glm32b": ("zai-org/GLM-4-32B-0414", f"results/{_MAIN}_scale3"),
}
LAMBDA_DIRS = {
    0.072: "{primary}",
    0.15: f"results/{_FLOOR}_l15/{{name}}",
    0.25: f"results/{_FLOOR}/{{name}}",
    0.40: f"results/{_FLOOR}_l40/{{name}}",
}
# Larger floors exist only for the closed-grid rerun; they are included in
# the grid (and the selection candidate set) when their directories exist.
if CLOSED:
    LAMBDA_DIRS[0.60] = "results/llm_floor_sensitivity_closed_l60/{name}"
    LAMBDA_DIRS[0.80] = "results/llm_floor_sensitivity_closed_l80/{name}"
ORIGINAL = 0.072


def cell_mse(directory: Path) -> tuple[dict[int, float], dict[int, float]]:
    tis: dict[int, float] = {}
    uniform: dict[int, float] = {}
    with (directory / "question_summary.csv").open() as handle:
        for row in csv.DictReader(handle):
            if any(row[key] != value for key, value in CELL.items()):
                continue
            index = int(row["panel_index"])
            if row["method"] == "tis":
                tis[index] = float(row["mse"])
            elif row["method"] == "uniform":
                uniform[index] = float(row["mse"])
    if sorted(tis) != sorted(uniform) or len(tis) != 50:
        raise ValueError(f"Incomplete cell in {directory}")
    return tis, uniform


def ratio(tis: dict[int, float], uniform: dict[int, float], questions) -> float:
    return sum(tis[q] for q in questions) / sum(uniform[q] for q in questions)


def main() -> None:
    data: dict[str, dict[float, tuple[dict[int, float], dict[int, float]]]] = {}
    for name, (identifier, primary) in GENERATORS.items():
        data[name] = {}
        for lam, pattern in LAMBDA_DIRS.items():
            directory = ROOT / pattern.format(primary=primary, name=name)
            if lam > 0.5 and not directory.exists():
                continue
            data[name][lam] = cell_mse(directory)
    shared_lambdas = sorted(set.intersection(*(set(d) for d in data.values())))
    for name in data:
        data[name] = {lam: data[name][lam] for lam in shared_lambdas}

    questions = sorted(data["qwen4b"][ORIGINAL][0])
    rng = np.random.default_rng(FOLD_SEED)
    order = rng.permutation(len(questions))
    fold_of = {questions[position]: int(rank % FOLDS) for rank, position in enumerate(order)}

    lambdas = sorted(data["qwen4b"])
    report: dict[str, object] = {
        "cell": {"horizon": 6, "alpha": 0.1, "budget_per_shared_group": 400},
        "lambdas": lambdas,
        "original_lambda": ORIGINAL,
        "folds": FOLDS,
        "fold_seed": FOLD_SEED,
        "generators": {},
    }

    full_ratio: dict[str, dict[float, float]] = {}
    for name in GENERATORS:
        full_ratio[name] = {
            lam: ratio(*data[name][lam], questions) for lam in lambdas
        }

    for name, (identifier, _) in GENERATORS.items():
        selected: dict[int, float] = {}
        fold_choices: dict[int, float] = {}
        for fold in range(FOLDS):
            development = [q for q in questions if fold_of[q] != fold]
            best = min(lambdas, key=lambda lam: ratio(*data[name][lam], development))
            fold_choices[fold] = best
            for q in questions:
                if fold_of[q] == fold:
                    selected[q] = best

        def pooled(subset) -> float:
            numerator = sum(data[name][selected[q]][0][q] for q in subset)
            denominator = sum(data[name][ORIGINAL][1][q] for q in subset)
            return numerator / denominator

        held_out = pooled(questions)
        jackknife = np.asarray(
            [pooled([q for q in questions if q != left_out]) for left_out in questions]
        )
        jackknife_se = float(
            np.sqrt((len(questions) - 1) * np.mean((jackknife - jackknife.mean()) ** 2))
        )
        report["generators"][name] = {
            "identifier": identifier,
            "full_data_ratio_by_lambda": full_ratio[name],
            "hindsight_best_lambda": min(lambdas, key=lambda lam: full_ratio[name][lam]),
            "hindsight_best_ratio": min(full_ratio[name].values()),
            "original_ratio": full_ratio[name][ORIGINAL],
            "cv_fold_choices": fold_choices,
            "cv_held_out_ratio": held_out,
            "cv_held_out_ratio_jackknife_se": jackknife_se,
        }

    loco: dict[str, object] = {}
    for name in GENERATORS:
        others = [other for other in GENERATORS if other != name]
        best = min(
            lambdas,
            key=lambda lam: float(np.mean([full_ratio[other][lam] for other in others])),
        )
        loco[name] = {
            "selected_lambda": best,
            "held_out_ratio": full_ratio[name][best],
            "original_ratio": full_ratio[name][ORIGINAL],
        }
    report["leave_one_generator_out"] = loco

    output = ROOT / "results" / ("llm_floor_holdout_closed.json" if CLOSED else "llm_floor_holdout.json")
    output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(output)
    for name, entry in report["generators"].items():
        print(
            f"{name}: original {entry['original_ratio']:.3f} | "
            f"CV held-out {entry['cv_held_out_ratio']:.3f} "
            f"(se {entry['cv_held_out_ratio_jackknife_se']:.3f}) | "
            f"hindsight {entry['hindsight_best_ratio']:.3f} "
            f"@ lambda={entry['hindsight_best_lambda']}"
        )
    for name, entry in loco.items():
        print(
            f"LOGO {name}: lambda={entry['selected_lambda']} -> "
            f"{entry['held_out_ratio']:.3f} (original {entry['original_ratio']:.3f})"
        )


if __name__ == "__main__":
    main()
