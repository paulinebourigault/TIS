"""Monte Carlo evaluation runner for the FinQA terminal-risk study.

Methods: uniform, learned occupancy, learned mean, plain TIS, anchored
TIS (fixed mu = 1/2), the two mixture controls (0.5u + 0.5w_occ and
0.5w_mean + 0.5w_occ, no second floor after mixing), oracle tail
(population diagnostic), and the complete-rollout empirical CVaR at
matched query cost. One workflow's kernel artifact per
run; the decision analysis joins the two workflows afterwards. Seeds,
coupling, pilot rule, floor, and rounding follow the primary study.
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..allocation import integer_allocation, pilot_size, regularized_weights
from ..engine import compute_influences, solve_stop_loss
from ..llm_study.common import file_sha256, object_sha256, read_json, write_json
from ..reporting import write_csv
from ..llm_study.runner import (
    _anchored_score,
    _complete_rollouts,
    _empirical_cvar,
    _mapping,
    _occupancy_scores,
    _rng,
)
from ..model import counts_from_streams, coupled_streams, model_from_counts, sample_counts
from . import HORIZON, STATES
from .models import finqa_question_model
from .toolfault import toolfault_question_model

METHODS = (
    "uniform",
    "learned_occupancy",
    "learned_mean",
    "tis",
    "tis_anchored",
    "occ_plus_uniform",
    "occ_plus_mean",
    "oracle_tail",
    "complete_rollout",
)
LEARNED = {"learned_occupancy", "learned_mean", "tis", "tis_anchored",
           "occ_plus_uniform", "occ_plus_mean"}
# Mixture controls: equal-weight blends of component weight vectors that
# already carry the exploration floor; no second floor after mixing.
MIXTURES = {"occ_plus_uniform", "occ_plus_mean"}


def run_finqa_study(
    config_path: Path,
    kernel_artifact_path: Path,
    banks_path: Path,
    output_directory: Path,
    overwrite: bool = False,
) -> Path:
    config = read_json(config_path)
    artifact = read_json(kernel_artifact_path)
    banks = {int(b["panel_index"]): b["candidates"] for b in read_json(banks_path)["banks"]}
    output = Path(output_directory)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Result directory already exists: {output}")
        import shutil

        shutil.rmtree(output)
    output.mkdir(parents=True)
    write_json(output / "config.json", config)

    methods = tuple(config.get("methods", METHODS))
    assert not set(methods) - set(METHODS)
    replications = int(config["replications"])
    alpha = float(config["alpha"])
    horizon = int(config.get("horizon", HORIZON))
    budgets = [int(b) for b in config["budgets_per_group"]]
    minimum_main = int(config["minimum_main_per_group"])
    master_seed = int(config["master_seed"])
    started = time.time()

    raw_rows: list[dict[str, object]] = []
    question_stats = defaultdict(lambda: {"n": 0, "sum": 0.0, "sum2": 0.0, "err": 0.0})
    panel_sums: dict[tuple, float] = defaultdict(float)
    diagnostics = []
    questions = artifact["questions"]
    for question in questions:
        qindex = int(question["panel_index"])
        bank = banks[qindex]
        if "tool_fault_probability" in config:
            model = toolfault_question_model(
                question, bank, horizon, float(config["tool_fault_probability"]))
        else:
            model = finqa_question_model(question, bank, horizon)
        truth = compute_influences(model, alpha)
        groups = model.group_count
        diagnostics.append(
            {
                "panel_index": qindex,
                "item_id": question["item_id"],
                "truth": truth.solution.cvar,
                "margin": truth.solution.margin,
                "positive_margin": truth.solution.margin > 0,
                "grid_atoms": len(model.grid),
                "uniform_to_oracle": (
                    (groups * float(np.sum(truth.tail_scales**2)))
                    / float(np.sum(truth.tail_scales)) ** 2
                    if float(np.sum(truth.tail_scales)) > 0
                    else None
                ),
                "candidate_utilities": model.metadata["candidate_utilities"],
            }
        )
        for budget in budgets:
            total = groups * budget
            exploration = total ** (-float(config["exploration_exponent"]))
            m = pilot_size(
                total, groups, float(config["pilot_multiplier"]),
                int(config["minimum_pilot_per_group"]), minimum_main,
            )
            main_total = total - groups * m
            for replication in range(replications):
                pilot_counts = sample_counts(
                    model, {g: m for g in model.query_groups or ()},
                    _rng(master_seed, horizon, 0, total, qindex, replication, 0),
                )
                pilot = compute_influences(model_from_counts(model, pilot_counts), alpha)
                occupancy_scores = _occupancy_scores(model, pilot)
                scores = {
                    "uniform": np.ones(groups),
                    "learned_occupancy": occupancy_scores,
                    "learned_mean": pilot.mean_scales,
                    "tis": pilot.tail_scales,
                    "tis_anchored": _anchored_score(pilot.tail_scales, occupancy_scores),
                    "oracle_tail": truth.tail_scales,
                }
                fixed = [meth for meth in methods
                         if meth in scores or meth in MIXTURES]
                counts_by = {}
                for meth in fixed:
                    if meth == "uniform":
                        weights = np.full(groups, 1.0 / groups)
                        budget_for = total
                    elif meth in MIXTURES:
                        w_occ = regularized_weights(occupancy_scores, exploration)
                        if meth == "occ_plus_uniform":
                            other = np.full(groups, 1.0 / groups)
                        else:
                            other = regularized_weights(pilot.mean_scales, exploration)
                        weights = 0.5 * other + 0.5 * w_occ
                        budget_for = main_total
                    else:
                        weights = regularized_weights(scores[meth], exploration)
                        budget_for = main_total if meth in LEARNED else total
                    counts_by[meth] = integer_allocation(budget_for, weights, minimum_main)
                maximum = np.max(np.vstack([counts_by[meth] for meth in fixed]), axis=0)
                streams = coupled_streams(
                    model, _mapping(model, maximum),
                    _rng(master_seed, horizon, 0, total, qindex, replication, 1),
                )
                rollout_estimate = None
                rollout_visits = None
                if "complete_rollout" in methods:
                    runs = total // horizon
                    returns, rollout_visits = _complete_rollouts(
                        model, horizon, runs,
                        _rng(master_seed, horizon, 0, total, qindex, replication, 6),
                    )
                    rollout_estimate = _empirical_cvar(returns, alpha)
                for meth in methods:
                    if meth == "complete_rollout":
                        estimate = rollout_estimate
                        counts = rollout_visits
                        pilot_total = 0
                    else:
                        empirical = counts_from_streams(
                            model, streams, _mapping(model, counts_by[meth])
                        )
                        estimate = solve_stop_loss(
                            model_from_counts(model, empirical), alpha
                        ).cvar
                        counts = counts_by[meth]
                        pilot_total = groups * m if meth in LEARNED else 0
                    error = estimate - truth.solution.cvar
                    raw_rows.append(
                        {
                            "panel_index": qindex,
                            "budget_per_group": budget,
                            "total_budget": total,
                            "replication": replication,
                            "method": meth,
                            "truth": truth.solution.cvar,
                            "estimate": estimate,
                            "error": error,
                            "squared_error": error**2,
                            "pilot_total": pilot_total,
                            "main_total": int(np.sum(counts)),
                        }
                    )
                    key = (budget, qindex, meth)
                    stats = question_stats[key]
                    stats["n"] += 1
                    stats["sum"] += error**2
                    stats["sum2"] += error**4
                    stats["err"] += error
                    panel_sums[(budget, replication, meth)] += error**2 / len(questions)

    write_csv(output / "raw.csv", raw_rows)
    question_rows = []
    for (budget, qindex, meth), stats in sorted(question_stats.items()):
        n = stats["n"]
        mse = stats["sum"] / n
        question_rows.append(
            {
                "budget_per_group": budget,
                "panel_index": qindex,
                "method": meth,
                "replications": n,
                "mse": mse,
                "bias": stats["err"] / n,
            }
        )
    write_csv(output / "question_summary.csv", question_rows)
    replication_rows = [
        {"budget_per_group": b, "replication": r, "method": meth, "panel_mse": v}
        for (b, r, meth), v in sorted(panel_sums.items())
    ]
    write_csv(output / "panel_replications.csv", replication_rows)
    summary_rows = []
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for (b, _, meth), v in panel_sums.items():
        grouped[(b, meth)].append(v)
    for (b, meth), values in sorted(grouped.items()):
        arr = np.asarray(values)
        uniform = np.asarray(grouped[(b, "uniform")])
        summary_rows.append(
            {
                "budget_per_group": b,
                "method": meth,
                "replications": len(arr),
                "panel_mse": float(arr.mean()),
                "mcse": float(arr.std(ddof=1) / np.sqrt(len(arr))),
                "normalized_to_uniform": float(arr.mean() / uniform.mean()),
            }
        )
    write_csv(output / "panel_summary.csv", summary_rows)
    write_json(output / "diagnostics.json", diagnostics)
    write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "study": "finqa_terminal_risk",
            "workflow": artifact.get("workflow"),
            "kernel_artifact_sha256": file_sha256(kernel_artifact_path),
            "banks_sha256": file_sha256(banks_path),
            "configuration_sha256": object_sha256(config),
            "methods": list(methods),
            "questions": len(questions),
            "replications": replications,
            "runtime_seconds": time.time() - started,
        },
    )
    return output
