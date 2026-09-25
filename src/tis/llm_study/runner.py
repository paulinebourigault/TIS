from __future__ import annotations

import csv
import gzip
import json
import math
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..allocation import integer_allocation, largest_remainder, pilot_size, regularized_weights
from ..engine import InfluenceBundle, compute_influences, solve_stop_loss
from ..model import (
    FiniteHorizonModel,
    counts_from_streams,
    coupled_streams,
    model_from_counts,
    sample_counts,
)
from ..reporting import sha256, write_csv
from ..runner import _mc_ucb_allocation
from . import (
    LLM_EXTENSION_METHODS,
    LLM_EXTENSION_PANEL,
    LLM_METHODS,
    LLM_ROLLOUT_PANEL,
)
from .calibrate import validate_kernel_artifact
from .common import object_sha256, read_json, write_json
from .models import shared_question_model, untied_question_model


SHARED_LEARNED = {
    "answer_entropy",
    "tis",
    "mc_ucb_frozen",
    "adaptive_neyman",
    "learned_mean",
    "learned_occupancy",
    "tis_no_cov",
    "tis_anchored",
    "occ_plus_uniform",
    "occ_plus_mean",
}
# Mixture controls: equal-weight blends of component weight vectors that
# already carry the exploration floor; no second floor after mixing.
MIXTURES = {"occ_plus_uniform", "occ_plus_mean"}


def _rng(
    master_seed: int,
    horizon: int,
    alpha_index: int,
    budget: int,
    question: int,
    replication: int,
    stage: int,
) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(
            [master_seed, horizon, alpha_index, budget, question, replication, stage]
        )
    )


def _mapping(model: FiniteHorizonModel, values: np.ndarray) -> dict[tuple[int, int], int]:
    return {group: int(value) for group, value in zip(model.query_groups or (), values)}


def _occupancy_scores(model: FiniteHorizonModel, bundle: InfluenceBundle) -> np.ndarray:
    return np.asarray(
        [
            np.sum(bundle.occupancy[:, state] * model.policy[:, state, action])
            for state, action in model.query_groups or ()
        ],
        dtype=float,
    )


def _answer_entropy(counts: Mapping[tuple[int, int], np.ndarray], model: FiniteHorizonModel) -> np.ndarray:
    scores: list[float] = []
    for group in model.query_groups or ():
        values = np.asarray(counts[group], dtype=float).reshape(10, 9).sum(axis=1)
        probabilities = values / values.sum()
        positive = probabilities > 0
        scores.append(float(-np.sum(probabilities[positive] * np.log(probabilities[positive]))))
    return np.asarray(scores)


def _complete_rollouts(
    model: FiniteHorizonModel,
    horizon: int,
    runs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent full trajectories through the true conditional laws.

    Returns the sampled H-step returns and the realized per-group visit
    counts. States are visited in sorted order at every step, so the output
    is deterministic given the generator.
    """
    groups = tuple(model.query_groups or ())
    index_of_state = {state: i for i, (state, _) in enumerate(groups)}
    state = np.full(runs, model.initial_state, dtype=int)
    totals = np.zeros(runs, dtype=float)
    visits = np.zeros(len(groups), dtype=np.int64)
    for _ in range(horizon):
        successor = np.empty_like(state)
        for current in np.unique(state):
            mask = state == current
            count = int(mask.sum())
            group = groups[index_of_state[int(current)]]
            kernel = model.kernels[group]
            visits[index_of_state[int(current)]] += count
            outcomes = rng.choice(kernel.size, size=count, p=kernel.probabilities)
            totals[mask] += kernel.rewards[outcomes]
            successor[mask] = kernel.next_states[outcomes]
        state = successor
    return totals, visits


def _empirical_cvar(values: np.ndarray, alpha: float) -> float:
    """Empirical lower-tail CVaR with fractional handling of the quantile atom."""
    ordered = np.sort(np.asarray(values, dtype=float))
    n = len(ordered)
    mass = alpha * n
    whole = int(math.floor(mass))
    total = float(ordered[:whole].sum())
    if whole < n and mass > whole:
        total += (mass - whole) * float(ordered[whole])
    return total / mass



def _anchored_score(influence: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
    """Equal-weight ensemble of the pilot influence and pilot occupancy designs.

    Components are normalized (uniform on a zero total) and averaged with
    weight one half; the exploration floor applies afterwards. Every blended
    weight is at least half of each component's, so the design variance is
    at most twice the better component's.
    """
    parts = []
    for score in (influence, occupancy):
        total = float(np.sum(score))
        parts.append(score / total if total > 0 else np.full(len(score), 1.0 / len(score)))
    return 0.5 * parts[0] + 0.5 * parts[1]


def _adaptive_plugin_neyman(
    model: FiniteHorizonModel,
    frozen_scores: tuple[np.ndarray, ...],
    main_budget: int,
    exploration: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray], np.ndarray]:
    """Frozen-score, batched adaptive plug-in Neyman allocation.

    Two observations initialize every group; each later batch of at most G
    samples closes the largest deficits against a cumulative fractional
    quota (batch*weights added per batch) by largest remainder. The quota
    is cumulative so small per-batch shares such as the exploration floor
    accumulate instead of being rounded away every batch. Scores stay
    frozen at the pilot estimate; only second moments update.
    """
    groups = tuple(model.query_groups or ())
    group_count = len(groups)
    if main_budget < 2 * group_count:
        raise ValueError("Adaptive main budget must permit two draws per group")
    arm_seeds = rng.integers(
        0, np.iinfo(np.uint64).max, size=group_count, dtype=np.uint64
    )
    arm_rng = [np.random.default_rng(seed) for seed in arm_seeds]
    outcome_counts = [
        np.zeros(model.kernels[group].size, dtype=int) for group in groups
    ]
    pulls = np.zeros(group_count, dtype=int)

    def add_batch(group_index: int, count: int) -> None:
        if count <= 0:
            return
        group = groups[group_index]
        draw = arm_rng[group_index].multinomial(
            count, model.kernels[group].probabilities
        )
        outcome_counts[group_index] += draw
        pulls[group_index] += count

    for group_index in range(group_count):
        add_batch(group_index, 2)
    remaining = main_budget - 2 * group_count
    quota = np.full(group_count, 2.0)  # cumulative fractional entitlement
    while remaining:
        scales = np.asarray(
            [
                np.sqrt(
                    max(
                        float(
                            np.sum(
                                counts
                                * (score - np.sum(counts * score) / count) ** 2
                            )
                            / count
                        ),
                        0.0,
                    )
                )
                for counts, score, count in zip(outcome_counts, frozen_scores, pulls)
            ]
        )
        weights = regularized_weights(scales, exploration)
        batch = min(group_count, remaining)
        quota += batch * weights
        deficit = np.maximum(quota - pulls, 0.0)
        if deficit.sum() <= 0.0:
            deficit = weights
        additions = largest_remainder(batch, deficit / deficit.sum())
        for group_index, count in enumerate(additions):
            add_batch(group_index, int(count))
        remaining -= batch
    final_scales = np.asarray(
        [
            np.sqrt(
                max(
                    float(
                        np.sum(
                            counts
                            * (score - np.sum(counts * score) / count) ** 2
                        )
                        / count
                    ),
                    0.0,
                )
            )
            for counts, score, count in zip(outcome_counts, frozen_scores, pulls)
        ]
    )
    return (
        pulls,
        {group: counts for group, counts in zip(groups, outcome_counts)},
        final_scales,
    )


def _token_cost(
    question: Mapping[str, object],
    model: FiniteHorizonModel,
    main_counts: np.ndarray,
    pilot_per_group: int,
) -> int:
    input_tokens = {
        int(kernel["state_index"]): int(kernel.get("input_tokens", 0))
        for kernel in question["kernels"]  # type: ignore[index]
    }
    costs = []
    for first, second in model.query_groups or ():
        state = second if model.group_mode == "untied_layer_state" else first
        costs.append(input_tokens[state] + 2)
    return int(np.dot(main_counts + pilot_per_group, np.asarray(costs, dtype=int)))


def _update_allocation(
    accumulator: dict[tuple[object, ...], dict[str, object]],
    key: tuple[object, ...],
    model: FiniteHorizonModel,
    counts: np.ndarray,
    weights: np.ndarray,
    scores: np.ndarray,
    truth_scales: np.ndarray,
) -> None:
    if key not in accumulator:
        accumulator[key] = {
            "replications": 0,
            "count_sum": np.zeros(model.group_count),
            "weight_sum": np.zeros(model.group_count),
            "score_sum": np.zeros(model.group_count),
            "score_square_sum": np.zeros(model.group_count),
            "truth_scales": truth_scales.copy(),
            "groups": tuple(model.query_groups or ()),
            "group_mode": model.group_mode,
        }
    record = accumulator[key]
    record["replications"] = int(record["replications"]) + 1
    record["count_sum"] += counts  # type: ignore[operator]
    record["weight_sum"] += weights  # type: ignore[operator]
    record["score_sum"] += scores  # type: ignore[operator]
    record["score_square_sum"] += scores**2  # type: ignore[operator]


def _question_cell(
    payload: tuple[Mapping[str, object], int, Mapping[str, object], tuple[str, ...], int],
) -> dict[str, object]:
    """Run every replication for one (question, horizon) pair.

    This is the unit of parallel work. All randomness is seeded from the
    (cell, stage) coordinates alone, so the output is identical whether cells
    run serially or across processes.
    """
    question, horizon, config, methods, question_count = payload
    replications = int(config["replications"])
    minimum_main = int(config["minimum_main_per_group"])
    master_seed = int(config["master_seed"])
    raw_rows: list[dict[str, object]] = []
    question_stats: dict[tuple[object, ...], dict[str, float]] = defaultdict(
        lambda: {"n": 0.0, "sum": 0.0, "sum2": 0.0, "error_sum": 0.0}
    )
    panel_rep_sums: dict[tuple[object, ...], float] = defaultdict(float)
    allocation_accumulator: dict[tuple[object, ...], dict[str, object]] = {}
    diagnostics: list[dict[str, object]] = []
    shared = shared_question_model(
        question,
        horizon,
        closed_grid=bool(config.get("closed_grid", False)),
        utility=str(config.get("utility", "brier")),
        policy=str(config.get("policy", "default")),
    )
    untied = untied_question_model(shared)
    for alpha_index, alpha_raw in enumerate(config["alphas"]):
        alpha = float(alpha_raw)
        truth = compute_influences(shared, alpha)
        untied_truth = compute_influences(untied, alpha)
        if abs(truth.solution.cvar - untied_truth.solution.cvar) > 1e-10:
            raise RuntimeError("Shared and population-untied targets disagree")
        uniform_constant = shared.group_count * float(np.sum(truth.tail_scales**2))
        oracle_constant = float(truth.tail_scales.sum() ** 2)
        diagnostics.append(
            {
                "horizon": horizon,
                "alpha": alpha,
                "panel_index": int(question["panel_index"]),
                "item_id": str(question["item_id"]),
                "subject": str(question["subject"]),
                "truth": truth.solution.cvar,
                "margin": truth.solution.margin,
                "positive_margin": truth.solution.margin > 0,
                "shared_groups": shared.group_count,
                "untied_groups": untied.group_count,
                "groups_reused_across_layers": 90 if horizon > 2 else 0,
                "uniform_to_oracle": (
                    uniform_constant / oracle_constant if oracle_constant > 0 else None
                ),
                "zero_tail_signal": oracle_constant == 0,
            }
        )
        for budget_per_group_raw in config["budgets_per_shared_group"]:
            budget_per_group = int(budget_per_group_raw)
            total_budget = shared.group_count * budget_per_group
            exploration = total_budget ** (-float(config["exploration_exponent"]))
            m_shared = pilot_size(
                total_budget,
                shared.group_count,
                float(config["pilot_multiplier"]),
                int(config["minimum_pilot_per_group"]),
                minimum_main,
            )
            shared_main = total_budget - shared.group_count * m_shared
            m_untied = pilot_size(
                total_budget,
                untied.group_count,
                float(config["pilot_multiplier"]),
                int(config["minimum_pilot_per_group"]),
                minimum_main,
            )
            untied_main = total_budget - untied.group_count * m_untied
            for replication in range(replications):
                qindex = int(question["panel_index"])
                pilot_counts = sample_counts(
                    shared,
                    {group: m_shared for group in shared.query_groups or ()},
                    _rng(master_seed, horizon, alpha_index, total_budget, qindex, replication, 0),
                )
                pilot_bundle = compute_influences(
                    model_from_counts(shared, pilot_counts), alpha
                )
                scores: dict[str, np.ndarray] = {
                    "uniform": np.ones(shared.group_count),
                    "occupancy": _occupancy_scores(shared, truth),
                    "mean_influence": truth.mean_scales,
                    "answer_entropy": _answer_entropy(pilot_counts, shared),
                    "oracle_tail": truth.tail_scales,
                    "tis": pilot_bundle.tail_scales,
                    "learned_mean": pilot_bundle.mean_scales,
                    "learned_occupancy": _occupancy_scores(shared, pilot_bundle),
                    "tis_no_cov": pilot_bundle.no_cov_scales,
                    "oracle_no_cov": truth.no_cov_scales,
                    "tis_anchored": _anchored_score(
                        pilot_bundle.tail_scales, _occupancy_scores(shared, pilot_bundle)
                    ),
                }
                counts_by_method: dict[str, np.ndarray] = {}
                weights_by_method: dict[str, np.ndarray] = {}
                fixed_methods = [
                    method for method in methods
                    if method in {"uniform", "occupancy", "mean_influence", "answer_entropy", "oracle_tail", "tis", "tis_anchored"}
                    or method in LLM_EXTENSION_METHODS
                    or method in MIXTURES
                ]
                for method in fixed_methods:
                    if method == "uniform":
                        weights = np.full(shared.group_count, 1.0 / shared.group_count)
                        main_budget = total_budget
                    elif method in MIXTURES:
                        w_occ = regularized_weights(scores["learned_occupancy"], exploration)
                        if method == "occ_plus_uniform":
                            other = np.full(shared.group_count, 1.0 / shared.group_count)
                        else:
                            other = regularized_weights(scores["learned_mean"], exploration)
                        weights = 0.5 * other + 0.5 * w_occ
                        scores[method] = weights
                        main_budget = shared_main
                    else:
                        weights = regularized_weights(scores[method], exploration)
                        main_budget = shared_main if method in SHARED_LEARNED else total_budget
                    weights_by_method[method] = weights
                    counts_by_method[method] = integer_allocation(
                        main_budget, weights, minimum_main
                    )
                empirical_by_method: dict[str, Mapping[tuple[int, int], np.ndarray]] = {}
                if fixed_methods:
                    maximum = np.max(
                        np.vstack([counts_by_method[method] for method in fixed_methods]),
                        axis=0,
                    )
                    streams = coupled_streams(
                        shared,
                        _mapping(shared, maximum),
                        _rng(master_seed, horizon, alpha_index, total_budget, qindex, replication, 1),
                    )
                    for method in fixed_methods:
                        empirical_by_method[method] = counts_from_streams(
                            shared, streams, _mapping(shared, counts_by_method[method])
                        )
                if "mc_ucb_frozen" in methods:
                    mc_counts, mc_empirical, mc_scales, _, _ = _mc_ucb_allocation(
                        shared,
                        pilot_bundle.tail_values,
                        shared_main,
                        _rng(master_seed, horizon, alpha_index, total_budget, qindex, replication, 2),
                    )
                    counts_by_method["mc_ucb_frozen"] = mc_counts
                    weights_by_method["mc_ucb_frozen"] = mc_counts / mc_counts.sum()
                    scores["mc_ucb_frozen"] = mc_scales
                    empirical_by_method["mc_ucb_frozen"] = mc_empirical
                if "adaptive_neyman" in methods:
                    adaptive_counts, adaptive_empirical, adaptive_scales = _adaptive_plugin_neyman(
                        shared,
                        pilot_bundle.tail_values,
                        shared_main,
                        exploration,
                        _rng(master_seed, horizon, alpha_index, total_budget, qindex, replication, 3),
                    )
                    counts_by_method["adaptive_neyman"] = adaptive_counts
                    weights_by_method["adaptive_neyman"] = adaptive_counts / adaptive_counts.sum()
                    scores["adaptive_neyman"] = adaptive_scales
                    empirical_by_method["adaptive_neyman"] = adaptive_empirical

                if "untied_tis" in methods:
                    untied_pilot_counts = sample_counts(
                        untied,
                        {group: m_untied for group in untied.query_groups or ()},
                        _rng(master_seed, horizon, alpha_index, total_budget, qindex, replication, 4),
                    )
                    untied_pilot = compute_influences(
                        model_from_counts(untied, untied_pilot_counts), alpha
                    )
                    untied_weights = regularized_weights(
                        untied_pilot.tail_scales, exploration
                    )
                    untied_counts = integer_allocation(
                        untied_main, untied_weights, minimum_main
                    )
                    untied_empirical = sample_counts(
                        untied,
                        _mapping(untied, untied_counts),
                        _rng(master_seed, horizon, alpha_index, total_budget, qindex, replication, 5),
                    )

                if "complete_rollout" in methods:
                    rollout_runs = total_budget // horizon
                    rollout_returns, rollout_visits = _complete_rollouts(
                        shared,
                        horizon,
                        rollout_runs,
                        _rng(master_seed, horizon, alpha_index, total_budget, qindex, replication, 6),
                    )
                    rollout_estimate = _empirical_cvar(rollout_returns, alpha)

                for method in methods:
                    if method == "untied_tis":
                        evaluation_model = untied
                        empirical = untied_empirical
                        counts = untied_counts
                        weights = untied_weights
                        method_scores = untied_pilot.tail_scales
                        method_truth_scales = untied_truth.tail_scales
                        pilot_per_group = m_untied
                    elif method == "complete_rollout":
                        evaluation_model = shared
                        empirical = None
                        counts = rollout_visits
                        weights = rollout_visits / max(int(rollout_visits.sum()), 1)
                        method_scores = np.zeros(shared.group_count)
                        method_truth_scales = truth.tail_scales
                        pilot_per_group = 0
                    else:
                        evaluation_model = shared
                        empirical = empirical_by_method[method]
                        counts = counts_by_method[method]
                        weights = weights_by_method[method]
                        method_scores = scores[method]
                        method_truth_scales = truth.tail_scales
                        pilot_per_group = m_shared if method in SHARED_LEARNED else 0
                    if method == "complete_rollout":
                        estimate = rollout_estimate
                        # Trajectory sampling has no per-group design constant;
                        # visits at unreached groups are legitimately zero.
                        design_constant = ""
                    else:
                        estimate = solve_stop_loss(
                            model_from_counts(evaluation_model, empirical), alpha
                        ).cvar
                        design_constant = total_budget * float(
                            np.sum(method_truth_scales**2 / counts)
                        )
                    error = estimate - truth.solution.cvar
                    squared_error = error**2
                    pilot_total = evaluation_model.group_count * pilot_per_group
                    raw_rows.append(
                        {
                            "horizon": horizon,
                            "alpha": alpha,
                            "budget_per_shared_group": budget_per_group,
                            "total_budget": total_budget,
                            "panel_index": qindex,
                            "item_id": question["item_id"],
                            "subject": question["subject"],
                            "replication": replication,
                            "method": method,
                            "truth": truth.solution.cvar,
                            "estimate": estimate,
                            "error": error,
                            "squared_error": squared_error,
                            "normalized_squared_error_scale": squared_error / max(truth.solution.cvar**2, 1e-15),
                            "pilot_per_group": pilot_per_group,
                            "pilot_total": pilot_total,
                            "main_total": int(counts.sum()),
                            "design_constant": design_constant,
                            "logical_queries": total_budget,
                            "estimated_tokens": _token_cost(
                                question, evaluation_model, counts, pilot_per_group
                            ),
                            "group_mode": evaluation_model.group_mode,
                            "groups": evaluation_model.group_count,
                        }
                    )
                    stat_key = (horizon, alpha, budget_per_group, qindex, method)
                    stats = question_stats[stat_key]
                    stats["n"] += 1
                    stats["sum"] += squared_error
                    stats["sum2"] += squared_error**2
                    stats["error_sum"] += error
                    panel_rep_sums[(horizon, alpha, budget_per_group, replication, method)] += (
                        squared_error / question_count
                    )
                    _update_allocation(
                        allocation_accumulator,
                        (horizon, alpha, budget_per_group, qindex, method),
                        evaluation_model,
                        counts,
                        weights,
                        method_scores,
                        method_truth_scales,
                    )
    return {
        "raw_rows": raw_rows,
        "question_stats": dict(question_stats),
        "panel_rep_sums": dict(panel_rep_sums),
        "allocation_accumulator": allocation_accumulator,
        "diagnostics": diagnostics,
    }


CELL_CACHE_VERSION = 1


def _keyed_pairs(mapping: Mapping[tuple[object, ...], object]) -> list[list[object]]:
    return [[list(key), value] for key, value in mapping.items()]


def _store_cell(path: Path, cell: Mapping[str, object]) -> None:
    payload = {
        "raw_rows": cell["raw_rows"],
        "question_stats": _keyed_pairs(cell["question_stats"]),
        "panel_rep_sums": _keyed_pairs(cell["panel_rep_sums"]),
        "allocation_accumulator": [
            [
                list(key),
                {
                    "replications": record["replications"],
                    "count_sum": record["count_sum"].tolist(),
                    "weight_sum": record["weight_sum"].tolist(),
                    "score_sum": record["score_sum"].tolist(),
                    "score_square_sum": record["score_square_sum"].tolist(),
                    "truth_scales": record["truth_scales"].tolist(),
                    "groups": [list(group) for group in record["groups"]],
                    "group_mode": record["group_mode"],
                },
            ]
            for key, record in cell["allocation_accumulator"].items()
        ],
        "diagnostics": cell["diagnostics"],
    }
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)
    os.replace(temporary, path)


def _load_cell(path: Path) -> dict[str, object]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    return {
        "raw_rows": payload["raw_rows"],
        "question_stats": {
            tuple(key): value for key, value in payload["question_stats"]
        },
        "panel_rep_sums": {
            tuple(key): value for key, value in payload["panel_rep_sums"]
        },
        "allocation_accumulator": {
            tuple(key): {
                "replications": record["replications"],
                "count_sum": np.asarray(record["count_sum"], dtype=float),
                "weight_sum": np.asarray(record["weight_sum"], dtype=float),
                "score_sum": np.asarray(record["score_sum"], dtype=float),
                "score_square_sum": np.asarray(
                    record["score_square_sum"], dtype=float
                ),
                "truth_scales": np.asarray(record["truth_scales"], dtype=float),
                "groups": tuple(tuple(group) for group in record["groups"]),
                "group_mode": record["group_mode"],
            }
            for key, record in payload["allocation_accumulator"]
        },
        "diagnostics": payload["diagnostics"],
    }


def run_llm_study(
    config_path: Path,
    kernel_artifact_path: Path,
    output_directory: Path | None = None,
    overwrite: bool = False,
    processes: int = 1,
) -> Path:
    config = read_json(config_path)
    artifact = read_json(kernel_artifact_path)
    if not isinstance(config, dict) or not isinstance(artifact, dict):
        raise ValueError("Configuration and kernel artifact must be JSON objects")
    validate_kernel_artifact(kernel_artifact_path)
    config_policy = str(config.get("policy", "default"))
    artifact_policy = str(artifact.get("policy", "default"))
    if config_policy != artifact_policy:
        raise ValueError(
            f"Configuration review policy {config_policy!r} does not match the "
            f"kernel artifact's calibrated policy {artifact_policy!r}"
        )
    configured = Path(str(config["output_directory"]))
    output = output_directory or (
        configured if configured.is_absolute() else config_path.resolve().parent.parent / configured
    )
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Result directory already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    write_json(output / "config.json", config)
    methods = tuple(str(method) for method in config.get("methods", LLM_METHODS))
    if (
        set(methods)
        - (set(LLM_METHODS) | set(LLM_EXTENSION_METHODS) | MIXTURES
           | {"complete_rollout", "tis_anchored"})
        or len(methods) != len(set(methods))
    ):
        raise ValueError("Invalid LLM method panel")
    questions = artifact["questions"]
    question_indices = [int(value) for value in config.get("question_indices", range(len(questions)))]
    selected_questions = [questions[index] for index in question_indices]
    replications = int(config["replications"])
    if processes < 1:
        raise ValueError("Process count must be at least one")
    raw_fields = [
        "horizon", "alpha", "budget_per_shared_group", "total_budget",
        "panel_index", "item_id", "subject", "replication", "method",
        "truth", "estimate", "error", "squared_error", "normalized_squared_error_scale",
        "pilot_per_group", "pilot_total", "main_total", "design_constant",
        "logical_queries", "estimated_tokens", "group_mode", "groups",
    ]
    raw_stream = (output / "raw.csv").open("w", newline="", encoding="utf-8")
    raw_writer = csv.DictWriter(raw_stream, fieldnames=raw_fields)
    raw_writer.writeheader()
    question_stats: dict[tuple[object, ...], dict[str, float]] = defaultdict(
        lambda: {"n": 0.0, "sum": 0.0, "sum2": 0.0, "error_sum": 0.0}
    )
    panel_rep_sums: dict[tuple[object, ...], float] = defaultdict(float)
    allocation_accumulator: dict[tuple[object, ...], dict[str, object]] = {}
    diagnostics: list[dict[str, object]] = []
    started = time.time()
    work_items = [
        (question, horizon, config, methods, len(selected_questions))
        for horizon in [int(value) for value in config["horizons"]]
        for question in selected_questions
    ]

    # Completed (question, horizon) cells are cached on disk so an
    # interrupted run resumes mid-simulation. Cells are deterministic given
    # the configuration and kernel artifact, which form the cache signature.
    cache_signature = object_sha256(
        {
            "cell_cache_version": CELL_CACHE_VERSION,
            "config": {
                key: value
                for key, value in config.items()
                if key != "output_directory"
            },
            "kernel_content_sha256": artifact.get("content_sha256"),
        }
    )[:16]
    cache_root = (
        config_path.resolve().parent.parent
        / "build"
        / "llm_cell_cache"
        / cache_signature
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_paths = [
        cache_root / f"h{item[1]}_q{int(item[0]['panel_index']):03d}.json.gz"
        for item in work_items
    ]
    cached_flags = [path.is_file() for path in cache_paths]
    pending_items = [
        item
        for item, cached in zip(work_items, cached_flags)
        if not cached
    ]
    cells_from_cache = len(work_items) - len(pending_items)

    import threading

    heartbeat_path = output / "heartbeat.json"
    heartbeat_stop = threading.Event()

    def _heartbeat() -> None:
        # Liveness signal on the shared filesystem: compute nodes are not
        # directly reachable, and long horizon-6 cells otherwise leave no
        # observable progress for hours.
        while not heartbeat_stop.wait(60.0):
            write_json(
                heartbeat_path,
                {
                    "unix_time": time.time(),
                    "pid": os.getpid(),
                    "cells_completed_at_start": cells_from_cache,
                },
            )

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()

    def _merge(cell: dict[str, object]) -> None:
        for row in cell["raw_rows"]:
            raw_writer.writerow(row)
        for key, stats in cell["question_stats"].items():
            merged = question_stats[key]
            for field in ("n", "sum", "sum2", "error_sum"):
                merged[field] += stats[field]
        for key, value in cell["panel_rep_sums"].items():
            panel_rep_sums[key] += value
        allocation_accumulator.update(cell["allocation_accumulator"])
        diagnostics.extend(cell["diagnostics"])

    def _drain(fresh_cells) -> None:
        fresh_iterator = iter(fresh_cells)
        for path, cached in zip(cache_paths, cached_flags):
            if cached:
                cell = _load_cell(path)
            else:
                cell = next(fresh_iterator)
                _store_cell(path, cell)
            _merge(cell)

    if processes > 1 and pending_items:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=processes) as pool:
            _drain(pool.map(_question_cell, pending_items, chunksize=1))
    else:
        _drain(_question_cell(item) for item in pending_items)
    raw_stream.close()
    heartbeat_stop.set()
    heartbeat_thread.join(timeout=5.0)
    heartbeat_path.unlink(missing_ok=True)

    question_rows: list[dict[str, object]] = []
    question_mse: dict[tuple[object, ...], float] = {}
    for key, stats in sorted(question_stats.items()):
        horizon, alpha, budget, qindex, method = key
        n = int(stats["n"])
        mse = stats["sum"] / n
        variance = max(stats["sum2"] / n - mse**2, 0.0)
        question_mse[key] = mse
        question_rows.append(
            {
                "horizon": horizon,
                "alpha": alpha,
                "budget_per_shared_group": budget,
                "panel_index": qindex,
                "method": method,
                "replications": n,
                "mse": mse,
                "mse_se": math.sqrt(variance / n),
                "bias": stats["error_sum"] / n,
                "normalized_to_uniform": mse / max(
                    question_stats[(horizon, alpha, budget, qindex, "uniform")]["sum"] / n,
                    1e-15,
                ) if "uniform" in methods else "",
            }
        )
    write_csv(output / "question_summary.csv", question_rows)

    panel_rep_rows: list[dict[str, object]] = []
    for key, panel_mse in sorted(panel_rep_sums.items()):
        horizon, alpha, budget, replication, method = key
        panel_rep_rows.append(
            {
                "horizon": horizon,
                "alpha": alpha,
                "budget_per_shared_group": budget,
                "replication": replication,
                "method": method,
                "panel_mean_squared_error": panel_mse,
            }
        )
    write_csv(output / "panel_replications.csv", panel_rep_rows)
    panel_summary_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    panel_groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for key, value in panel_rep_sums.items():
        panel_groups[(key[0], key[1], key[2], key[4])].append(value)
    for key, values_raw in sorted(panel_groups.items()):
        values = np.asarray(values_raw)
        horizon, alpha, budget, method = key
        uniform = np.asarray(panel_groups[(horizon, alpha, budget, "uniform")]) if "uniform" in methods else None
        panel_summary_rows.append(
            {
                "horizon": horizon,
                "alpha": alpha,
                "budget_per_shared_group": budget,
                "method": method,
                "replications": len(values),
                "panel_mse": float(values.mean()),
                "mcse": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0,
                "normalized_to_uniform": float(values.mean() / uniform.mean()) if uniform is not None else "",
            }
        )
        if method != "tis" and "tis" in methods:
            tis_values = np.asarray(panel_groups[(horizon, alpha, budget, "tis")])
            difference = values - tis_values
            paired_rows.append(
                {
                    "horizon": horizon,
                    "alpha": alpha,
                    "budget_per_shared_group": budget,
                    "comparison": f"{method}-minus-tis",
                    "replications": len(difference),
                    "mean_difference": float(difference.mean()),
                    "se": float(difference.std(ddof=1) / np.sqrt(len(difference))) if len(difference) > 1 else 0.0,
                }
            )
    write_csv(output / "panel_summary.csv", panel_summary_rows)
    if paired_rows:
        write_csv(output / "paired_panel.csv", paired_rows)

    allocation_rows: list[dict[str, object]] = []
    for key, record in sorted(allocation_accumulator.items()):
        horizon, alpha, budget, qindex, method = key
        n = int(record["replications"])
        score_mean = record["score_sum"] / n
        score_variance = np.maximum(record["score_square_sum"] / n - score_mean**2, 0.0)
        for group_index, group in enumerate(record["groups"]):
            allocation_rows.append(
                {
                    "horizon": horizon,
                    "alpha": alpha,
                    "budget_per_shared_group": budget,
                    "panel_index": qindex,
                    "method": method,
                    "group_mode": record["group_mode"],
                    "group_index": group_index,
                    "group_first": group[0],
                    "group_second": group[1],
                    "replications": n,
                    "mean_main_count": float(record["count_sum"][group_index] / n),
                    "mean_weight": float(record["weight_sum"][group_index] / n),
                    "mean_score": float(score_mean[group_index]),
                    "score_sd": float(np.sqrt(score_variance[group_index])),
                    "true_tail_scale": float(record["truth_scales"][group_index]),
                }
            )
    write_csv(output / "allocation_summary.csv", allocation_rows)
    write_json(output / "diagnostics.json", diagnostics)
    full_grid = (
        artifact.get("evidence_status") == "paper_eligible"
        and len(selected_questions) == 50
        and replications == 300
        and set(int(value) for value in config["budgets_per_shared_group"]) == {50, 100, 200, 400}
        and set(int(value) for value in config["horizons"]) == {2, 4, 6}
        and set(float(value) for value in config["alphas"]) == {0.1, 0.2}
    )
    closed_grid = bool(config.get("closed_grid", False))
    utility = str(config.get("utility", "brier"))
    review_policy = str(config.get("policy", "default"))
    full_grid = full_grid and utility == "brier" and review_policy == "default"
    expected_complete = full_grid and set(methods) == set(LLM_METHODS)
    extension_complete = full_grid and set(methods) in (
        set(LLM_EXTENSION_PANEL),
        set(LLM_ROLLOUT_PANEL),
    )
    result_files = [
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = {
        "schema_version": 1,
        "result_evidence_status": (
            ("paper_eligible_closed_grid" if closed_grid else "paper_eligible_complete")
            if expected_complete
            else (
                "paper_eligible_closed_grid_extension"
                if closed_grid
                else "paper_eligible_extension"
            )
            if extension_complete
            else "execution_or_partial_only"
        ),
        "closed_grid": closed_grid,
        "utility": utility,
        "policy": review_policy,
        "kernel_evidence_status": artifact.get("evidence_status"),
        "kernel_model_key": artifact.get("model_key", "model"),
        "kernel_model_identifier": (
            artifact.get("model", {}).get("identifier")
            if isinstance(artifact.get("model"), dict)
            else None
        ),
        "kernel_artifact_sha256": sha256(kernel_artifact_path),
        "configuration_sha256": sha256(config_path),
        "runtime_seconds": time.time() - started,
        "worker_processes": processes,
        "cells_from_cache": cells_from_cache,
        "cell_cache_signature": cache_signature,
        "questions": len(selected_questions),
        "replications": replications,
        "methods": list(methods),
        "files": {
            path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(result_files)
        },
    }
    write_json(output / "manifest.json", manifest)
    return output
