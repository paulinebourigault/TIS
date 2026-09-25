from __future__ import annotations

import json
import heapq
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Mapping

import matplotlib
import numpy as np

from . import __version__
from .allocation import integer_allocation, pilot_size, regularized_weights
from .engine import compute_influences, solve_stop_loss
from .environments import make_environment
from .model import (
    FiniteHorizonModel,
    counts_from_streams,
    coupled_streams,
    model_from_counts,
    sample_counts,
)
from .reporting import json_dump, sha256, summarize, write_csv, write_figure, write_latex_table


METHODS = (
    "uniform",
    "occupancy",
    "mean_influence",
    "mc_ucb_frozen",
    "tis",
    "oracle_tail",
    "tail_no_cov",
    "learned_mean",
    "oracle_no_cov",
)

SUPPORTED_METHODS = METHODS + (
    "reachability_only",
    "local_only",
    "tail_anchored",
    "learned_occupancy",
)

LEARNED_METHODS = {
    "mc_ucb_frozen",
    "tis",
    "tail_no_cov",
    "learned_mean",
    "tail_anchored",
    "learned_occupancy",
}



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


def _group_mapping(model: FiniteHorizonModel, values: np.ndarray) -> dict[tuple[int, int], int]:
    return {group: int(value) for group, value in zip(model.query_groups or (), values)}


def _environment_snapshot(model: FiniteHorizonModel) -> dict[str, object]:
    return {
        "name": model.name,
        "horizon": model.horizon,
        "initial_state": model.initial_state,
        "grid": model.grid.tolist(),
        "policy": model.policy.tolist(),
        "query_groups": [list(group) for group in model.query_groups or ()],
        "group_mode": model.group_mode,
        "kernels": {
            f"{first},{second}": {
                "rewards": kernel.rewards.tolist(),
                "next_states": kernel.next_states.tolist(),
                "probabilities": kernel.probabilities.tolist(),
            }
            for (first, second), kernel in sorted(model.kernels.items())
        },
        "metadata": dict(model.metadata or {}),
    }


def _source_snapshot(repository_root: Path) -> dict[str, dict[str, object]]:
    """Hash exactly the files that determine simulation results.

    The snapshot is restricted to the package sources, configurations,
    tests, frozen specs, and dependency locks. Documentation, papers, job
    scripts, logs, and repository housekeeping files are excluded so that
    editing them cannot invalidate the verification of an existing run.
    """
    retained: list[Path] = []
    top_level_allowlist = {
        "pyproject.toml", "requirements-lock.txt", "requirements-llm-lock.txt"
    }
    for path in repository_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(repository_root)
        if len(relative.parts) == 1:
            if relative.parts[0] not in top_level_allowlist:
                continue
        elif relative.parts[0] not in {"src", "tests", "configs", "specs"}:
            continue
        if relative.parts[0] in {"results", "build", "dist"}:
            continue
        if (
            "__pycache__" in relative.parts
            or ".pytest_cache" in relative.parts
            or any(part.endswith(".egg-info") for part in relative.parts)
            or path.suffix == ".pyc"
        ):
            continue
        if path.name == "SOURCE_MANIFEST.json":
            continue
        retained.append(path)
    return {
        str(path.relative_to(repository_root)): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(retained)
    }


def _seed(master_seed: int, environment_index: int, budget: int, replication: int, stage: int) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [master_seed, environment_index, budget, replication, stage]
    )
    return np.random.default_rng(sequence)


def _occupancy_scores(model: FiniteHorizonModel, occupancy: np.ndarray) -> np.ndarray:
    if model.group_mode == "untied_layer_state":
        return np.asarray(
            [occupancy[h, state] for h, state in model.query_groups or ()],
            dtype=float,
        )
    return np.asarray(
        [
            np.sum(occupancy[:, state] * model.policy[:, state, action])
            for state, action in model.query_groups or ()
        ],
        dtype=float,
    )


def _methods_for(
    config: Mapping[str, object], environment_config: Mapping[str, object]
) -> tuple[str, ...]:
    raw = environment_config.get("methods", config.get("methods", METHODS))
    methods = tuple(str(method) for method in raw)  # type: ignore[arg-type]
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Method panel must be nonempty and contain no duplicates")
    unknown = set(methods) - set(SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    return methods


def _budgets_for(
    environment_config: Mapping[str, object], group_count: int
) -> list[tuple[str, int, int, float]]:
    has_per_group = "budgets_per_group" in environment_config
    has_total = "total_budgets" in environment_config
    if has_per_group == has_total:
        raise ValueError(
            "Specify exactly one of budgets_per_group or total_budgets"
        )
    if has_per_group:
        values = environment_config["budgets_per_group"]  # type: ignore[index]
        return [
            ("per_group", int(value), group_count * int(value), float(value))
            for value in values  # type: ignore[union-attr]
        ]
    values = environment_config["total_budgets"]  # type: ignore[index]
    return [
        ("total", int(value), int(value), float(value) / group_count)
        for value in values  # type: ignore[union-attr]
    ]


def _mc_ucb_allocation(
    model: FiniteHorizonModel,
    frozen_scores: tuple[np.ndarray, ...],
    main_budget: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray], np.ndarray, float, float]:
    """Published bounded-arm MC-UCB index applied to pilot-frozen scores."""
    groups = tuple(model.query_groups or ())
    group_count = len(groups)
    if main_budget < 2 * group_count:
        raise ValueError("MC-UCB main budget must permit two pulls per group")
    ranges = np.asarray([np.ptp(values) for values in frozen_scores], dtype=float)
    score_range = float(np.max(ranges))
    delta = float(main_budget) ** (-4.5)
    beta = score_range * np.sqrt(np.log(2.0 / delta))

    pulls = np.zeros(group_count, dtype=int)
    score_sum = np.zeros(group_count, dtype=float)
    score_square_sum = np.zeros(group_count, dtype=float)
    outcome_counts = [
        np.zeros(model.kernels[group].size, dtype=int) for group in groups
    ]
    cdfs = [np.cumsum(model.kernels[group].probabilities) for group in groups]
    arm_seeds = rng.integers(
        0, np.iinfo(np.uint64).max, size=group_count, dtype=np.uint64
    )
    arm_rngs = [np.random.default_rng(seed) for seed in arm_seeds]

    def pull(group_index: int) -> None:
        outcome = int(
            np.searchsorted(
                cdfs[group_index], arm_rngs[group_index].random(), side="right"
            )
        )
        outcome = min(outcome, len(cdfs[group_index]) - 1)
        outcome_counts[group_index][outcome] += 1
        value = float(frozen_scores[group_index][outcome])
        pulls[group_index] += 1
        score_sum[group_index] += value
        score_square_sum[group_index] += value * value

    def standard_deviation(group_index: int) -> float:
        count = pulls[group_index]
        mean = score_sum[group_index] / count
        return float(
            np.sqrt(max(score_square_sum[group_index] / count - mean * mean, 0.0))
        )

    def index(group_index: int) -> float:
        count = pulls[group_index]
        return (
            (1.0 / group_count)
            / count
            * (standard_deviation(group_index) + 2.0 * beta / np.sqrt(count))
        )

    for group_index in range(group_count):
        pull(group_index)
        pull(group_index)
    heap = [(-index(group_index), group_index) for group_index in range(group_count)]
    heapq.heapify(heap)
    for _ in range(main_budget - 2 * group_count):
        _, group_index = heapq.heappop(heap)
        pull(group_index)
        heapq.heappush(heap, (-index(group_index), group_index))

    empirical_counts = {
        group: counts for group, counts in zip(groups, outcome_counts)
    }
    final_scales = np.asarray(
        [standard_deviation(group_index) for group_index in range(group_count)]
    )
    return pulls, empirical_counts, final_scales, score_range, float(beta)


def run_configuration(config_path: Path, overwrite: bool = False) -> Path:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repository_root = config_path.parent.parent
    source_root = Path(__file__).resolve().parents[2]
    configured_output = Path(config["output_directory"])
    output = configured_output if configured_output.is_absolute() else repository_root / configured_output
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Result directory already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copy2(config_path, output / "config.json")
    started = time.time()

    master_seed = int(config["master_seed"])
    minimum_main = int(config["minimum_main_per_group"])
    raw_rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    for environment_index, environment_config in enumerate(config["environments"]):
        model = make_environment(str(environment_config["name"]))
        methods = _methods_for(config, environment_config)
        alpha = float(environment_config["alpha"])
        environment_seed = int(environment_config.get("seed", master_seed))
        replications = int(environment_config.get("replications", config["replications"]))
        truth_bundle = compute_influences(model, alpha)
        if truth_bundle.solution.margin <= 1e-10:
            raise RuntimeError(f"{model.name} has a nonpositive categorical margin")
        if truth_bundle.tail_scales.sum() <= 0:
            raise RuntimeError(f"{model.name} has no first-order tail signal")
        group_count = model.group_count
        uniform_constant = group_count * float(np.sum(truth_bundle.tail_scales**2))
        oracle_constant = float(truth_bundle.tail_scales.sum() ** 2)
        diagnostics.append(
            {
                "environment": model.name,
                "alpha": alpha,
                "seed": environment_seed,
                "replications": replications,
                "groups": group_count,
                "group_mode": model.group_mode,
                "methods": list(methods),
                "truth": truth_bundle.solution.cvar,
                "margin": truth_bundle.solution.margin,
                "uniform_to_oracle": uniform_constant / oracle_constant,
                "covariance_score_relative_l2": float(
                    np.linalg.norm(truth_bundle.tail_scales - truth_bundle.no_cov_scales)
                    / max(np.linalg.norm(truth_bundle.tail_scales), 1e-15)
                ),
                "untied_factorization_max_abs_error": float(
                    np.max(
                        np.abs(
                            truth_bundle.tail_scales
                            - truth_bundle.reachability_scales
                            * truth_bundle.local_scales
                        )
                    )
                ),
            }
        )
        json_dump(output / "specs" / f"{model.name}.json", _environment_snapshot(model))

        for budget_unit, budget_value, total_budget, budget_per_group in _budgets_for(
            environment_config, group_count
        ):
            exploration = total_budget ** (-float(config["exploration_exponent"]))
            m = pilot_size(
                total_budget,
                group_count,
                float(config["pilot_multiplier"]),
                int(config["minimum_pilot_per_group"]),
                minimum_main,
            )
            learned_main_budget = total_budget - group_count * m

            for replication in range(replications):
                pilot_rng = _seed(environment_seed, 0, total_budget, replication, 0)
                pilot_sizes = {group: m for group in model.query_groups or ()}
                pilot_model = model_from_counts(model, sample_counts(model, pilot_sizes, pilot_rng))
                pilot_bundle = compute_influences(pilot_model, alpha)

                scores = {
                    "uniform": np.ones(group_count),
                    "occupancy": _occupancy_scores(model, truth_bundle.occupancy),
                    "mean_influence": truth_bundle.mean_scales,
                    "oracle_tail": truth_bundle.tail_scales,
                    "oracle_no_cov": truth_bundle.no_cov_scales,
                    "tis": pilot_bundle.tail_scales,
                    "tail_no_cov": pilot_bundle.no_cov_scales,
                    "learned_mean": pilot_bundle.mean_scales,
                    "reachability_only": truth_bundle.reachability_scales,
                    "local_only": truth_bundle.local_scales,
                    "learned_occupancy": _occupancy_scores(model, pilot_bundle.occupancy),
                    "tail_anchored": _anchored_score(
                        pilot_bundle.tail_scales,
                        _occupancy_scores(model, pilot_bundle.occupancy),
                    ),
                }
                counts_by_method: dict[str, np.ndarray] = {}
                weights_by_method: dict[str, np.ndarray] = {}
                for method in methods:
                    if method == "mc_ucb_frozen":
                        continue
                    if method == "uniform":
                        weights = np.full(group_count, 1.0 / group_count)
                        main_budget = total_budget
                    else:
                        weights = regularized_weights(scores[method], exploration)
                        main_budget = (
                            learned_main_budget if method in LEARNED_METHODS else total_budget
                        )
                    weights_by_method[method] = weights
                    counts_by_method[method] = integer_allocation(
                        main_budget, weights, minimum_main
                    )
                    if int(counts_by_method[method].sum()) != main_budget:
                        raise AssertionError("Integer allocation failed to exhaust budget")

                maximum = np.max(np.vstack(list(counts_by_method.values())), axis=0)
                main_rng = _seed(environment_seed, 0, total_budget, replication, 1)
                streams = coupled_streams(model, _group_mapping(model, maximum), main_rng)

                mc_empirical_counts: dict[tuple[int, int], np.ndarray] = {}
                mc_score_range = 0.0
                mc_beta = 0.0
                if "mc_ucb_frozen" in methods:
                    mc_rng = _seed(environment_seed, 0, total_budget, replication, 2)
                    (
                        mc_counts,
                        mc_empirical_counts,
                        mc_final_scales,
                        mc_score_range,
                        mc_beta,
                    ) = _mc_ucb_allocation(
                        model,
                        pilot_bundle.tail_values,
                        learned_main_budget,
                        mc_rng,
                    )
                    counts_by_method["mc_ucb_frozen"] = mc_counts
                    weights_by_method["mc_ucb_frozen"] = mc_counts / mc_counts.sum()
                    scores["mc_ucb_frozen"] = mc_final_scales

                for method in methods:
                    counts = counts_by_method[method]
                    empirical_counts = (
                        mc_empirical_counts
                        if method == "mc_ucb_frozen"
                        else counts_from_streams(
                            model, streams, _group_mapping(model, counts)
                        )
                    )
                    empirical_model = model_from_counts(model, empirical_counts)
                    estimate = solve_stop_loss(empirical_model, alpha).cvar
                    error = estimate - truth_bundle.solution.cvar
                    design_constant = total_budget * float(
                        np.sum(truth_bundle.tail_scales**2 / counts)
                    )
                    pilot_total = group_count * m if method in LEARNED_METHODS else 0
                    raw_rows.append(
                        {
                            "environment": model.name,
                            "alpha": alpha,
                            "budget_per_group": budget_per_group,
                            "budget_unit": budget_unit,
                            "budget_value": budget_value,
                            "total_budget": total_budget,
                            "replication": replication,
                            "method": method,
                            "truth": truth_bundle.solution.cvar,
                            "estimate": estimate,
                            "error": error,
                            "squared_error": error**2,
                            "pilot_per_group": 0 if pilot_total == 0 else m,
                            "pilot_total": pilot_total,
                            "main_total": int(counts.sum()),
                            "exploration": exploration,
                            "design_constant": design_constant,
                            "mc_ucb_score_range": mc_score_range if method == "mc_ucb_frozen" else "",
                            "mc_ucb_beta": mc_beta if method == "mc_ucb_frozen" else "",
                        }
                    )
                    for group_index, (group, count, weight, score) in enumerate(
                        zip(model.query_groups or (), counts, weights_by_method[method], scores[method])
                    ):
                        first, second = group
                        layer = first if model.group_mode == "untied_layer_state" else ""
                        state = second if model.group_mode == "untied_layer_state" else first
                        action = "" if model.group_mode == "untied_layer_state" else second
                        allocation_rows.append(
                            {
                                "environment": model.name,
                                "alpha": alpha,
                                "budget_per_group": budget_per_group,
                                "budget_unit": budget_unit,
                                "budget_value": budget_value,
                                "total_budget": total_budget,
                                "replication": replication,
                                "method": method,
                                "group_index": group_index,
                                "group_mode": model.group_mode,
                                "group_first": first,
                                "group_second": second,
                                "layer": layer,
                                "state": state,
                                "action": action,
                                "score": float(score),
                                "weight": float(weight),
                                "main_count": int(count),
                                "true_tail_scale": float(truth_bundle.tail_scales[group_index]),
                                "pilot_tail_scale": float(pilot_bundle.tail_scales[group_index]),
                                "pilot_no_cov_scale": float(pilot_bundle.no_cov_scales[group_index]),
                                "pilot_mean_scale": float(pilot_bundle.mean_scales[group_index]),
                                "true_reachability": float(truth_bundle.reachability_scales[group_index]),
                                "true_local_scale": float(truth_bundle.local_scales[group_index]),
                            }
                        )

    summary_rows, paired_rows = summarize(raw_rows)
    write_csv(output / "raw.csv", raw_rows)
    write_csv(output / "summary.csv", summary_rows)
    write_csv(output / "paired.csv", paired_rows)
    write_csv(output / "allocations.csv", allocation_rows)
    write_latex_table(output / "table.tex", summary_rows)
    write_figure(output / "mse", summary_rows)
    json_dump(output / "diagnostics.json", diagnostics)

    result_files = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "artifact_version": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "runtime_seconds": time.time() - started,
        "configuration": config,
        "diagnostics": diagnostics,
        "source_root_from_result": os.path.relpath(source_root, output),
        "source_files": _source_snapshot(source_root),
        "files": {
            str(path.relative_to(output)): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in result_files
        },
    }
    json_dump(output / "manifest.json", manifest)
    return output
