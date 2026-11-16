#!/usr/bin/env python3
"""Estimator versus allocation on CliffWalking (Appendix F.2).

Configuration: configs/rebuttal_cliffwalking_attribution.json. For every
budget and replication:

  rollouts      floor(N/H) complete trajectories; arm A estimates CVaR from
                their returns, arm B fits the categorical model to exactly
                the same transitions (pooled per kernel) and solves it, and
                B' repeats B with the sensitivity completion of empty rows;
  conditional   uniform queries (no pilot) and the learned designs, which
                share one charged pilot and coupled per-kernel main streams;
                all use the categorical Bellman evaluator.

Usage:
  python scripts/rebuttal_cliffwalking_attribution.py run [processes]
  python scripts/rebuttal_cliffwalking_attribution.py analyze
Writes results/rebuttal_cliffwalking_attribution/{raw.csv,summary.csv,analysis.json}.
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

from tis.allocation import integer_allocation, pilot_size, regularized_weights  # noqa: E402
from tis.engine import compute_influences, solve_stop_loss  # noqa: E402
from tis.environments import make_environment  # noqa: E402
from tis.llm_study.runner import _empirical_cvar, _mapping  # noqa: E402
from tis.model import counts_from_streams, coupled_streams, model_from_counts, sample_counts  # noqa: E402
from tis.runner import _anchored_score, _occupancy_scores  # noqa: E402

CONFIG = json.loads((ROOT / "configs/rebuttal_cliffwalking_attribution.json").read_text())
ALPHA = float(CONFIG["alpha"])
PILOT = CONFIG["pilot"]
SEED = int(CONFIG["master_seed"])
OUT = ROOT / CONFIG["output_directory"]
CONDITIONAL = ("uniform", "learned_occupancy", "learned_mean", "tis", "tis_anchored",
               "occ_plus_mean", "model_accuracy", "oracle_tail")


def _rng(budget: int, replication: int, stage: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SEED, budget, replication, stage]))


def model_accuracy_scores(model, pilot_counts) -> np.ndarray:
    """Neyman scale for estimating each kernel's law: sqrt(1 - sum_w p_hat(w)^2)."""
    scores = []
    for group in model.query_groups:
        p = np.asarray(pilot_counts[group], dtype=float)
        p = p / p.sum()
        scores.append(np.sqrt(max(1.0 - float(np.sum(p ** 2)), 0.0)))
    return np.asarray(scores)


def rollout_with_counts(model, runs: int, rng: np.random.Generator):
    """Complete trajectories through the true laws, recording every transition.

    Returns the per-run returns and, for every query group, the counts of its
    declared outcomes among the sampled transitions.
    """
    counts = {g: np.zeros(model.kernels[g].probabilities.size, dtype=int) for g in model.query_groups}
    state = np.full(runs, model.initial_state, dtype=int)
    totals = np.zeros(runs)
    for step in range(model.horizon):
        h = model.horizon - step
        successor = np.empty_like(state)
        for current in np.unique(state):
            mask = state == current
            probs = np.asarray(model.policy[h, int(current)], dtype=float)
            actions = rng.choice(probs.size, size=int(mask.sum()), p=probs / probs.sum())
            rewards = np.empty(actions.size)
            nexts = np.empty(actions.size, dtype=int)
            for action in np.unique(actions):
                amask = actions == action
                group = (int(current), int(action))
                kernel = model.kernels[group]
                outcomes = rng.choice(kernel.probabilities.size, size=int(amask.sum()), p=kernel.probabilities)
                rewards[amask] = kernel.rewards[outcomes]
                nexts[amask] = kernel.next_states[outcomes]
                if group in counts:
                    counts[group] += np.bincount(outcomes, minlength=counts[group].size)
            totals[mask] += rewards
            successor[mask] = nexts
        state = successor
    return totals, counts


def complete_rows(model, counts, rule: str):
    """Fill kernels with no observed transition using only their declared support."""
    filled, empty = {}, []
    for group, c in counts.items():
        if c.sum() > 0:
            filled[group] = c
            continue
        empty.append(group)
        kernel = model.kernels[group]
        if rule == "uniform":
            filled[group] = np.ones_like(c)
        else:
            order = np.lexsort((kernel.next_states, kernel.rewards))
            point = np.zeros_like(c)
            point[order[0]] = 1
            filled[group] = point
    return filled, empty


def replication(args):
    budget, rep = args
    model = make_environment(CONFIG["environment"])
    truth = compute_influences(model, ALPHA)
    C = float(truth.solution.cvar)
    groups = model.group_count
    total = groups * budget
    exploration = total ** (-float(PILOT["exploration_exponent"]))
    minimum_main = int(PILOT["minimum_main_per_group"])
    m = pilot_size(total, groups, float(PILOT["multiplier"]), int(PILOT["minimum_pilot_per_group"]), minimum_main)
    main_total = total - groups * m
    occupancy_truth = _occupancy_scores(model, truth.occupancy)

    pilot_counts = sample_counts(model, {g: m for g in model.query_groups}, _rng(budget, rep, 0))
    pilot = compute_influences(model_from_counts(model, pilot_counts), ALPHA)
    occ = _occupancy_scores(model, pilot.occupancy)
    w_occ = regularized_weights(occ, exploration)
    scores = {
        "learned_occupancy": occ,
        "learned_mean": np.asarray(pilot.mean_scales),
        "tis": np.asarray(pilot.tail_scales),
        "tis_anchored": _anchored_score(np.asarray(pilot.tail_scales), occ),
        "model_accuracy": model_accuracy_scores(model, pilot_counts),
    }
    allocations = {
        "uniform": integer_allocation(total, np.full(groups, 1.0 / groups), minimum_main),
        "oracle_tail": integer_allocation(total, regularized_weights(np.asarray(truth.tail_scales), exploration), minimum_main),
        "occ_plus_mean": integer_allocation(
            main_total, 0.5 * regularized_weights(scores["learned_mean"], exploration) + 0.5 * w_occ, minimum_main),
    }
    for method, score in scores.items():
        allocations[method] = integer_allocation(main_total, regularized_weights(score, exploration), minimum_main)
    maximum = np.max(np.vstack([allocations[k] for k in CONDITIONAL]), axis=0)
    streams = coupled_streams(model, _mapping(model, maximum), _rng(budget, rep, 1))
    rows = []
    for method in CONDITIONAL:
        empirical = counts_from_streams(model, streams, _mapping(model, allocations[method]))
        estimate = solve_stop_loss(model_from_counts(model, empirical), ALPHA).cvar
        rows.append({"budget": budget, "replication": rep, "arm": method, "error": estimate - C,
                     "charged_queries": total, "empty_rows": 0, "empty_row_visit_mass": 0.0})

    runs = total // model.horizon
    returns, counts = rollout_with_counts(model, runs, _rng(budget, rep, 6))
    used = runs * model.horizon
    rows.append({"budget": budget, "replication": rep, "arm": "rollout_empirical",
                 "error": _empirical_cvar(returns, ALPHA) - C, "charged_queries": used,
                 "empty_rows": 0, "empty_row_visit_mass": 0.0})
    for arm, rule in (("rollout_model", "uniform"), ("rollout_model_pointmass", "pointmass")):
        filled, empty = complete_rows(model, counts, rule)
        estimate = solve_stop_loss(model_from_counts(model, filled), ALPHA).cvar
        index = {g: i for i, g in enumerate(model.query_groups)}
        rows.append({"budget": budget, "replication": rep, "arm": arm, "error": estimate - C,
                     "charged_queries": used, "empty_rows": len(empty),
                     "empty_row_visit_mass": float(sum(occupancy_truth[index[g]] for g in empty))})
    return rows


def run(processes: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(b, r) for b in CONFIG["budgets_per_kernel"] for r in range(int(CONFIG["replications"]))]
    rows = []
    with ProcessPoolExecutor(max_workers=processes) as pool:
        for chunk in pool.map(replication, jobs, chunksize=4):
            rows.extend(chunk)
    rows.sort(key=lambda r: (r["budget"], r["replication"], r["arm"]))
    with (OUT / "raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT / 'raw.csv'}")


def _load():
    errors = defaultdict(dict)
    extra = defaultdict(list)
    for row in csv.DictReader((OUT / "raw.csv").open()):
        key = (int(row["budget"]), row["arm"])
        errors[key][int(row["replication"])] = float(row["error"])
        extra[key].append((int(row["empty_rows"]), float(row["empty_row_visit_mass"]), int(row["charged_queries"])))
    return errors, extra


def paired(errors, budget, method, reference, reps=None, resamples=0, rng=None):
    a, r = errors[(budget, method)], errors[(budget, reference)]
    keys = sorted(set(a) & set(r)) if reps is None else [k for k in sorted(set(a) & set(r)) if reps[0] <= k < reps[1]]
    sa = np.array([a[k] ** 2 for k in keys])
    sr = np.array([r[k] ** 2 for k in keys])
    d = sa - sr
    out = {"ratio": float(sa.mean() / sr.mean()), "z": float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size))),
           "mse_difference": float(d.mean())}
    if resamples:
        idx = rng.integers(0, d.size, size=(resamples, d.size))
        ratios = sa[idx].mean(axis=1) / sr[idx].mean(axis=1)
        centred = d[idx].mean(axis=1) - d.mean()
        out["ratio_ci95"] = [float(np.quantile(ratios, .025)), float(np.quantile(ratios, .975))]
        out["bootstrap_p_two_sided"] = float(np.mean(np.abs(centred) >= abs(d.mean())))
    return out


def unpaired(errors, budget, method, reference):
    """Contrast between arms on independent streams (rollouts versus conditional queries)."""
    sa = np.square(list(errors[(budget, method)].values()))
    sr = np.square(list(errors[(budget, reference)].values()))
    se = np.sqrt(sa.var(ddof=1) / sa.size + sr.var(ddof=1) / sr.size)
    return {"ratio": float(sa.mean() / sr.mean()), "z": float((sa.mean() - sr.mean()) / se),
            "mse_difference": float(sa.mean() - sr.mean()), "paired": False}


def analyze() -> None:
    errors, extra = _load()
    arms = sorted({k[1] for k in errors})
    budgets = sorted({k[0] for k in errors})
    summary = []
    for b in budgets:
        uniform = np.mean(np.square(list(errors[(b, "uniform")].values())))
        for arm in arms:
            e = np.array(list(errors[(b, arm)].values()))
            sq = e ** 2
            empty = np.array([x[0] for x in extra[(b, arm)]])
            mass = np.array([x[1] for x in extra[(b, arm)]])
            summary.append({"budget": b, "arm": arm, "replications": e.size, "mse": float(sq.mean()),
                            "mse_se": float(sq.std(ddof=1) / np.sqrt(sq.size)), "bias": float(e.mean()),
                            "variance": float(e.var(ddof=1)), "mse_over_uniform": float(sq.mean() / uniform),
                            "mean_empty_rows": float(empty.mean()), "mean_empty_row_visit_mass": float(mass.mean()),
                            "charged_queries": int(np.mean([x[2] for x in extra[(b, arm)]]))})
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    rng = np.random.default_rng(SEED + 1)
    p1 = CONFIG["primary"]["P1"]
    analysis = {"P1": paired(errors, p1["budget"], p1["method"], p1["reference"],
                             resamples=int(CONFIG["bootstrap_resamples"]), rng=rng),
                "contrasts": {}, "batches": {}}
    contrasts = [("rollout_model", "rollout_empirical"), ("rollout_model_pointmass", "rollout_empirical"),
                 ("rollout_model", "rollout_model_pointmass"),
                 ("tis", "learned_mean"), ("tis", "learned_occupancy"), ("tis", "model_accuracy"),
                 ("tis_anchored", "occ_plus_mean"), ("tis", "uniform"), ("learned_mean", "uniform")]
    # Conditional-query arms and rollouts use independent streams; those
    # contrasts are computed unpaired below, all others are paired.
    for b in budgets:
        for method, reference in contrasts:
            analysis["contrasts"][f"{method}_vs_{reference}|{b}"] = paired(errors, b, method, reference)
        for method in ("tis", "tis_anchored", "learned_mean", "uniform"):
            analysis["contrasts"][f"{method}_vs_rollout_model|{b}"] = unpaired(errors, b, method, "rollout_model")
        for batch in CONFIG["batches"]:
            analysis["batches"][f"tis_vs_learned_mean|{b}|{batch[0]}-{batch[1]}"] = paired(
                errors, b, "tis", "learned_mean", reps=batch)
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=1) + "\n")
    print("P1 (TIS vs learned mean, b=400):", json.dumps(analysis["P1"]))
    for row in summary:
        print(f"b{row['budget']:<4} {row['arm']:24s} mse/unif {row['mse_over_uniform']:.3f}  bias {row['bias']:+.4f}"
              f"  empty rows {row['mean_empty_rows']:.1f} (visit mass {row['mean_empty_row_visit_mass']:.4f})")
    for key, value in analysis["contrasts"].items():
        print(f"  {key:45s} ratio {value['ratio']:.3f}  z {value['z']:+.1f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif sys.argv[1] == "analyze":
        analyze()
    else:
        raise SystemExit("usage: rebuttal_cliffwalking_attribution.py run [processes] | analyze")
