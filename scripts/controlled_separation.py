#!/usr/bin/env python3
"""Controlled separation experiment.

Protocol frozen in configs/controlled_separation.json before this
script first ran. A one-step fixed stochastic policy chooses uniformly
among G = 10 independently queryable action kernels. Laws A and B share
mean 1/2 and variance 1/36; mixing parameter t moves every kernel from
the common average law Pbar (t = 0) to the separated endpoint (t = 1,
kernel 0 = A, others = B) while visitation, every kernel's mean and
variance, and the root return law stay unchanged at every t. Only the
lower-tail influence concentrates.

Stage 1 (analytical verification, must pass before any simulation is
read): the engine must reproduce the frozen quantities - at t = 1 the
root quantile 1/3, CVaR .3, margin .09, and influence confined to
kernel 0; uniform/oracle leading-variance ratios approximately 1,
1.300, 10 at t = 0, .5, 1; means, variances, and the root law constant
in t.

Stage 2 (simulation): all frozen methods, budgets, and replications.
Learned methods pay their pilot; complete rollouts consume one
environment sample per draw (the known uniform action policy is free);
no method is told which kernel is exceptional, and empirical laws use
the full outcome support. Writes
results/controlled_separation/{verification.json,summary.csv,
underallocation.csv,paired.csv}.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tis.allocation import integer_allocation, pilot_size, regularized_weights  # noqa: E402
from tis.engine import compute_influences, solve_stop_loss  # noqa: E402
from tis.llm_study.runner import _anchored_score, _empirical_cvar, _mapping, _occupancy_scores, _rng  # noqa: E402
from tis.model import (  # noqa: E402
    FiniteHorizonModel, Kernel, counts_from_streams, coupled_streams,
    model_from_counts, sample_counts,
)

CONFIG = json.loads((ROOT / "configs/controlled_separation.json").read_text())
G = int(CONFIG["construction"]["G"])
ALPHA = float(CONFIG["construction"]["alpha"])
T_VALUES = [float(x) for x in CONFIG["construction"]["t_values"]]
BUDGETS = [int(b) for b in CONFIG["budgets_per_kernel"]]
REPLICATIONS = int(CONFIG["replications"])
MASTER_SEED = int(CONFIG["master_seed"])
PILOT = CONFIG["pilot"]

SUPPORT = np.array([0.0, 1.0 / 3.0, 5.0 / 9.0, 2.0 / 3.0])
LAW_A = np.array([0.1, 0.0, 0.9, 0.0])
LAW_B = np.array([0.0, 0.5, 0.0, 0.5])
PBAR = (LAW_A + (G - 1) * LAW_B) / G
GRID = np.array([0.0, 1.0 / 3.0, 5.0 / 9.0, 2.0 / 3.0, 1.0])


def build_model(t: float) -> FiniteHorizonModel:
    """One root state, G actions with uniform policy, horizon 1."""
    policy = np.zeros((2, 1, G))
    policy[1, 0, :] = 1.0 / G
    kernels, groups = {}, []
    for g in range(G):
        endpoint = LAW_A if g == 0 else LAW_B
        probs = (1.0 - t) * PBAR + t * endpoint
        kernels[(0, g)] = Kernel(rewards=SUPPORT.copy(),
                                 next_states=np.zeros(len(SUPPORT), dtype=int),
                                 probabilities=probs.copy())
        groups.append((0, g))
    return FiniteHorizonModel(
        name=f"controlled_separation_t{t:g}", horizon=1, initial_state=0,
        grid=GRID.copy(), policy=policy, kernels=kernels,
        query_groups=tuple(groups),
        metadata={"t": t, "construction": "controlled separation"})


def verify() -> dict:
    report = {}
    root_law_ref = None
    for t in T_VALUES:
        model = build_model(t)
        truth = compute_influences(model, ALPHA)
        sigma = np.asarray(truth.tail_scales)
        S = float(sigma.sum())
        ratio = (G * float((sigma ** 2).sum()) / S ** 2) if S > 0 else float("nan")
        means = [float(k.probabilities @ k.rewards) for k in model.kernels.values()]
        variances = [float(k.probabilities @ (k.rewards - m) ** 2)
                     for k, m in zip(model.kernels.values(), means)]
        root_law = sum(model.kernels[(0, g)].probabilities for g in range(G)) / G
        if root_law_ref is None:
            root_law_ref = root_law
        report[f"t={t:g}"] = {
            "cvar": float(truth.solution.cvar),
            "margin": float(truth.solution.margin),
            "uniform_over_oracle": ratio,
            "influence_shares": [float(x) for x in (sigma / S if S > 0 else sigma)],
            "means_max_dev_from_half": float(max(abs(m - 0.5) for m in means)),
            "variances_max_dev": float(max(abs(v - 1.0 / 36.0) for v in variances)),
            "root_law_max_dev_from_t0": float(np.max(np.abs(root_law - root_law_ref))),
        }
    r = report
    checks = {
        "t1_cvar_.3": abs(r["t=1"]["cvar"] - 0.3) < 1e-12,
        "t1_margin_.09": abs(r["t=1"]["margin"] - 0.09) < 1e-12,
        "t1_influence_only_kernel0": max(r["t=1"]["influence_shares"][1:]) < 1e-12,
        "ratio_t0_1": abs(r["t=0"]["uniform_over_oracle"] - 1.0) < 1e-9,
        "ratio_t05_1.300": abs(r["t=0.5"]["uniform_over_oracle"] - 1.300) < 5e-3,
        "ratio_t1_10": abs(r["t=1"]["uniform_over_oracle"] - 10.0) < 1e-9,
        "means_invariant": max(r[k]["means_max_dev_from_half"] for k in r) < 1e-12,
        "variances_invariant": max(r[k]["variances_max_dev"] for k in r) < 1e-12,
        "root_law_invariant": max(r[k]["root_law_max_dev_from_t0"] for k in r) < 1e-12,
    }
    report["checks"] = checks
    report["all_passed"] = all(checks.values())
    return report


def simulate(out_dir: Path) -> None:
    minimum_main = int(PILOT["minimum_main_per_group"])
    rows, under_rows, raw_rows = [], [], []
    paired: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for t in T_VALUES:
        model = build_model(t)
        truth = compute_influences(model, ALPHA)
        C = float(truth.solution.cvar)
        for budget in BUDGETS:
            total = G * budget
            exploration = total ** (-float(PILOT["exploration_exponent"]))
            m_pilot = pilot_size(total, G, float(PILOT["multiplier"]),
                                 int(PILOT["minimum_pilot_per_group"]), minimum_main)
            main_total = total - G * m_pilot
            errors = defaultdict(list)
            under = []
            for rep in range(REPLICATIONS):
                pilot_counts = sample_counts(
                    model, {g: m_pilot for g in model.query_groups},
                    _rng(MASTER_SEED, 1, 0, total, int(round(10 * t)), rep, 0))
                pilot = compute_influences(model_from_counts(model, pilot_counts), ALPHA)
                occ = _occupancy_scores(model, pilot)
                scores = {
                    "uniform": np.ones(G),
                    "learned_occupancy": occ,
                    "learned_mean": np.asarray(pilot.mean_scales),
                    "tis": np.asarray(pilot.tail_scales),
                    "tis_anchored": _anchored_score(np.asarray(pilot.tail_scales), occ),
                    "oracle_tail": np.asarray(truth.tail_scales),
                }
                counts_by = {}
                for method, score in scores.items():
                    if method == "uniform":
                        weights = np.full(G, 1.0 / G)
                        budget_for = total
                    else:
                        weights = regularized_weights(score, exploration)
                        budget_for = main_total if method != "oracle_tail" else total
                    counts_by[method] = integer_allocation(budget_for, weights, minimum_main)
                maximum = np.max(np.vstack(list(counts_by.values())), axis=0)
                streams = coupled_streams(
                    model, _mapping(model, maximum),
                    _rng(MASTER_SEED, 1, 0, total, int(round(10 * t)), rep, 1))
                for method, counts in counts_by.items():
                    empirical = counts_from_streams(model, streams, _mapping(model, counts))
                    est = solve_stop_loss(model_from_counts(model, empirical), ALPHA).cvar
                    err = est - C
                    errors[method].append(err)
                    paired[(t, budget, method)]["err"].append(err)
                    raw_rows.append({"t": t, "budget_per_kernel": budget,
                                     "replication": rep, "method": method,
                                     "truth": C, "error": err})
                under.append(counts_by["tis"][0] / counts_by["tis"].sum())
                returns, _ = _rollout(model, total,
                                      _rng(MASTER_SEED, 1, 0, total, int(round(10 * t)), rep, 6))
                err = _empirical_cvar(returns, ALPHA) - C
                errors["complete_rollout"].append(err)
                paired[(t, budget, "complete_rollout")]["err"].append(err)
                raw_rows.append({"t": t, "budget_per_kernel": budget,
                                 "replication": rep, "method": "complete_rollout",
                                 "truth": C, "error": err})
            for method, errs in errors.items():
                arr = np.asarray(errs)
                sq = arr ** 2
                rows.append({
                    "t": t, "budget_per_kernel": budget, "method": method,
                    "replications": len(arr), "truth": C,
                    "mse": float(sq.mean()),
                    "mse_se": float(sq.std(ddof=1) / np.sqrt(len(sq))),
                    "bias": float(arr.mean()),
                })
            u = np.asarray(under)
            under_rows.append({
                "t": t, "budget_per_kernel": budget,
                "tis_share_of_kernel0_median": float(np.median(u)),
                "tis_share_of_kernel0_p10": float(np.percentile(u, 10)),
                "oracle_share_of_kernel0": float(
                    (truth.tail_scales / truth.tail_scales.sum())[0])
                if truth.tail_scales.sum() > 0 else 1.0 / G,
            })
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with (out_dir / "underallocation.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(under_rows[0]))
        w.writeheader(); w.writerows(under_rows)
    pair_rows = []
    for (t, budget, method), d in paired.items():
        if method == "learned_occupancy":
            continue
        base = paired[(t, budget, "learned_occupancy")]["err"]
        diff = np.asarray(d["err"]) ** 2 - np.asarray(base) ** 2
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        pair_rows.append({"t": t, "budget_per_kernel": budget, "method": method,
                          "sqerr_minus_occupancy_mean": float(diff.mean()),
                          "se": float(se),
                          "z": float(diff.mean() / se) if se > 0 else 0.0})
    with (out_dir / "paired.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0]))
        w.writeheader(); w.writerows(pair_rows)
    # Per-replication errors are retained so paired and bootstrap
    # analyses reproduce from the released files.
    with (out_dir / "raw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
        w.writeheader(); w.writerows(raw_rows)


def _rollout(model, draws: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """One-step rollouts: each draw samples the uniform policy for free
    and one kernel outcome (one environment sample)."""
    actions = rng.integers(0, G, size=draws)
    returns = np.empty(draws)
    visits = np.zeros(G, dtype=int)
    for g in range(G):
        mask = actions == g
        n = int(mask.sum())
        visits[g] = n
        if n:
            kernel = model.kernels[(0, g)]
            outcomes = rng.choice(kernel.probabilities.size, size=n,
                                  p=kernel.probabilities)
            returns[mask] = kernel.rewards[outcomes]
    return returns, visits


def main() -> None:
    out_dir = ROOT / "results" / "controlled_separation"
    report = verify()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verification.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k in ("checks", "all_passed")}, indent=1))
    if not report["all_passed"]:
        print("ANALYTICAL VERIFICATION FAILED - no simulation run")
        sys.exit(1)
    if "--verify-only" in sys.argv:
        return
    simulate(out_dir)
    print(out_dir / "summary.csv")


if __name__ == "__main__":
    main()
