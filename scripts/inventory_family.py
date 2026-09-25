#!/usr/bin/env python3
"""Inventory disruption family: construction, verification gate, simulation.

An 18-case family around the v1 seasonal inventory instance: base
disruption probabilities {0.01, 0.04, 0.12} (capacity keeps the v1
additive convention, p_capacity = p_base + 0.02), disruption losses
{1, 2, 3} units, and two frozen policies (the v1 seasonal base-stock
targets, and the same targets plus one unit capped at capacity). The
reward rule, quantization, demand laws, horizon, and grid are the v1
ones, unchanged; the loss parameter changes how many remaining units a
disruption removes (transition law), and the policy variant changes the
order-up-to targets.

Stage 1 (``gate``): population-only enumeration for every case - exact
CVaR, categorical quantile margin, uniform-to-oracle leading-variance
ratio, reachable block count, and pairwise distinctness of the kernel
laws - written to results/inventory_family/verification.json before any
method comparison. Cases with near-zero margins are reported as such,
never dropped.

Stage 2 (``simulate``): all methods of the frozen declaration on every
case (no filtering by outcome), budgets per retained block from the
declaration, charged pilots, the standard floor and rounding, coupled
conditional-query streams, and an independent rollout stream.

Usage: python scripts/inventory_family.py gate | simulate [case_index]
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tis.allocation import integer_allocation, pilot_size, regularized_weights  # noqa: E402
from tis.engine import compute_influences, solve_stop_loss  # noqa: E402
from tis.environments import (  # noqa: E402
    INVENTORY_BASE_STOCK, INVENTORY_CAPACITY, INVENTORY_POISSON_MEAN,
    _censored_poisson_probabilities, _inventory_reward, _kernel,
)
from tis.llm_study.runner import _empirical_cvar, _mapping, _rng  # noqa: E402
from tis.runner import _anchored_score, _occupancy_scores  # noqa: E402
from tis.model import (  # noqa: E402
    FiniteHorizonModel, counts_from_streams, coupled_streams,
    model_from_counts, sample_counts,
)

DECLARATION = json.loads((ROOT / "configs/inventory_family.json").read_text())
BASE_PROBS = DECLARATION["base_disruption_probabilities"]
LOSSES = DECLARATION["disruption_losses"]
POLICIES = DECLARATION["policies"]
ALPHA = float(DECLARATION["alpha"])
BUDGETS = [int(b) for b in DECLARATION["budgets_per_block"]]
REPLICATIONS = int(DECLARATION["replications"])
MASTER_SEED = int(DECLARATION["master_seed"])
PILOT = DECLARATION["pilot"]
METHODS = tuple(DECLARATION["methods"])
MIXTURES = {"occ_plus_uniform", "occ_plus_mean"}
LEARNED = {"learned_occupancy", "learned_mean", "tis", "tis_anchored"} | MIXTURES

CASES = [
    {"index": i, "p_base": p, "loss": loss, "policy": pol}
    for i, (pol, p, loss) in enumerate(
        (pol, p, loss) for pol in POLICIES for p in BASE_PROBS for loss in LOSSES
    )
]


def build_case(p_base: float, loss: int, policy: str) -> FiniteHorizonModel:
    horizon, states = 8, INVENTORY_CAPACITY + 1
    pol = np.zeros((horizon + 1, states, 1), dtype=float)
    pol[1:, :, 0] = 1.0
    kernels = {}
    for h in range(1, horizon + 1):
        season = horizon - h
        target = INVENTORY_BASE_STOCK[season]
        if policy == "plus_one":
            target = min(target + 1, INVENTORY_CAPACITY)
        demand_probabilities = _censored_poisson_probabilities(
            INVENTORY_POISSON_MEAN[season])
        for state in range(states):
            order = min(INVENTORY_CAPACITY - state, max(0, target - state))
            available = state + order
            disruption_probability = (
                p_base + 0.02 if available == INVENTORY_CAPACITY else p_base)
            outcomes = []
            for demand, demand_probability in enumerate(demand_probabilities):
                sales = min(available, demand)
                lost = max(demand - available, 0)
                ending = max(available - demand, 0)
                for disrupted, event_probability in (
                        (False, 1.0 - disruption_probability),
                        (True, disruption_probability)):
                    next_state = max(ending - loss * int(disrupted), 0)
                    reward = _inventory_reward(sales, order, ending, lost, disrupted)
                    outcomes.append(
                        (reward, next_state,
                         float(demand_probability * event_probability)))
            kernels[(h, state)] = _kernel(outcomes)
    reachable = {horizon: {0}}
    for h in range(horizon, 1, -1):
        nxt = set()
        for state in reachable[h]:
            k = kernels[(h, state)]
            nxt.update(int(s) for s, pr in zip(k.next_states, k.probabilities) if pr > 0)
        reachable[h - 1] = nxt
    groups = tuple((h, s) for h in range(1, horizon + 1)
                   for s in sorted(reachable.get(h, set())))
    return FiniteHorizonModel(
        name=f"inventory_family_p{p_base:g}_L{loss}_{policy}",
        horizon=horizon, initial_state=0,
        grid=np.arange(0.0, horizon + 0.25, 0.5),
        policy=pol, kernels=kernels, query_groups=groups,
        group_mode="untied_layer_state",
        metadata={"p_base": p_base, "loss": loss, "policy": policy})


def _law_hash(model: FiniteHorizonModel) -> str:
    h = hashlib.sha256()
    for g in model.query_groups or ():
        k = model.kernels[g]
        h.update(np.ascontiguousarray(k.rewards).tobytes())
        h.update(np.ascontiguousarray(k.next_states).tobytes())
        h.update(np.round(np.ascontiguousarray(k.probabilities), 15).tobytes())
    return h.hexdigest()


def gate() -> None:
    out_dir = ROOT / "results" / "inventory_family"
    out_dir.mkdir(parents=True, exist_ok=True)
    report, hashes = [], {}
    for case in CASES:
        model = build_case(case["p_base"], case["loss"], case["policy"])
        truth = compute_influences(model, ALPHA)
        sigma = np.asarray(truth.tail_scales)
        S = float(sigma.sum())
        ratio = (model.group_count * float((sigma ** 2).sum()) / S ** 2
                 if S > 0 else None)
        hashes[case["index"]] = _law_hash(model)
        report.append({
            **case,
            "blocks": model.group_count,
            "cvar": float(truth.solution.cvar),
            "margin": float(truth.solution.margin),
            "near_zero_margin": bool(truth.solution.margin < 1e-6),
            "uniform_to_oracle": ratio,
            "zero_tail_signal": S <= 0,
        })
    n = len(CASES)
    duplicates = [(i, j) for i in range(n) for j in range(i + 1, n)
                  if hashes[i] == hashes[j]]
    verification = {
        "cases": report,
        "pairwise_identical_laws": duplicates,
        "all_laws_distinct": not duplicates,
    }
    (out_dir / "verification.json").write_text(json.dumps(verification, indent=1) + "\n")
    for r in report:
        print(f"case {r['index']:2} p={r['p_base']:<5} L={r['loss']} {r['policy']:<9} "
              f"blocks={r['blocks']:2} cvar={r['cvar']:.5f} margin={r['margin']:.5f}"
              f"{' NEAR-ZERO' if r['near_zero_margin'] else ''} "
              f"unif/oracle={r['uniform_to_oracle']:.3f}")
    print("all laws distinct:", verification["all_laws_distinct"])


def simulate(only_case: int | None = None) -> None:
    out_dir = ROOT / "results" / "inventory_family"
    out_dir.mkdir(parents=True, exist_ok=True)
    minimum_main = int(PILOT["minimum_main_per_group"])
    for case in CASES:
        if only_case is not None and case["index"] != only_case:
            continue
        model = build_case(case["p_base"], case["loss"], case["policy"])
        truth = compute_influences(model, ALPHA)
        C = float(truth.solution.cvar)
        groups = model.group_count
        rows = []
        raw = []
        for budget in BUDGETS:
            total = groups * budget
            exploration = total ** (-float(PILOT["exploration_exponent"]))
            m = pilot_size(total, groups, float(PILOT["multiplier"]),
                           int(PILOT["minimum_pilot_per_group"]), minimum_main)
            main_total = total - groups * m
            errors = defaultdict(list)
            for rep in range(REPLICATIONS):
                pilot_counts = sample_counts(
                    model, {g: m for g in model.query_groups},
                    _rng(MASTER_SEED, model.horizon, 0, total, case["index"], rep, 0))
                pilot = compute_influences(model_from_counts(model, pilot_counts), ALPHA)
                occ = _occupancy_scores(model, pilot.occupancy)
                w_occ = regularized_weights(occ, exploration)
                base_scores = {
                    "learned_occupancy": occ,
                    "learned_mean": np.asarray(pilot.mean_scales),
                    "tis": np.asarray(pilot.tail_scales),
                    "tis_anchored": _anchored_score(np.asarray(pilot.tail_scales), occ),
                    "oracle_tail": np.asarray(truth.tail_scales),
                }
                counts_by = {}
                for meth in METHODS:
                    if meth == "complete_rollout":
                        continue
                    if meth == "uniform":
                        weights = np.full(groups, 1.0 / groups)
                        budget_for = total
                    elif meth == "occ_plus_uniform":
                        weights = 0.5 * np.full(groups, 1.0 / groups) + 0.5 * w_occ
                        budget_for = main_total
                    elif meth == "occ_plus_mean":
                        w_mean = regularized_weights(base_scores["learned_mean"], exploration)
                        weights = 0.5 * w_mean + 0.5 * w_occ
                        budget_for = main_total
                    else:
                        weights = regularized_weights(base_scores[meth], exploration)
                        budget_for = main_total if meth in LEARNED else total
                    counts_by[meth] = integer_allocation(budget_for, weights, minimum_main)
                maximum = np.max(np.vstack(list(counts_by.values())), axis=0)
                streams = coupled_streams(
                    model, _mapping(model, maximum),
                    _rng(MASTER_SEED, model.horizon, 0, total, case["index"], rep, 1))
                for meth, counts in counts_by.items():
                    empirical = counts_from_streams(model, streams, _mapping(model, counts))
                    est = solve_stop_loss(model_from_counts(model, empirical), ALPHA).cvar
                    err = est - C
                    errors[meth].append(err)
                    raw.append({"case": case["index"], "budget_per_block": budget,
                                "replication": rep, "method": meth,
                                "truth": C, "error": err})
                if "complete_rollout" in METHODS:
                    from tabular_rollout import rollout_returns
                    returns = rollout_returns(
                        model, total // model.horizon,
                        _rng(MASTER_SEED, model.horizon, 0, total, case["index"], rep, 6))
                    err = _empirical_cvar(returns, ALPHA) - C
                    errors["complete_rollout"].append(err)
                    raw.append({"case": case["index"], "budget_per_block": budget,
                                "replication": rep, "method": "complete_rollout",
                                "truth": C, "error": err})
            for meth, errs in errors.items():
                arr = np.asarray(errs)
                sq = arr ** 2
                rows.append({
                    **{k: case[k] for k in ("index", "p_base", "loss", "policy")},
                    "budget_per_block": budget, "method": meth,
                    "replications": len(arr), "truth": C,
                    "mse": float(sq.mean()),
                    "mse_se": float(sq.std(ddof=1) / np.sqrt(len(sq))),
                    "bias": float(arr.mean()),
                })
        import csv
        tag = f"case{case['index']:02d}"
        with (out_dir / f"summary_{tag}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        with (out_dir / f"raw_{tag}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(raw[0]))
            w.writeheader(); w.writerows(raw)
        print(f"case {case['index']} done: {out_dir}/summary_{tag}.csv")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "scripts"))
    if sys.argv[1] == "gate":
        gate()
    elif sys.argv[1] == "simulate":
        simulate(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        raise SystemExit("usage: inventory_family.py gate|simulate [case]")
