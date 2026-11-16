#!/usr/bin/env python3
"""Lower-tail profit of the inventory disruption family (Appendices F.3, F.4, F.6).

Configuration: configs/rebuttal_inventory_profit.json (alpha = .1); the
configs rebuttal_inventory_profit_alpha*.json give the other tail levels.
The 18 cases keep their
dynamics; each period earns raw modelled profit (no category conversion, no
added disruption penalty), encoded exactly as r = (p + 12)/36 on the grid
{j/72}. Errors are reported in profit units (36 x the encoded scale).

Usage:
  python scripts/rebuttal_inventory_profit.py gate
  python scripts/rebuttal_inventory_profit.py run [processes]
  python scripts/rebuttal_inventory_profit.py analyze
Writes results/rebuttal_inventory_profit/{gate.json,raw.csv,summary.csv,analysis.json}.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tis.allocation import integer_allocation, pilot_size, regularized_weights  # noqa: E402
from tis.engine import compute_influences, solve_stop_loss  # noqa: E402
from tis.environments import (  # noqa: E402
    INVENTORY_BASE_STOCK, INVENTORY_CAPACITY, INVENTORY_POISSON_MEAN,
    _censored_poisson_probabilities, _kernel,
)
from tis.llm_study.runner import _empirical_cvar, _mapping  # noqa: E402
from tis.model import (  # noqa: E402
    FiniteHorizonModel, counts_from_streams, coupled_streams, model_from_counts, sample_counts,
)
from tis.runner import _anchored_score, _occupancy_scores  # noqa: E402

# REBUTTAL_PROFIT_CONFIG selects another tail level; the default is alpha = .1.
CONFIG = json.loads(Path(os.environ.get("REBUTTAL_PROFIT_CONFIG",
                                       ROOT / "configs/rebuttal_inventory_profit.json")).read_text())
FAMILY = json.loads((ROOT / CONFIG["family"]).read_text())
PROFIT = CONFIG["profit"]
ALPHA = float(CONFIG["alpha"])
PILOT = CONFIG["pilot"]
SEED = int(CONFIG["master_seed"])
OUT = ROOT / CONFIG["output_directory"]
HORIZON = 8
LOW, HIGH, DENOM = float(PROFIT["lower_bound"]), float(PROFIT["upper_bound"]), int(PROFIT["grid_denominator"])
SPAN = HIGH - LOW
CASES = [
    {"index": i, "p_base": p, "loss": loss, "policy": pol}
    for i, (pol, p, loss) in enumerate(
        (pol, p, loss) for pol in FAMILY["policies"]
        for p in FAMILY["base_disruption_probabilities"] for loss in FAMILY["disruption_losses"])
]
CONDITIONAL = ("uniform", "learned_occupancy", "learned_mean", "tis", "tis_anchored",
               "occ_plus_mean", "model_accuracy", "oracle_tail")


def period_profit(sales: int, order: int, ending: int, unmet: int) -> float:
    return (PROFIT["sale"] * sales + PROFIT["order"] * order
            + PROFIT["holding"] * ending + PROFIT["unmet_demand"] * unmet)


def build_case(p_base: float, loss: int, policy: str) -> FiniteHorizonModel:
    states = INVENTORY_CAPACITY + 1
    pol = np.zeros((HORIZON + 1, states, 1), dtype=float)
    pol[1:, :, 0] = 1.0
    kernels = {}
    for h in range(1, HORIZON + 1):
        season = HORIZON - h
        target = INVENTORY_BASE_STOCK[season]
        if policy == "plus_one":
            target = min(target + 1, INVENTORY_CAPACITY)
        demand_probabilities = _censored_poisson_probabilities(INVENTORY_POISSON_MEAN[season])
        for state in range(states):
            order = min(INVENTORY_CAPACITY - state, max(0, target - state))
            available = state + order
            disruption = p_base + 0.02 if available == INVENTORY_CAPACITY else p_base
            outcomes = []
            for demand, demand_probability in enumerate(demand_probabilities):
                sales = min(available, demand)
                unmet = max(demand - available, 0)
                ending = max(available - demand, 0)
                profit = period_profit(sales, order, ending, unmet)
                if not LOW <= profit <= HIGH:
                    raise ValueError("profit outside the declared structural bounds")
                reward = (profit - LOW) / SPAN
                for disrupted, event_probability in ((False, 1.0 - disruption), (True, disruption)):
                    outcomes.append((reward, max(ending - loss * int(disrupted), 0),
                                     float(demand_probability * event_probability)))
            kernels[(h, state)] = _kernel(outcomes)
    reachable = {HORIZON: {0}}
    for h in range(HORIZON, 1, -1):
        nxt = set()
        for state in reachable[h]:
            k = kernels[(h, state)]
            nxt.update(int(s) for s, pr in zip(k.next_states, k.probabilities) if pr > 0)
        reachable[h - 1] = nxt
    groups = tuple((h, s) for h in range(1, HORIZON + 1) for s in sorted(reachable.get(h, set())))
    return FiniteHorizonModel(
        name=f"inventory_profit_p{p_base:g}_L{loss}_{policy}", horizon=HORIZON, initial_state=0,
        grid=np.arange(HORIZON * DENOM + 1) / DENOM, policy=pol, kernels=kernels, query_groups=groups,
        group_mode="untied_layer_state", metadata={"p_base": p_base, "loss": loss, "policy": policy})


def to_profit(encoded_total: float) -> float:
    return SPAN * encoded_total + HORIZON * LOW


def direct_profit_cvar(model: FiniteHorizonModel) -> tuple[float, float]:
    """Independent check: propagate (state, total profit in half units) exactly."""
    dist = {(model.initial_state, 0): 1.0}
    for h in range(HORIZON, 0, -1):
        new = defaultdict(float)
        for (state, half_units), mass in dist.items():
            k = model.kernels[(h, state)]
            for r, s, p in zip(k.rewards, k.next_states, k.probabilities):
                profit_half = int(round(2 * (SPAN * r + LOW)))
                new[(int(s), half_units + profit_half)] += mass * p
        dist = new
    totals = defaultdict(float)
    for (_, half_units), mass in dist.items():
        totals[half_units / 2.0] += mass
    values = sorted(totals)
    remaining, acc = ALPHA, 0.0
    for v in values:
        take = min(totals[v], remaining)
        acc += take * v
        remaining -= take
        if remaining <= 1e-15:
            break
    return acc / ALPHA, float(sum(totals.values()))


def gate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for case in CASES:
        model = build_case(case["p_base"], case["loss"], case["policy"])
        truth = compute_influences(model, ALPHA)
        solver = to_profit(truth.solution.cvar)
        direct, mass = direct_profit_cvar(model)
        sigma, mean = np.asarray(truth.tail_scales), np.asarray(truth.mean_scales)
        shares = [x / x.sum() if x.sum() > 0 else np.full(x.size, 1 / x.size) for x in (sigma, mean)]
        report.append({
            **case, "blocks": model.group_count, "cvar_profit": solver, "direct_cvar_profit": direct,
            "abs_difference": abs(solver - direct), "total_mass": mass, "margin": float(truth.solution.margin),
            "uniform_to_oracle": (model.group_count * float((sigma ** 2).sum()) / float(sigma.sum()) ** 2
                                  if sigma.sum() > 0 else None),
            "tail_mean_distance": float(0.5 * np.abs(shares[0] - shares[1]).sum()),
            "zero_tail_signal": bool(sigma.sum() == 0),
            "rewards_on_grid": bool(all(np.allclose(np.round(k.rewards * DENOM), k.rewards * DENOM)
                                        for k in model.kernels.values())),
        })
    passed = all(r["abs_difference"] < 1e-9 and abs(r["total_mass"] - 1) < 1e-12 and r["rewards_on_grid"]
                 for r in report)
    (OUT / "gate.json").write_text(json.dumps({"passed": passed, "cases": report}, indent=1) + "\n")
    for r in report:
        print(f"case {r['index']:2d} {r['policy']:9s} p={r['p_base']:<5} L={r['loss']} blocks={r['blocks']} "
              f"CVaR={r['cvar_profit']:+.4f} (direct {r['direct_cvar_profit']:+.4f}) margin={r['margin']:.5f} "
              f"unif/oracle={r['uniform_to_oracle']:.2f} tail-mean={r['tail_mean_distance']:.3f}")
    print("gate passed:", passed)


def _rng(case: int, budget: int, replication: int, stage: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SEED, case, budget, replication, stage]))


def model_accuracy_scores(model, pilot_counts) -> np.ndarray:
    out = []
    for group in model.query_groups:
        p = np.asarray(pilot_counts[group], dtype=float)
        p = p / p.sum()
        out.append(np.sqrt(max(1.0 - float(np.sum(p ** 2)), 0.0)))
    return np.asarray(out)


def replications(args):
    from tabular_rollout import rollout_returns
    case_index, budget, start, stop = args
    case = CASES[case_index]
    model = build_case(case["p_base"], case["loss"], case["policy"])
    truth = compute_influences(model, ALPHA)
    C = float(truth.solution.cvar)
    groups = model.group_count
    total = groups * budget
    exploration = total ** (-float(PILOT["exploration_exponent"]))
    minimum_main = int(PILOT["minimum_main_per_group"])
    m = pilot_size(total, groups, float(PILOT["multiplier"]), int(PILOT["minimum_pilot_per_group"]), minimum_main)
    main_total = total - groups * m
    rows = []
    for rep in range(start, stop):
        t0 = time.perf_counter()
        pilot_counts = sample_counts(model, {g: m for g in model.query_groups}, _rng(case_index, budget, rep, 0))
        pilot = compute_influences(model_from_counts(model, pilot_counts), ALPHA)
        pilot_seconds = time.perf_counter() - t0
        occ = _occupancy_scores(model, pilot.occupancy)
        w_occ = regularized_weights(occ, exploration)
        scores = {"learned_occupancy": occ, "learned_mean": np.asarray(pilot.mean_scales),
                  "tis": np.asarray(pilot.tail_scales),
                  "tis_anchored": _anchored_score(np.asarray(pilot.tail_scales), occ),
                  "model_accuracy": model_accuracy_scores(model, pilot_counts)}
        allocations = {
            "uniform": integer_allocation(total, np.full(groups, 1.0 / groups), minimum_main),
            "oracle_tail": integer_allocation(
                total, regularized_weights(np.asarray(truth.tail_scales), exploration), minimum_main),
            "occ_plus_mean": integer_allocation(
                main_total, 0.5 * regularized_weights(scores["learned_mean"], exploration) + 0.5 * w_occ,
                minimum_main),
        }
        for method, score in scores.items():
            allocations[method] = integer_allocation(main_total, regularized_weights(score, exploration), minimum_main)
        maximum = np.max(np.vstack([allocations[k] for k in CONDITIONAL]), axis=0)
        streams = coupled_streams(model, _mapping(model, maximum), _rng(case_index, budget, rep, 1))
        for method in CONDITIONAL:
            t1 = time.perf_counter()
            empirical = counts_from_streams(model, streams, _mapping(model, allocations[method]))
            estimate = solve_stop_loss(model_from_counts(model, empirical), ALPHA).cvar
            rows.append({"case": case_index, "budget": budget, "replication": rep, "method": method,
                         "error_profit": SPAN * (estimate - C), "charged_queries": total,
                         "seconds": time.perf_counter() - t1 + (pilot_seconds if method not in
                                                               ("uniform", "oracle_tail") else 0.0)})
        t1 = time.perf_counter()
        runs = total // HORIZON
        returns = rollout_returns(model, runs, _rng(case_index, budget, rep, 6))
        rows.append({"case": case_index, "budget": budget, "replication": rep, "method": "complete_rollout",
                     "error_profit": SPAN * (_empirical_cvar(returns, ALPHA) - C),
                     "charged_queries": runs * HORIZON, "seconds": time.perf_counter() - t1})
    return rows


def run(processes: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gate_report = json.loads((OUT / "gate.json").read_text())
    if not gate_report["passed"]:
        raise SystemExit("gate failed; no simulation")
    reps = int(CONFIG["replications"])
    jobs = [(c["index"], b, s, min(s + 25, reps)) for c in CASES
            for b in CONFIG["budgets_per_block"] for s in range(0, reps, 25)]
    rows = []
    with ProcessPoolExecutor(max_workers=processes) as pool:
        for chunk in pool.map(replications, jobs):
            rows.extend(chunk)
    rows.sort(key=lambda r: (r["case"], r["budget"], r["replication"], r["method"]))
    with (OUT / "raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows")


def analyze() -> None:
    errors = defaultdict(dict)
    seconds = defaultdict(list)
    queries = {}
    for row in csv.DictReader((OUT / "raw.csv").open()):
        key = (int(row["case"]), int(row["budget"]), row["method"])
        errors[key][int(row["replication"])] = float(row["error_profit"])
        seconds[key].append(float(row["seconds"]))
        queries[key] = int(row["charged_queries"])
    budgets = CONFIG["budgets_per_block"]
    methods = sorted({k[2] for k in errors})
    summary = []
    for (case, budget, method), e in sorted(errors.items()):
        arr = np.array([e[r] for r in sorted(e)])
        sq = arr ** 2
        summary.append({"case": case, "budget": budget, "method": method, "replications": arr.size,
                        "mse": float(sq.mean()), "rmse": float(np.sqrt(sq.mean())),
                        "mse_se": float(sq.std(ddof=1) / np.sqrt(sq.size)), "bias": float(arr.mean()),
                        "charged_queries": queries[(case, budget, method)],
                        "seconds_per_replication": float(np.mean(seconds[(case, budget, method)]))})
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    mse = {(r["case"], r["budget"], r["method"]): r["mse"] for r in summary}

    def paired(case, budget, method, reference):
        a, r = errors[(case, budget, method)], errors[(case, budget, reference)]
        d = np.array([a[k] ** 2 - r[k] ** 2 for k in sorted(a)])
        return float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size))) if d.std() > 0 else 0.0

    (primary_key, p2), = CONFIG["primary"].items()
    b, meth, ref = p2["budget"], p2["method"], p2["reference"]
    cases = [c["index"] for c in CASES]
    sa = {c: np.array([errors[(c, b, meth)][k] ** 2 for k in sorted(errors[(c, b, meth)])]) for c in cases}
    sr = {c: np.array([errors[(c, b, ref)][k] ** 2 for k in sorted(errors[(c, b, ref)])]) for c in cases}
    family_ratio = np.mean([sa[c].mean() for c in cases]) / np.mean([sr[c].mean() for c in cases])
    family_diff = np.mean([sa[c].mean() - sr[c].mean() for c in cases])
    rng = np.random.default_rng(SEED + 1)
    resamples = int(CONFIG["bootstrap_resamples"])
    ratios, centred = np.empty(resamples), np.empty(resamples)
    idx = {c: rng.integers(0, sa[c].size, size=(resamples, sa[c].size)) for c in cases}
    num = np.mean([sa[c][idx[c]].mean(axis=1) for c in cases], axis=0)
    den = np.mean([sr[c][idx[c]].mean(axis=1) for c in cases], axis=0)
    ratios[:] = num / den
    centred[:] = (num - den) - family_diff
    case_ratios = {c: float(sa[c].mean() / sr[c].mean()) for c in cases}
    analysis = {primary_key: {"family_mse_ratio": float(family_ratio),
                       "ratio_ci95": [float(np.quantile(ratios, .025)), float(np.quantile(ratios, .975))],
                       "family_mse_difference_profit2": float(family_diff),
                       "bootstrap_p_two_sided": float(np.mean(np.abs(centred) >= abs(family_diff))),
                       "case_ratios": case_ratios,
                       "median_case_ratio": float(np.median(list(case_ratios.values()))),
                       "worst_case_ratio": float(max(case_ratios.values()))}}
    batch_ratios = {}
    for lo, hi in CONFIG["batches"]:
        br = {c: sa[c][lo:hi].mean() / sr[c][lo:hi].mean() for c in cases}
        batch_ratios[f"{lo}-{hi}"] = {
            "family_mse_ratio": float(np.mean([sa[c][lo:hi].mean() for c in cases])
                                      / np.mean([sr[c][lo:hi].mean() for c in cases])),
            "median_case_ratio": float(np.median(list(br.values()))), "worst_case_ratio": float(max(br.values()))}
    analysis[primary_key]["batches"] = batch_ratios
    pairs =[("tis_anchored", "occ_plus_mean"), ("tis", "learned_mean"), ("tis", "occ_plus_mean"),
             ("tis", "learned_occupancy"), ("tis_anchored", "learned_occupancy"), ("tis", "model_accuracy"),
             ("tis", "uniform"), ("tis_anchored", "uniform")]
    scoreboard = {}
    for budget in budgets:
        for method, reference in pairs:
            zs = [paired(c, budget, method, reference) for c in cases]
            rr = [mse[(c, budget, method)] / mse[(c, budget, reference)] for c in cases]
            scoreboard[f"{method}_vs_{reference}|{budget}"] = {
                "resolved_better": int(sum(z <= -2 for z in zs)), "resolved_worse": int(sum(z >= 2 for z in zs)),
                "median_ratio": float(np.median(rr)), "min_ratio": float(min(rr)), "max_ratio": float(max(rr))}
        for method in ("tis", "tis_anchored", "occ_plus_mean", "learned_mean"):
            rr = [mse[(c, budget, method)] / mse[(c, budget, "complete_rollout")] for c in cases]
            scoreboard[f"{method}_vs_complete_rollout|{budget}"] = {
                "median_ratio": float(np.median(rr)), "min_ratio": float(min(rr)), "max_ratio": float(max(rr)),
                "paired": False}
    analysis["scoreboard"] = scoreboard
    tolerance = CONFIG["rmse_tolerances_profit"]
    crossings = {}
    for tol in [tolerance["primary"]] + tolerance["secondary"]:
        for method in methods:
            per_case = {}
            for c in cases:
                rm = [np.sqrt(mse[(c, bb, method)]) for bb in budgets]
                ok = [bb for bb, r in zip(budgets, rm) if r <= tol]
                if not ok:
                    per_case[c] = {"status": "not reached"}
                    continue
                first = ok[0]
                status = "below grid" if first == budgets[0] else "crossed"
                if any(r > tol for bb, r in zip(budgets, rm) if bb > first):
                    status += ", non-monotone"
                per_case[c] = {"status": status, "first_budget": first,
                               "charged_queries": queries[(c, first, method)]}
            crossings[f"{method}|{tol}"] = per_case
    analysis["tolerance_crossings"] = crossings
    a_path = ROOT / "results/rebuttal_cliffwalking_attribution/analysis.json"
    if "holm" not in CONFIG and a_path.exists():
        p1 = json.loads(a_path.read_text())["P1"]["bootstrap_p_two_sided"]
        p2v = analysis[primary_key]["bootstrap_p_two_sided"]
        resolution = 1.0 / resamples
        ps = {"P1": max(p1, resolution), "P2": max(p2v, resolution)}
        order = sorted(ps, key=ps.get)
        adjusted, running = {}, 0.0
        for rank, key in enumerate(order):
            running = max(running, min(1.0, (len(ps) - rank) * ps[key]))
            adjusted[key] = running
        analysis["holm"] = {"raw_p_floor_at_bootstrap_resolution": ps, "adjusted": adjusted}
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=1) + "\n")
    print(f"{primary_key}:", {k: v for k, v in analysis[primary_key].items() if k != "case_ratios"})
    if "holm" in analysis:
        print("Holm:", analysis["holm"])
    for key, value in scoreboard.items():
        print(f"  {key:42s} {value}")
    for tol in [tolerance["primary"]]:
        for method in methods:
            per = crossings[f"{method}|{tol}"]
            reached = [v for v in per.values() if "charged_queries" in v]
            print(f"  RMSE<={tol}: {method:18s} reached in {len(reached)}/18 cases; "
                  f"median charged queries {np.median([v['charged_queries'] for v in reached]) if reached else 'n/a'}")


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "gate":
        gate()
    elif step == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif step == "analyze":
        analyze()
    else:
        raise SystemExit("usage: rebuttal_inventory_profit.py gate | run [processes] | analyze")
