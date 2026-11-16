#!/usr/bin/env python3
"""Restarting only from states the evaluator has visited (Appendix F.5).

Configuration: configs/rebuttal_checkpoint_access.json. Each checkpoint design spends the
full-access pilot budget on complete trajectories, fits a pilot model
(unvisited kernels get the uniform law over their declared outcomes), and
spends the rest of the budget on fresh queries at kernels visited by those
trajectories only. The final estimate uses the main queries only.

References come from the committed results of the CliffWalking and inventory
raw-profit studies. For inventory, fitted-model rollouts are computed here on
exactly that study's rollout trajectories; their empirical CVaR must
reproduce its complete-rollout errors (`check`).

Usage:
  python scripts/rebuttal_checkpoint_access.py check
  python scripts/rebuttal_checkpoint_access.py run [processes]
  python scripts/rebuttal_checkpoint_access.py analyze
Writes results/rebuttal_checkpoint_access/{raw.csv,summary.csv,analysis.json}.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.pop("REBUTTAL_PROFIT_CONFIG", None)  # the inventory module must load the raw-profit study's config

import rebuttal_cliffwalking_attribution as A  # noqa: E402
import rebuttal_inventory_profit as B  # noqa: E402
from tis.allocation import integer_allocation, pilot_size, regularized_weights  # noqa: E402
from tis.engine import compute_influences, solve_stop_loss  # noqa: E402
from tis.environments import make_environment  # noqa: E402
from tis.llm_study.runner import _empirical_cvar, _mapping  # noqa: E402
from tis.model import counts_from_streams, coupled_streams, model_from_counts  # noqa: E402
from tis.runner import _anchored_score, _occupancy_scores  # noqa: E402

CONFIG = json.loads((ROOT / "configs/rebuttal_checkpoint_access.json").read_text())
SEED = int(CONFIG["master_seed"])
DESIGNS = tuple(CONFIG["designs"])
OUT = ROOT / CONFIG["output_directory"]
ALPHA = 0.1
assert A.ALPHA == ALPHA and B.ALPHA == ALPHA and B.SEED == 20261108
CLIFF = CONFIG["cliffwalking"]
INV = CONFIG["inventory"]
# Full-access counterpart of each checkpoint design (secondary contrasts).
COUNTERPART = {"ckpt_uniform": "uniform", "ckpt_occupancy": "learned_occupancy", "ckpt_mean": "learned_mean",
               "ckpt_tis": "tis", "ckpt_anchored": "tis_anchored", "ckpt_occ_plus_mean": "occ_plus_mean"}


def _rng(setting: int, case: int, budget: int, replication: int, stage: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SEED, setting, case, budget, replication, stage]))


def rollout_with_counts(model, runs: int, rng: np.random.Generator):
    """Trajectories through the true laws with per-kernel outcome counts.

    Draws in exactly the order of scripts/tabular_rollout.py::rollout_returns
    (and of the CliffWalking study's rollouts), so a shared generator gives the same
    trajectories.
    """
    untied = model.group_mode == "untied_layer_state"
    counts = {g: np.zeros(model.kernels[g].size, dtype=int) for g in model.query_groups}
    state = np.full(runs, model.initial_state, dtype=int)
    totals = np.zeros(runs)
    for step in range(model.horizon):
        h = model.horizon - step
        successor = np.empty_like(state)
        for current in np.unique(state):
            mask = state == current
            count = int(mask.sum())
            if untied:
                group = (h, int(current))
                kernel = model.kernels[group]
                outcomes = rng.choice(kernel.size, size=count, p=kernel.probabilities)
                totals[mask] += kernel.rewards[outcomes]
                successor[mask] = kernel.next_states[outcomes]
                if group in counts:
                    counts[group] += np.bincount(outcomes, minlength=kernel.size)
                continue
            probs = np.asarray(model.policy[h, int(current)], dtype=float)
            actions = rng.choice(probs.size, size=count, p=probs / probs.sum())
            rewards, nexts = np.empty(count), np.empty(count, dtype=int)
            for action in np.unique(actions):
                amask = actions == action
                group = (int(current), int(action))
                kernel = model.kernels[group]
                outcomes = rng.choice(kernel.size, size=int(amask.sum()), p=kernel.probabilities)
                rewards[amask] = kernel.rewards[outcomes]
                nexts[amask] = kernel.next_states[outcomes]
                if group in counts:
                    counts[group] += np.bincount(outcomes, minlength=kernel.size)
            totals[mask] += rewards
            successor[mask] = nexts
        state = successor
    return totals, counts


def checkpoint_designs(model, truth_occupancy, C, total, m, minimum_main, exploration, setting, case, budget,
                       rep, scale):
    """All checkpoint designs for one replication; errors are scale * (estimate - C)."""
    horizon, groups = model.horizon, model.group_count
    runs = (groups * m) // horizon
    _, pilot_counts = rollout_with_counts(model, runs, _rng(setting, case, budget, rep, 7))
    visited = np.flatnonzero([pilot_counts[g].sum() > 0 for g in model.query_groups])
    filled, _ = A.complete_rows(model, pilot_counts, "uniform")
    pilot = compute_influences(model_from_counts(model, filled), ALPHA)
    occ = _occupancy_scores(model, pilot.occupancy)
    tail, mean = np.asarray(pilot.tail_scales), np.asarray(pilot.mean_scales)
    main_total = total - runs * horizon

    def floored(score):
        return regularized_weights(score[visited], exploration)

    def allocate(weights):
        n = np.zeros(groups, dtype=int)
        n[visited] = integer_allocation(main_total, weights, minimum_main)
        return n

    allocations = {
        "ckpt_uniform": allocate(np.full(visited.size, 1.0 / visited.size)),
        "ckpt_occupancy": allocate(floored(occ)),
        "ckpt_mean": allocate(floored(mean)),
        "ckpt_tis": allocate(floored(tail)),
        "ckpt_anchored": allocate(regularized_weights(_anchored_score(tail[visited], occ[visited]), exploration)),
        "ckpt_occ_plus_mean": allocate(0.5 * floored(mean) + 0.5 * floored(occ)),
    }
    maximum = np.max(np.vstack([allocations[d] for d in DESIGNS]), axis=0)
    streams = coupled_streams(model, _mapping(model, maximum), _rng(setting, case, budget, rep, 8))
    unvisited = np.setdiff1d(np.arange(groups), visited)
    rows = []
    for design in DESIGNS:
        n = allocations[design]
        assert runs * horizon + int(n.sum()) == total and np.all(n[unvisited] == 0)
        counts, _ = A.complete_rows(model, counts_from_streams(model, streams, _mapping(model, n)), "uniform")
        estimate = solve_stop_loss(model_from_counts(model, counts), ALPHA).cvar
        rows.append({"setting": "cliffwalking" if setting == 0 else "inventory", "case": case, "budget": budget,
                     "replication": rep, "arm": design, "error": scale * (estimate - C), "charged_queries": total,
                     "pilot_runs": runs, "visited_kernels": int(visited.size),
                     "unvisited_visits_per_run": float(truth_occupancy[unvisited].sum())})
    return rows


def cliff_job(args):
    budget, rep = args
    model = make_environment(A.CONFIG["environment"])
    truth = compute_influences(model, ALPHA)
    total = model.group_count * budget
    m = pilot_size(total, model.group_count, float(A.PILOT["multiplier"]), int(A.PILOT["minimum_pilot_per_group"]),
                   int(A.PILOT["minimum_main_per_group"]))
    return checkpoint_designs(model, _occupancy_scores(model, truth.occupancy), float(truth.solution.cvar), total,
                              m, int(A.PILOT["minimum_main_per_group"]),
                              total ** (-float(A.PILOT["exploration_exponent"])), 0, 0, budget, rep, 1.0)


def inventory_job(args):
    case_index, budget, start, stop = args
    case = B.CASES[case_index]
    model = B.build_case(case["p_base"], case["loss"], case["policy"])
    truth = compute_influences(model, ALPHA)
    C = float(truth.solution.cvar)
    occupancy = _occupancy_scores(model, truth.occupancy)
    total = model.group_count * budget
    minimum_main = int(B.PILOT["minimum_main_per_group"])
    m = pilot_size(total, model.group_count, float(B.PILOT["multiplier"]), int(B.PILOT["minimum_pilot_per_group"]),
                   minimum_main)
    exploration = total ** (-float(B.PILOT["exploration_exponent"]))
    rows = []
    for rep in range(start, stop):
        rows += checkpoint_designs(model, occupancy, C, total, m, minimum_main, exploration, 1, case_index, budget,
                                   rep, B.SPAN)
        # Fitted-model rollouts on exactly the raw-profit study's rollout trajectories (same seed and stream).
        runs = total // model.horizon
        returns, counts = rollout_with_counts(model, runs, B._rng(case_index, budget, rep, 6))
        filled, empty = A.complete_rows(model, counts, "uniform")
        index = [model.query_groups.index(g) for g in empty]
        base = {"setting": "inventory", "case": case_index, "budget": budget, "replication": rep,
                "charged_queries": runs * model.horizon, "pilot_runs": 0,
                "visited_kernels": model.group_count - len(empty),
                "unvisited_visits_per_run": float(occupancy[index].sum())}
        rows.append({**base, "arm": "rollout_model",
                     "error": B.SPAN * (solve_stop_loss(model_from_counts(model, filled), ALPHA).cvar - C)})
        rows.append({**base, "arm": "rollout_empirical_check", "error": B.SPAN * (_empirical_cvar(returns, ALPHA) - C)})
    return rows


def _jobs():
    cliff = [("cliff", (b, r)) for b in CLIFF["budgets_per_kernel"] for r in range(int(CLIFF["replications"]))]
    reps = int(INV["replications"])
    inventory = [("inventory", (c["index"], b, s, min(s + 25, reps))) for c in B.CASES
                 for b in INV["budgets_per_block"] for s in range(0, reps, 25)]
    return cliff, inventory


def _dispatch(job):
    kind, args = job
    return cliff_job(args) if kind == "cliff" else inventory_job(args)


def check() -> None:
    """Pre-run correctness checks; prints no method comparison."""
    # 1. Rollout draws reproduce the raw-profit study's complete-rollout errors and the CliffWalking study's rollout model.
    b_errors = {}
    for row in csv.DictReader((ROOT / INV["reference_results"]).open()):
        if row["method"] == "complete_rollout" and int(row["replication"]) < 2 and int(row["budget"]) in (150, 2400):
            b_errors[(int(row["case"]), int(row["budget"]), int(row["replication"]))] = float(row["error_profit"])
    worst = 0.0
    for (c, b, r), reference in sorted(b_errors.items()):
        rows = inventory_job((c, b, r, r + 1))
        mine = next(x["error"] for x in rows if x["arm"] == "rollout_empirical_check")
        worst = max(worst, abs(mine - reference))
        assert all(x["charged_queries"] <= b * B.build_case(**{k: B.CASES[c][k] for k in ("p_base", "loss",
                   "policy")}).group_count for x in rows)
    print(f"inventory: rollout trajectories reproduce the raw-profit study in {len(b_errors)} checks (max |diff| {worst:.1e})")
    a_errors = {}
    for row in csv.DictReader((ROOT / CLIFF["reference_results"]).open()):
        if row["arm"] == "rollout_model" and int(row["replication"]) < 2:
            a_errors[(int(row["budget"]), int(row["replication"]))] = float(row["error"])
    model = make_environment(A.CONFIG["environment"])
    truth = compute_influences(model, ALPHA)
    worst = 0.0
    for (b, r), reference in sorted(a_errors.items()):
        _, counts = rollout_with_counts(model, model.group_count * b // model.horizon, A._rng(b, r, 6))
        filled, _ = A.complete_rows(model, counts, "uniform")
        mine = solve_stop_loss(model_from_counts(model, filled), ALPHA).cvar - truth.solution.cvar
        worst = max(worst, abs(mine - reference))
    print(f"cliffwalking: rollout counts reproduce the CliffWalking study's fitted-model rollouts in {len(a_errors)} checks "
          f"(max |diff| {worst:.1e})")
    # 2. Budget conservation and visited-only allocation (asserted inside the jobs) for one replication each.
    rows = cliff_job((400, 0)) + inventory_job((0, 1200, 0, 1)) + inventory_job((17, 150, 0, 1))
    print("budget conservation and visited-only allocation asserted for",
          len([r for r in rows if r["arm"].startswith("ckpt")]), "design-replications; visited kernels:",
          sorted({(r["setting"], r["visited_kernels"]) for r in rows if r["arm"].startswith("ckpt")}))


def run(processes: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cliff, inventory = _jobs()
    rows = []
    with ProcessPoolExecutor(max_workers=processes) as pool:
        for chunk in pool.map(_dispatch, inventory + cliff, chunksize=1):
            rows.extend(chunk)
    rows.sort(key=lambda r: (r["setting"], r["case"], r["budget"], r["replication"], r["arm"]))
    with (OUT / "raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows")


def _load():
    """errors[(setting, case, budget, arm)][replication]; references from Experiments A and B."""
    errors = defaultdict(dict)
    extra = defaultdict(list)
    for row in csv.DictReader((OUT / "raw.csv").open()):
        key = (row["setting"], int(row["case"]), int(row["budget"]), row["arm"])
        errors[key][int(row["replication"])] = float(row["error"])
        extra[key].append((int(row["visited_kernels"]), float(row["unvisited_visits_per_run"])))
    for row in csv.DictReader((ROOT / CLIFF["reference_results"]).open()):
        errors[("cliffwalking", 0, int(row["budget"]), row["arm"])][int(row["replication"])] = float(row["error"])
    for row in csv.DictReader((ROOT / INV["reference_results"]).open()):
        errors[("inventory", int(row["case"]), int(row["budget"]), row["method"])][int(row["replication"])] = \
            float(row["error_profit"])
    return errors, extra


def _sq(errors, key):
    e = errors[key]
    return np.array([e[k] ** 2 for k in sorted(e)])


def paired_z(errors, a, r):
    d = _sq(errors, a) - _sq(errors, r)
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))


def unpaired_z(errors, a, r):
    sa, sr = _sq(errors, a), _sq(errors, r)
    return float((sa.mean() - sr.mean()) / np.sqrt(sa.var(ddof=1) / sa.size + sr.var(ddof=1) / sr.size))


def family_test(errors, keys_a, keys_r, rng, resamples):
    """Ratio of equal-weight mean MSEs over cases; arms resampled independently within each case."""
    sa = [_sq(errors, k) for k in keys_a]
    sr = [_sq(errors, k) for k in keys_r]
    num = np.mean([x.mean() for x in sa])
    den = np.mean([x.mean() for x in sr])
    boot_a = np.mean([x[rng.integers(0, x.size, size=(resamples, x.size))].mean(axis=1) for x in sa], axis=0)
    boot_r = np.mean([x[rng.integers(0, x.size, size=(resamples, x.size))].mean(axis=1) for x in sr], axis=0)
    centred = (boot_a - boot_r) - (num - den)
    return {"mse_ratio": float(num / den), "ratio_ci95": [float(np.quantile(boot_a / boot_r, .025)),
                                                          float(np.quantile(boot_a / boot_r, .975))],
            "mse_difference": float(num - den),
            "bootstrap_p_two_sided": float(np.mean(np.abs(centred) >= abs(num - den)))}


def analyze() -> None:
    errors, extra = _load()
    # Correctness: the raw-profit study's rollouts reproduced exactly.
    diffs = [abs(errors[("inventory", c, b, "rollout_empirical_check")][r] - errors[("inventory", c, b,
             "complete_rollout")][r]) for c in range(len(B.CASES)) for b in INV["budgets_per_block"]
             for r in range(int(INV["replications"]))]
    analysis = {"check_rollouts_reproduce_experiment_B_max_abs_diff": float(max(diffs))}
    summary = []
    for key in sorted(errors):
        setting, case, budget, arm = key
        e = np.array([errors[key][k] for k in sorted(errors[key])])
        visited = [x[0] for x in extra.get(key, [])]
        mass = [x[1] for x in extra.get(key, [])]
        summary.append({"setting": setting, "case": case, "budget": budget, "arm": arm, "replications": e.size,
                        "mse": float(np.mean(e ** 2)), "mse_se": float(np.std(e ** 2, ddof=1) / np.sqrt(e.size)),
                        "bias": float(e.mean()), "mean_visited_kernels": float(np.mean(visited)) if visited else "",
                        "mean_unvisited_visits_per_run": float(np.mean(mass)) if mass else ""})
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    mse = {(r["setting"], r["case"], r["budget"], r["arm"]): r["mse"] for r in summary}
    rng = np.random.default_rng(SEED + 1)
    resamples = int(CONFIG["bootstrap_resamples"])
    cases = [c["index"] for c in B.CASES]
    p3, p4 = CONFIG["primary"]["P3"], CONFIG["primary"]["P4"]
    analysis["P3"] = family_test(errors, [("cliffwalking", 0, p3["budget"], p3["method"])],
                                 [("cliffwalking", 0, p3["budget"], p3["reference"])], rng, resamples)
    analysis["P4"] = family_test(errors, [("inventory", c, p4["budget"], p4["method"]) for c in cases],
                                 [("inventory", c, p4["budget"], p4["reference"]) for c in cases], rng, resamples)
    for name, p, setting, case_list, batches in (("P3", p3, "cliffwalking", [0], CLIFF["batches"]),
                                                 ("P4", p4, "inventory", cases, INV["batches"])):
        out = {}
        for lo, hi in batches:
            num = np.mean([np.mean([errors[(setting, c, p["budget"], p["method"])][k] ** 2 for k in range(lo, hi)])
                           for c in case_list])
            den = np.mean([np.mean([errors[(setting, c, p["budget"], p["reference"])][k] ** 2 for k in range(lo, hi)])
                           for c in case_list])
            out[f"{lo}-{hi}"] = float(num / den)
        analysis[name]["batches"] = out
    scoreboard = {}
    for budget in CLIFF["budgets_per_kernel"]:
        def key(arm):
            return ("cliffwalking", 0, budget, arm)
        for design in DESIGNS:
            for ref, test in (("rollout_model", unpaired_z), ("ckpt_uniform", paired_z),
                              (COUNTERPART[design], unpaired_z)):
                if ref == design:
                    continue
                scoreboard[f"cliffwalking|{design}_vs_{ref}|{budget}"] = {
                    "ratio": mse[key(design)] / mse[key(ref)], "z": test(errors, key(design), key(ref))}
        for a, r in (("ckpt_tis", "ckpt_mean"), ("ckpt_anchored", "ckpt_occ_plus_mean")):
            scoreboard[f"cliffwalking|{a}_vs_{r}|{budget}"] = {"ratio": mse[key(a)] / mse[key(r)],
                                                               "z": paired_z(errors, key(a), key(r))}
    for budget in INV["budgets_per_block"]:
        pairs = [(d, "rollout_model", unpaired_z) for d in DESIGNS] + \
                [(d, "ckpt_uniform", paired_z) for d in DESIGNS if d != "ckpt_uniform"] + \
                [(d, COUNTERPART[d], unpaired_z) for d in DESIGNS] + \
                [("ckpt_tis", "ckpt_mean", paired_z), ("ckpt_anchored", "ckpt_occ_plus_mean", paired_z),
                 ("rollout_model", "complete_rollout", paired_z)]
        for a, r, test in pairs:
            zs = [test(errors, ("inventory", c, budget, a), ("inventory", c, budget, r)) for c in cases]
            rr = [mse[("inventory", c, budget, a)] / mse[("inventory", c, budget, r)] for c in cases]
            fam = (np.mean([mse[("inventory", c, budget, a)] for c in cases])
                   / np.mean([mse[("inventory", c, budget, r)] for c in cases]))
            scoreboard[f"inventory|{a}_vs_{r}|{budget}"] = {
                "family_ratio": float(fam), "median_ratio": float(np.median(rr)), "min_ratio": float(min(rr)),
                "max_ratio": float(max(rr)), "resolved_better": int(sum(z <= -2 for z in zs)),
                "resolved_worse": int(sum(z >= 2 for z in zs)), "paired": test is paired_z}
    analysis["scoreboard"] = scoreboard
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=1) + "\n")
    print("check: max |diff| vs the raw-profit study rollouts", analysis["check_rollouts_reproduce_experiment_B_max_abs_diff"])
    print("P3:", analysis["P3"])
    print("P4:", analysis["P4"])
    for k, v in scoreboard.items():
        print(f"  {k:62s} {v}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "check":
        check()
    elif step == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif step == "analyze":
        analyze()
    else:
        raise SystemExit("usage: rebuttal_checkpoint_access.py check | run [processes] | analyze")
