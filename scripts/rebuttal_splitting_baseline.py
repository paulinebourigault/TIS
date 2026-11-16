#!/usr/bin/env python3
"""Splitting baselines on the inventory raw-profit family (Appendix F.6).

The tail level and the cases come from REBUTTAL_PROFIT_CONFIG, as for
scripts/rebuttal_inventory_profit.py.

Usage:
  REBUTTAL_PROFIT_CONFIG=configs/rebuttal_inventory_profit_alpha01.json \
      python scripts/rebuttal_splitting_baseline.py check | run [processes] | analyze
  python scripts/rebuttal_splitting_baseline.py ams_check | ams [processes]      (adaptive multilevel splitting; all tail levels)
Writes <output directory of the config>/splitting_{check.json,raw.csv,analysis.json} and
results/rebuttal_splitting_ams/{check.json,raw.csv}.
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
os.environ.setdefault("REBUTTAL_PROFIT_CONFIG", str(ROOT / "configs/rebuttal_inventory_profit.json"))

import rebuttal_cliffwalking_attribution as A  # noqa: E402
import rebuttal_inventory_profit as B  # noqa: E402
from tis.engine import compute_influences, solve_stop_loss  # noqa: E402
from tis.llm_study.runner import _empirical_cvar  # noqa: E402
from tis.model import model_from_counts  # noqa: E402

BUDGETS = (300, 1200, 2400)
TILTS = (0.5, 1.0, 1.5, 2.0, 3.0) if B.ALPHA >= 0.01 else (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
LEVELS = ("profit", "profit_value")
VARIANTS = [("profit", 0.0)] + [(level, c) for level in LEVELS for c in TILTS]
H = B.HORIZON


def moments(model):
    """True expected remaining encoded return V[h][s] and the sd of the root return."""
    states = model.policy.shape[1]
    V = np.zeros((H + 1, states))
    M2 = np.zeros((H + 1, states))
    for h in range(1, H + 1):
        for s in range(states):
            k = model.kernels[(h, s)]
            nxt = V[h - 1][k.next_states]
            V[h, s] = float(np.sum(k.probabilities * (k.rewards + nxt)))
            M2[h, s] = float(np.sum(k.probabilities * (k.rewards ** 2 + 2 * k.rewards * nxt
                                                        + M2[h - 1][k.next_states])))
    s0 = model.initial_state
    return V, float(np.sqrt(max(M2[H, s0] - V[H, s0] ** 2, 0.0)))


def root_distribution(model):
    """Exact distribution of the encoded eight-period return (grid index -> mass)."""
    dist = {(model.initial_state, 0): 1.0}
    for h in range(H, 0, -1):
        new = defaultdict(float)
        for (state, units), mass in dist.items():
            k = model.kernels[(h, state)]
            for r, s, p in zip(k.rewards, k.next_states, k.probabilities):
                new[(int(s), units + int(round(r * B.DENOM)))] += mass * p
        dist = new
    totals = defaultdict(float)
    for (_, units), mass in dist.items():
        totals[units] += mass
    values = np.array(sorted(totals))
    return values / B.DENOM, np.array([totals[v] for v in values])


def systematic(weights, rng):
    c = np.cumsum(weights)
    u = (rng.random() + np.arange(weights.size)) / weights.size * c[-1]
    return np.minimum(np.searchsorted(c, u, side="left"), weights.size - 1)


def splitting_run(model, V, n, lam, level, rng):
    """One interacting-particle run. Returns returns, unbiased weights, block counts."""
    counts = {g: np.zeros(model.kernels[g].size, dtype=int) for g in model.query_groups}
    state = np.full(n, model.initial_state, dtype=int)
    cum = np.zeros(n)
    use_value = level == "profit_value"
    start = V[H, model.initial_state] if use_value else 0.0
    previous = np.full(n, start)
    log_z = 0.0
    for step in range(H):
        h = H - step
        successor = np.empty_like(state)
        for current in np.unique(state):
            mask = state == current
            group = (h, int(current))
            kernel = model.kernels[group]
            outcomes = rng.choice(kernel.size, size=int(mask.sum()), p=kernel.probabilities)
            cum[mask] += kernel.rewards[outcomes]
            successor[mask] = kernel.next_states[outcomes]
            if group in counts:
                counts[group] += np.bincount(outcomes, minlength=kernel.size)
        state = successor
        if step < H - 1 and lam > 0:
            current_level = cum + (V[h - 1][state] if use_value else 0.0)
            logw = -lam * (current_level - previous)
            top = logw.max()
            w = np.exp(logw - top)
            log_z += top + np.log(w.mean())
            keep = systematic(w, rng)
            state, cum, previous = state[keep], cum[keep], current_level[keep]
    weights = np.exp(log_z + lam * (previous - start)) / n if lam > 0 else np.full(n, 1.0 / n)
    return cum, weights, counts


def weighted_cvar(values, weights, alpha, normalize):
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    if normalize:
        w = w / w.sum()
    c = np.cumsum(w)
    k = int(np.searchsorted(c, alpha, side="left"))
    if k >= v.size:
        return float((np.sum(w * v) + (alpha - c[-1]) * v[-1]) / alpha)
    before = c[k - 1] if k > 0 else 0.0
    return float((np.sum(w[:k] * v[:k]) + (alpha - before) * v[k]) / alpha)


def estimates(model, returns, weights, counts):
    filled, empty = A.complete_rows(model, counts, "uniform")
    fitted = solve_stop_loss(model_from_counts(model, filled), B.ALPHA).cvar
    return (weighted_cvar(returns, weights, B.ALPHA, False), weighted_cvar(returns, weights, B.ALPHA, True),
            float(fitted), len(empty))


def job(args):
    case_index, budget, start, stop = args
    case = B.CASES[case_index]
    model = B.build_case(case["p_base"], case["loss"], case["policy"])
    C = float(compute_influences(model, B.ALPHA).solution.cvar)
    V, sd = moments(model)
    n = model.group_count * budget // H
    rows = []
    for rep in range(start, stop):
        for variant, (level, c) in enumerate(VARIANTS):
            rng = B._rng(case_index, budget, rep, 20 + variant)
            returns, weights, counts = splitting_run(model, V, n, c / sd, level, rng)
            weighted, selfnorm, fitted, empty = estimates(model, returns, weights, counts)
            rows.append({"case": case_index, "budget": budget, "replication": rep, "level": level, "tilt": c,
                         "error_weighted": f"{B.SPAN * (weighted - C):.9g}",
                         "error_selfnorm": f"{B.SPAN * (selfnorm - C):.9g}",
                         "error_fitted": f"{B.SPAN * (fitted - C):.9g}",
                         "total_mass": f"{weights.sum():.6g}", "empty_blocks": empty,
                         "charged_queries": n * H})
    return rows



# ----------------------------------------------------------------------------- adaptive multilevel splitting
AMS_OUT = ROOT / "results/rebuttal_splitting_ams"
AMS_SEED = 20261116
AMS_BUDGET = 1200
AMS_ALPHAS = (0.1, 0.05, 0.01, 0.001, 0.0001)
AMS_FRACTIONS = (0.7, 0.55, 0.5, 0.45, 0.42, 0.39, 0.36, 0.33)
AMS_REPLICATIONS = 200
AMS_LEVELS = ("time", "value")
AMS_VARIANTS = [(level, f) for level in AMS_LEVELS for f in AMS_FRACTIONS]


def level_tables(model, V):
    """Predicted remaining encoded profit after t periods, by state: the true value function ("value") or
    the true expected remaining profit as a function of the period only ("time")."""
    states = model.policy.shape[1]
    value = np.zeros((H + 1, states))
    for t in range(H):
        value[t] = V[H - t]
    occupancy = np.zeros(states)
    occupancy[model.initial_state] = 1.0
    expected = np.zeros(H)
    for t in range(H):
        following = np.zeros(states)
        for s in np.flatnonzero(occupancy):
            k = model.kernels[(H - t, int(s))]
            expected[t] += occupancy[s] * float(np.sum(k.probabilities * k.rewards))
            np.add.at(following, k.next_states, occupancy[s] * k.probabilities)
        occupancy = following
    trend = np.zeros((H + 1, states))
    for t in range(H):
        trend[t, :] = expected[t:].sum()
    return {"time": trend, "value": value}


def ams_run(model, table, n, budget, rng):
    """Adaptive multilevel splitting (Cerou and Guyader, 2007) in the form valid for discrete scores
    (Brehier et al., 2016). Score of a run: the largest value over periods of minus its predicted final
    profit. Each iteration removes every run at the lowest score, replaces each by a copy of a surviving
    run up to the first period at which that run's level exceeds the removed score, and simulates the rest
    afresh. Stops when the next iteration would exceed the budget of simulated transitions.
    Returns final profits and unbiased weights (removed runs included), block counts, transitions used."""
    kernels = {g: (k.rewards, k.next_states, np.cumsum(k.probabilities)) for g, k in model.kernels.items()}
    counts = {g: np.zeros(model.kernels[g].size, dtype=np.int64) for g in model.query_groups}
    states = np.empty((n, H + 1), dtype=int)
    cum = np.zeros((n, H + 1))
    states[:, 0] = model.initial_state
    for t in range(H):
        for current in np.unique(states[:, t]):
            mask = states[:, t] == current
            group = (H - t, int(current))
            kernel = model.kernels[group]
            outcomes = rng.choice(kernel.size, size=int(mask.sum()), p=kernel.probabilities)
            cum[mask, t + 1] = cum[mask, t] + kernel.rewards[outcomes]
            states[mask, t + 1] = kernel.next_states[outcomes]
            counts[group] += np.bincount(outcomes, minlength=kernel.size)
    periods = np.arange(H + 1)
    xi = -(cum + table[periods[None, :], states])
    score = xi.max(axis=1)
    used = n * H
    log_w = 0.0
    retired_values, retired_weights = [], []
    iterations = 0
    while True:
        z = score.min()
        kill = np.flatnonzero(score <= z)
        if kill.size == n:
            break
        survivors = np.flatnonzero(score > z)
        parents = rng.choice(survivors, size=kill.size)
        branch = (xi[parents] > z).argmax(axis=1)
        cost = int((H - branch).sum())
        if used + cost > budget:
            break
        retired_values.append(cum[kill, H].copy())
        retired_weights.append(np.full(kill.size, np.exp(log_w) / n))
        log_w += np.log((n - kill.size) / n)
        uniforms = rng.random(cost)
        position = 0
        for i, parent, start in zip(kill, parents, branch):
            states[i, :start + 1] = states[parent, :start + 1]
            cum[i, :start + 1] = cum[parent, :start + 1]
            s, c = int(states[i, start]), float(cum[i, start])
            for t in range(start, H):
                group = (H - t, s)
                rewards, successors, cdf = kernels[group]
                o = min(int(np.searchsorted(cdf, uniforms[position], side="right")), rewards.size - 1)
                position += 1
                counts[group][o] += 1
                c += rewards[o]
                s = int(successors[o])
                states[i, t + 1] = s
                cum[i, t + 1] = c
            xi[i] = -(cum[i] + table[periods, states[i]])
            score[i] = xi[i].max()
        used += cost
        iterations += 1
    values = np.concatenate(retired_values + [cum[:, H]])
    weights = np.concatenate(retired_weights + [np.full(n, np.exp(log_w) / n)])
    return values, weights, counts, used, iterations


def ams_job(args):
    case_index, start, stop = args
    case = B.CASES[case_index]
    model = B.build_case(case["p_base"], case["loss"], case["policy"])
    truth = {a: float(compute_influences(model, a).solution.cvar) for a in AMS_ALPHAS}
    V, _ = moments(model)
    tables = level_tables(model, V)
    total = model.group_count * AMS_BUDGET // H * H
    rows = []
    for rep in range(start, stop):
        for variant, (level, fraction) in enumerate(AMS_VARIANTS):
            rng = np.random.default_rng(np.random.SeedSequence([AMS_SEED, case_index, rep, variant]))
            n = max(int(fraction * total / H), 2)
            values, weights, counts, used, iterations = ams_run(model, tables[level], n, total, rng)
            filled, _ = A.complete_rows(model, counts, "uniform")
            fitted = model_from_counts(model, filled)
            for a in AMS_ALPHAS:
                rows.append({"case": case_index, "budget": AMS_BUDGET, "replication": rep, "level": level,
                             "fraction": fraction, "alpha": a,
                             "error_weighted": f"{B.SPAN * (weighted_cvar(values, weights, a, False) - truth[a]):.9g}",
                             "error_selfnorm": f"{B.SPAN * (weighted_cvar(values, weights, a, True) - truth[a]):.9g}",
                             "error_fitted": f"{B.SPAN * (solve_stop_loss(fitted, a).cvar - truth[a]):.9g}",
                             "total_mass": f"{weights.sum():.6g}", "iterations": iterations, "charged_queries": used})
    return rows


def ams_check() -> None:
    """Unbiasedness of the weighted return distribution on separate test streams (no method comparison)."""
    AMS_OUT.mkdir(parents=True, exist_ok=True)
    report, worst, mass_gap = {"entries": []}, 0.0, 0.0
    for case_index in (0, 17):
        case = B.CASES[case_index]
        model = B.build_case(case["p_base"], case["loss"], case["policy"])
        V, _ = moments(model)
        tables = level_tables(model, V)
        values, masses = root_distribution(model)
        cdf = np.cumsum(masses)
        total = model.group_count * 300 // H * H
        cutoffs = {a: values[int(np.searchsorted(cdf, a, side="left"))] for a in (.1, .01, .001)}
        truth = {a: float(masses[values <= q + 1e-12].sum()) for a, q in cutoffs.items()}
        mean_truth = float(np.sum(values * masses))
        for v, (level, fraction) in enumerate([(lv, f) for lv in AMS_LEVELS for f in (0.45, 0.36)]):
            below, totals, means, iterations = {a: [] for a in cutoffs}, [], [], []
            for rep in range(400):
                rng = np.random.default_rng(np.random.SeedSequence([AMS_SEED, 900 + v, case_index, rep]))
                r, w, _, _, it = ams_run(model, tables[level], max(int(fraction * total / H), 2), total, rng)
                totals.append(w.sum())
                means.append(float(np.sum(w * r)))
                iterations.append(it)
                for a, q in cutoffs.items():
                    below[a].append(w[r <= q + 1e-12].sum())
            entry = {"case": case_index, "level": level, "fraction": fraction,
                     "mean_iterations": float(np.mean(iterations))}
            for name, sample, target in ([("total", totals, 1.0), ("mean_return", means, mean_truth)]
                                         + [(f"mass_below_q{a}", below[a], truth[a]) for a in cutoffs]):
                sample = np.asarray(sample)
                if name == "total":  # the weights of removed and final runs sum to one by construction
                    entry[name] = {"truth": target, "mean": float(sample.mean()),
                                   "largest_abs_gap": float(np.abs(sample - 1.0).max())}
                    mass_gap = max(mass_gap, entry[name]["largest_abs_gap"])
                    continue
                z = (sample.mean() - target) / (sample.std(ddof=1) / np.sqrt(sample.size))
                entry[name] = {"truth": target, "mean": float(sample.mean()), "z": float(z)}
                worst = max(worst, abs(z))
            report["entries"].append(entry)
    report["largest_abs_z"] = worst
    report["largest_total_mass_gap"] = mass_gap
    report["passed"] = bool(worst < 4.0 and mass_gap < 1e-9)
    (AMS_OUT / "check.json").write_text(json.dumps(report, indent=1) + "\n")
    for e in report["entries"]:
        print(e["case"], e["level"], e["fraction"], f"iterations {e['mean_iterations']:.0f}",
              {k: (round(v["truth"], 5), round(v["mean"], 5), round(v.get("z", 0.0), 2)) for k, v in e.items() if isinstance(v, dict)})
    print(f"largest |z| {worst:.2f}; passed {report['passed']}")


def ams(processes: int) -> None:
    if not json.loads((AMS_OUT / "check.json").read_text())["passed"]:
        raise SystemExit("check failed; no run")
    reps = AMS_REPLICATIONS
    jobs = [(c["index"], s, min(s + 10, reps)) for c in B.CASES for s in range(0, reps, 10)]
    with ProcessPoolExecutor(max_workers=processes) as pool, (AMS_OUT / "raw.csv").open("w", newline="") as stream:
        writer = None
        for done, chunk in enumerate(pool.map(job_wrapper, jobs), 1):
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(chunk[0]))
                writer.writeheader()
            writer.writerows(chunk)
            stream.flush()
            if done % 54 == 0:
                print(f"{done} of {len(jobs)} jobs", flush=True)


def job_wrapper(args):
    return ams_job(args)


def job_extension(args):
    """Further tilts of the fixed-time scheme, same design, new streams."""
    case_index, budget, start, stop, tilts = args
    case = B.CASES[case_index]
    model = B.build_case(case["p_base"], case["loss"], case["policy"])
    C = float(compute_influences(model, B.ALPHA).solution.cvar)
    V, sd = moments(model)
    n = model.group_count * budget // H
    rows = []
    for rep in range(start, stop):
        for index, level in enumerate(LEVELS):
            for c in tilts:
                rng = B._rng(case_index, budget, rep, 100 + 50 * index + int(round(10 * c)))
                returns, weights, counts = splitting_run(model, V, n, c / sd, level, rng)
                weighted, selfnorm, fitted, empty = estimates(model, returns, weights, counts)
                rows.append({"case": case_index, "budget": budget, "replication": rep, "level": level, "tilt": c,
                             "error_weighted": f"{B.SPAN * (weighted - C):.9g}",
                             "error_selfnorm": f"{B.SPAN * (selfnorm - C):.9g}",
                             "error_fitted": f"{B.SPAN * (fitted - C):.9g}",
                             "total_mass": f"{weights.sum():.6g}", "empty_blocks": empty,
                             "charged_queries": n * H})
    return rows


def extend(tilts, processes: int) -> None:
    reps = int(B.CONFIG["replications"])
    jobs = [(c["index"], b, s, min(s + 25, reps), tuple(tilts)) for c in B.CASES for b in BUDGETS
            for s in range(0, reps, 25)]
    rows = []
    with ProcessPoolExecutor(max_workers=processes) as pool:
        for chunk in pool.map(job_extension, jobs):
            rows.extend(chunk)
    rows.sort(key=lambda r: (r["case"], r["budget"], r["replication"], r["level"], r["tilt"]))
    with (B.OUT / "splitting_raw_ext.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows")


def check() -> None:
    report = {"alpha": B.ALPHA, "c0_max_abs_diff": 0.0, "unbiasedness": []}
    worst = 0.0
    for case_index in (0, 17):
        case = B.CASES[case_index]
        model = B.build_case(case["p_base"], case["loss"], case["policy"])
        V, sd = moments(model)
        values, masses = root_distribution(model)
        cdf = np.cumsum(masses)
        n = model.group_count * 300 // H
        for rep in range(5):
            r, w, _ = splitting_run(model, V, n, 0.0, "profit", B._rng(case_index, 300, rep, 900))
            report["c0_max_abs_diff"] = max(report["c0_max_abs_diff"],
                                            abs(weighted_cvar(r, w, B.ALPHA, False) - _empirical_cvar(r, B.ALPHA)))
        cutoffs = {a: values[int(np.searchsorted(cdf, a, side="left"))] for a in (.1, .05, .01)}
        truth = {a: float(masses[values <= q + 1e-12].sum()) for a, q in cutoffs.items()}
        for v, (level, c) in enumerate([(lv, c) for lv in LEVELS for c in (1.0, 3.0)]):
            below = {a: [] for a in cutoffs}
            total = []
            for rep in range(400):
                r, w, _ = splitting_run(model, V, n, c / sd, level, B._rng(case_index, 300, rep, 901 + v))
                total.append(w.sum())
                for a, q in cutoffs.items():
                    below[a].append(w[r <= q + 1e-12].sum())
            entry = {"case": case_index, "level": level, "tilt": c, "sd_encoded": sd}
            for name, sample, target in [("total", total, 1.0)] + [(f"mass_below_q{a}", below[a], truth[a])
                                                                    for a in cutoffs]:
                sample = np.asarray(sample)
                z = (sample.mean() - target) / (sample.std(ddof=1) / np.sqrt(sample.size))
                entry[name] = {"truth": target, "mean": float(sample.mean()), "z": float(z)}
                worst = max(worst, abs(z))
            report["unbiasedness"].append(entry)
    report["largest_abs_z"] = worst
    report["passed"] = bool(report["c0_max_abs_diff"] < 1e-12 and worst < 4.0)
    (B.OUT / "splitting_check.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"alpha {B.ALPHA}: c=0 max |diff| {report['c0_max_abs_diff']:.1e}; largest |z| {worst:.2f}; "
          f"passed {report['passed']}")


def run(processes: int) -> None:
    if not json.loads((B.OUT / "splitting_check.json").read_text())["passed"]:
        raise SystemExit("check failed; no run")
    reps = int(B.CONFIG["replications"])
    jobs = [(c["index"], b, s, min(s + 25, reps)) for c in B.CASES for b in BUDGETS for s in range(0, reps, 25)]
    rows = []
    with ProcessPoolExecutor(max_workers=processes) as pool:
        for chunk in pool.map(job, jobs):
            rows.extend(chunk)
    rows.sort(key=lambda r: (r["case"], r["budget"], r["replication"], r["level"], r["tilt"]))
    with (B.OUT / "splitting_raw.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows")


def analyze() -> None:
    """Analysis: both splitting schemes, all variants and budgets, the best variant in hindsight, and the
    per-case hindsight choice."""
    cases = [c["index"] for c in B.CASES]
    reps = int(B.CONFIG["replications"])
    split = defaultdict(lambda: np.full(reps, np.nan))
    mass = defaultdict(list)
    for path in (B.OUT / "splitting_raw.csv", B.OUT / "splitting_raw_ext.csv"):
        if not path.exists():
            continue
        for row in csv.DictReader(path.open()):
            for est in ("weighted", "selfnorm", "fitted"):
                key = (int(row["case"]), int(row["budget"]), row["level"], float(row["tilt"]), est)
                split[key][int(row["replication"])] = float(row[f"error_{est}"]) ** 2
            mass[(row["level"], float(row["tilt"]))].append(float(row["total_mass"]))
    variants = sorted({(k[2], k[3]) for k in split})
    adaptive = defaultdict(lambda: np.full(AMS_REPLICATIONS, np.nan))
    if (AMS_OUT / "raw.csv").exists():
        for row in csv.DictReader((AMS_OUT / "raw.csv").open()):
            if abs(float(row["alpha"]) - B.ALPHA) < 1e-12:
                for est in ("weighted", "selfnorm", "fitted"):
                    key = (int(row["case"]), int(row["budget"]), "ams_" + row["level"], float(row["fraction"]), est)
                    adaptive[key][int(row["replication"])] = float(row[f"error_{est}"]) ** 2
    reference = defaultdict(lambda: np.full(reps, np.nan))
    for row in csv.DictReader((B.OUT / "raw.csv").open()):
        reference[(int(row["case"]), int(row["budget"]), row["method"])][int(row["replication"])] = \
            float(row["error_profit"]) ** 2
    if (B.OUT / "fitted_rollouts.csv").exists():
        source = "fitted_rollouts.csv (scripts/rebuttal_fitted_rollouts_alpha.py)"
        for row in csv.DictReader((B.OUT / "fitted_rollouts.csv").open()):
            if row["method"] == "rollout_model":
                reference[(int(row["case"]), int(row["budget"]), "rollout_model")][int(row["replication"])] = \
                    float(row["error_profit"]) ** 2
    elif abs(B.ALPHA - 0.1) < 1e-12:
        source = "results/rebuttal_checkpoint_access/raw.csv"
        for row in csv.DictReader((ROOT / "results/rebuttal_checkpoint_access/raw.csv").open()):
            if row["setting"] != "cliffwalking" and row["arm"] == "rollout_model":
                reference[(int(row["case"]), int(row["budget"]), "rollout_model")][int(row["replication"])] = \
                    float(row["error"]) ** 2
    else:  # alpha .001 and .0001: the run without resampling, evaluated by the fitted estimator
        source = "this experiment's runs without resampling"
        for c in cases:
            for b in BUDGETS:
                reference[(c, b, "rollout_model")] = split[(c, b, "profit", 0.0, "fitted")]
    assert not any(np.isnan(v).any() for v in split.values())
    assert not any(np.isnan(v).any() for v in adaptive.values())
    rng = np.random.default_rng(B.SEED + 7)
    resamples = int(B.CONFIG["bootstrap_resamples"])

    def family(arrays):
        return float(np.mean([a.mean() for a in arrays]))

    def contrast(top, bottom):
        ratio = family(top) / family(bottom)
        num = np.mean([a[rng.integers(0, a.size, size=(resamples, a.size))].mean(axis=1) for a in top], axis=0)
        den = np.mean([a[rng.integers(0, a.size, size=(resamples, a.size))].mean(axis=1) for a in bottom], axis=0)
        boot = num / den
        return {"family_ratio": ratio, "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}

    def z(a, b):
        return float((a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))

    out = {"alpha": B.ALPHA, "fitted_model_rollouts_from": source,
           "mean_total_mass": {f"{lv}|{c}": float(np.mean(v)) for (lv, c), v in mass.items()}, "budgets": {}}
    refs = ("complete_rollout", "rollout_model", "tis", "tis_anchored", "learned_mean")
    for b in BUDGETS:
        ref = {m: [reference[(c, b, m)] for c in cases] for m in refs}
        assert not any(np.isnan(a).any() for arrays in ref.values() for a in arrays)
        entry = {"family_mse": {m: family(ref[m]) for m in refs}, "variants": {}, "best": {}}
        arms = {}
        for level, c in variants:
            for est in ("weighted", "selfnorm", "fitted"):
                arms[(level, c, est)] = [split[(case, b, level, c, est)] for case in cases]
        for key in sorted({k[2:] for k in adaptive if k[1] == b}):
            arms[key] = [adaptive[(case, b) + key] for case in cases]
        for (level, c, est), arrays in arms.items():
            entry["variants"][f"{level}|{c}|{est}"] = {
                "family_mse": family(arrays),
                "ratio_to_complete_rollouts": family(arrays) / family(ref["complete_rollout"])}
        for level in sorted({k[0] for k in arms}):
            for label, ests in (("own", ("weighted", "selfnorm")), ("fitted", ("fitted",))):
                key = min((k for k in arms if k[0] == level and k[1] > 0 and k[2] in ests),
                          key=lambda k: family(arms[k]))
                grid = sorted({k[1] for k in arms if k[0] == level and k[1] > 0})
                best = {"variant": {"parameter": key[1], "estimator": key[2]}, "family_mse": family(arms[key]),
                        "parameter_at_edge_of_grid": bool(key[1] in (grid[0], grid[-1])),
                        "vs_complete_rollouts": contrast(arms[key], ref["complete_rollout"])}
                for m in ("tis", "tis_anchored", "rollout_model", "learned_mean"):
                    best[f"{m}_over_best"] = contrast(ref[m], arms[key])
                zs = [z(ref["tis"][i], arms[key][i]) for i in range(len(cases))]
                best["tis_lower_in_cases"] = int(sum(ref["tis"][i].mean() < arms[key][i].mean()
                                                     for i in range(len(cases))))
                best["tis_resolved_better"] = int(sum(x <= -2 for x in zs))
                best["tis_resolved_worse"] = int(sum(x >= 2 for x in zs))
                entry["best"][f"{level}|{label}"] = best
            own = min((k for k in arms if k[0] == level and k[1] > 0 and k[2] != "fitted"),
                      key=lambda k: family(arms[k]))
            entry["best"][f"{level}|fitted_same_run_over_own"] = contrast(arms[(level, own[1], "fitted")], arms[own])
        # per-case hindsight: for every case the splitting variant with the lowest MSE in that case
        pools = {"model_free_levels": ("profit", "ams_time"), "all_levels": tuple(sorted({k[0] for k in arms}))}
        for name, levels in pools.items():
            keys = [k for k in arms if k[0] in levels and k[1] > 0 and k[2] != "fitted"]
            floor = [min(arms[k][i].mean() for k in keys) for i in range(len(cases))]
            entry["best"][f"per_case_hindsight|{name}"] = {
                "variants_in_pool": len(keys),
                "tis_over_per_case_best": family(ref["tis"]) / float(np.mean(floor)),
                "tis_anchored_over_per_case_best": family(ref["tis_anchored"]) / float(np.mean(floor)),
                "rollout_model_over_per_case_best": family(ref["rollout_model"]) / float(np.mean(floor)),
                "tis_lower_in_cases": int(sum(ref["tis"][i].mean() < floor[i] for i in range(len(cases))))}
        entry["check_c0_over_complete_rollouts"] = contrast(arms[("profit", 0.0, "weighted")], ref["complete_rollout"])
        out["budgets"][str(b)] = entry
    (B.OUT / "splitting_analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"alpha {B.ALPHA}; fitted-model rollouts from {source}")
    for b, entry in out["budgets"].items():
        print(f" budget {b}: c=0 / complete rollouts {entry['check_c0_over_complete_rollouts']['family_ratio']:.3f}; "
              f"family MSE {({k: round(v, 5) for k, v in entry['family_mse'].items()})}")
        for name, best in entry["best"].items():
            if name.startswith("per_case"):
                print(f"   {name}: TIS/per-case-best {best['tis_over_per_case_best']:.3f}, anchor "
                      f"{best['tis_anchored_over_per_case_best']:.3f}, fitted rollouts "
                      f"{best['rollout_model_over_per_case_best']:.3f}; TIS lower in {best['tis_lower_in_cases']}/18 "
                      f"({best['variants_in_pool']} variants)")
            elif "variant" not in best:
                print(f"   {name}: {best['family_ratio']:.3f} {[round(x, 3) for x in best['ci95']]}")
            else:
                print(f"   best {name}: {best['variant']} / complete {best['vs_complete_rollouts']['family_ratio']:.3f}; "
                      f"TIS/best {best['tis_over_best']['family_ratio']:.3f} "
                      f"{[round(x, 3) for x in best['tis_over_best']['ci95']]}; "
                      f"anchor/best {best['tis_anchored_over_best']['family_ratio']:.3f}; "
                      f"fitted rollouts/best {best['rollout_model_over_best']['family_ratio']:.3f}; "
                      f"TIS lower in {best['tis_lower_in_cases']}/18 "
                      f"(resolved {best['tis_resolved_better']} better, {best['tis_resolved_worse']} worse)")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "check":
        check()
    elif step == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif step == "analyze":
        analyze()
    elif step == "extend":
        extend([float(x) for x in sys.argv[2].split(",")], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    elif step == "ams_check":
        ams_check()
    elif step == "ams":
        ams(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    else:
        raise SystemExit("usage: rebuttal_splitting_baseline.py check | run [processes] | analyze | ams_check | ams [processes]")
