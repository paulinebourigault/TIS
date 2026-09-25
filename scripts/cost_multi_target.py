#!/usr/bin/env python3
"""Multi-target cost-to-accuracy analysis, censor-preserving.

Design decisions:

1. Censoring is preserved: a left-censored crossing contributes the bound
   [0, smallest tested budget], a right-censored one [largest budget,
   infinity]; every draw contributes bounds on the budget ratio, and the
   5th percentile of lower bounds / 95th percentile of upper bounds form
   the reported envelope. Censored-draw counts are reported per target.
2. Replication data are used where the archive contains them, with an
   explicit coupling map: methods sharing coupled streams within a
   replication are resampled jointly; complete rollouts use their own
   independent stream and are resampled independently. Budgets are
   resampled independently. Pairs without raw data fall back to a
   summary-only Gaussian sensitivity (MSE ~ N(mse, mse_se), clipped at
   zero with the clip count reported) and are labeled as such.
3. All selection choices are recorded in the output metadata.
4. The inventory learned-mean comparison is included (summary-only:
   its replication file was not retained).
5. Interpretation: pointwise percentile summaries at 21 targets on the
   trimmed common log-RMSE range, conditional on the tested grid and the
   first-crossing log-log interpolation convention; they are not
   simultaneous confidence bands.

Writes results/cost_multi_target.json and prints a summary.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GAUSS_SEED, GAUSS_DRAWS = 20260925, 2000
RAW_SEED, RAW_DRAWS = 20260926, 10000
NTAU, TRIM = 21, 0.02

sys.path.insert(0, str(ROOT / "scripts"))
from cost_to_accuracy import tabular_curves, finqa_curves  # noqa: E402

PAIRS = {
    "cliffwalking": [("tis", "uniform"), ("tis", "learned_mean"),
                     ("tis", "learned_occupancy"), ("tis", "complete_rollout"),
                     ("tail_anchored", "learned_occupancy"),
                     ("tail_anchored", "complete_rollout")],
    "inventory": [("tis", "uniform"), ("tail_anchored", "uniform"),
                  ("tis", "learned_mean"), ("tis", "learned_occupancy"),
                  ("tis", "complete_rollout")],
    "finqa_phi_ordinary": [("tis_anchored", "uniform"),
                           ("tis_anchored", "learned_occupancy"),
                           ("tis_anchored", "complete_rollout")],
}
# Per-cell FinQA tasks including the 400/800 extension budgets; added
# only when those results exist.
FINQA_EXT_RUNS = {
    "finqa_qwen_ordinary_ext": "finqa_heldout_ordinary",
    "finqa_phi_ordinary_ext": "finqa_heldout_ordinary_phi",
    "finqa_qwen_unitcheck_ext": "finqa_heldout_unitcheck",
    "finqa_phi_unitcheck_ext": "finqa_heldout_unitcheck_phi",
}
FINQA_EXT_PAIRS = [("tis_anchored", "uniform"),
                   ("tis_anchored", "learned_occupancy"),
                   ("tis_anchored", "complete_rollout"),
                   ("tis", "learned_occupancy"),
                   ("tis", "complete_rollout")]
# Replication files retained in the release, and the methods they hold.
RAW_SOURCES = {
    "cliffwalking": ("results/public_paper/raw.csv", "cliffwalking_primary", 1),
    "inventory": ("results/replacement_paper/raw.csv", "inventory_untied_v1", 1),
    "finqa_phi_ordinary": ("results/finqa_heldout_ordinary_phi/panel_replications.csv",
                           None, 50 * 73),
}
INDEPENDENT_STREAMS = {"complete_rollout"}


def curve_arrays(curves, method):
    pts = sorted(curves[method].items())
    b = np.array([p[0] for p in pts])
    mse = np.array([p[1][0] ** 2 for p in pts])
    mse_se = np.array([2.0 * p[1][0] * p[1][1] for p in pts])
    return b, mse, mse_se


def budget_at(b, rmse, tau):
    below = np.nonzero(rmse <= tau)[0]
    if below.size == 0:
        return None
    i = below[0]
    if i == 0:
        return ("below", float(b[0]))
    lb = np.log(b[i - 1]) + (np.log(tau) - np.log(rmse[i - 1])) * (
        np.log(b[i]) - np.log(b[i - 1])) / (np.log(rmse[i]) - np.log(rmse[i - 1]))
    return float(np.exp(lb))


def crossing_bounds(budgets, rmse, tau):
    x = budget_at(budgets, rmse, tau)
    if x is None:
        return float(budgets[-1]), np.inf, "right"
    if isinstance(x, tuple):
        return 0.0, float(budgets[0]), "left"
    return x, x, "interior"


def percentile(a, p):
    a = np.sort(np.asarray(a, dtype=float))
    h = (len(a) - 1) * p
    i, j = int(np.floor(h)), int(np.ceil(h))
    if i == j or a[i] == a[j]:
        return float(a[i])
    if not np.isfinite(a[j]):
        return float(a[j])
    return float(a[i] + (h - i) * (a[j] - a[i]))


def target_grid(curves, method, reference):
    _, mm, _ = curve_arrays(curves, method)
    _, mr, _ = curve_arrays(curves, reference)
    lo = max(np.sqrt(mm).min(), np.sqrt(mr).min())
    hi = min(np.sqrt(mm).max(), np.sqrt(mr).max())
    if not lo < hi:
        return None
    span = np.log(hi) - np.log(lo)
    return np.exp(np.linspace(np.log(lo) + TRIM * span,
                              np.log(hi) - TRIM * span, NTAU))


def analyze_draws(curves, method, reference, rm_draws, rr_draws, taus, source):
    bm, mm, _ = curve_arrays(curves, method)
    br, mr, _ = curve_arrays(curves, reference)
    rows = []
    for tau in taus:
        lows, highs = [], []
        statuses = defaultdict(int)
        for rm, rr in zip(rm_draws, rr_draws):
            ml, mh, ms = crossing_bounds(bm, rm, tau)
            rl, rh, rs = crossing_bounds(br, rr, tau)
            statuses[f"{ms}|{rs}"] += 1
            lows.append(0.0 if not np.isfinite(rh) else ml / rh)
            highs.append(np.inf if rl == 0 else mh / rl)
        low, high = percentile(lows, 0.05), percentile(highs, 0.95)
        pm = budget_at(bm, np.sqrt(mm), tau)
        pr = budget_at(br, np.sqrt(mr), tau)
        point = ((pm[1] if isinstance(pm, tuple) else pm) /
                 (pr[1] if isinstance(pr, tuple) else pr)
                 if pm is not None and pr is not None else None)
        censored = len(rm_draws) - statuses.get("interior|interior", 0)
        rows.append({"tau": float(tau), "ratio_point": point,
                     "envelope_p5": low,
                     "envelope_p95": (high if np.isfinite(high) else "infinity"),
                     "censored_draws": censored,
                     "status_counts_method_reference": dict(statuses),
                     "envelope_below_one": bool(high < 1.0),
                     "envelope_above_one": bool(low > 1.0)})
    finite = [r["ratio_point"] for r in rows if r["ratio_point"] is not None]
    return {"uncertainty_source": source, "targets": rows,
            "n_targets": len(rows),
            "n_envelope_method_cheaper": sum(r["envelope_below_one"] for r in rows),
            "n_envelope_reference_cheaper": sum(r["envelope_above_one"] for r in rows),
            "max_censored_fraction": max(r["censored_draws"] for r in rows) / len(rm_draws),
            "point_ratio_range": ([float(min(finite)), float(max(finite))]
                                  if finite else None)}


def load_replications(task):
    if task in FINQA_EXT_RUNS:
        run = FINQA_EXT_RUNS[task]
        paths = [ROOT / f"results/{run}/panel_replications.csv",
                 ROOT / f"results/{run}_b48/panel_replications.csv"]
        env, factor = None, 50 * 73
    else:
        path, env, factor = RAW_SOURCES[task]
        paths = [ROOT / path]
    if not all(p.exists() for p in paths):
        return None, factor
    panel = env is None
    rows = defaultdict(dict)
    for path in paths:
        for r in csv.DictReader(path.open()):
            if env is not None and r["environment"] != env:
                continue
            budget = int(r["budget_per_group"]) if panel else int(float(r["total_budget"]))
            value = float(r["panel_mse"] if panel else r["squared_error"])
            rows[budget, int(r["replication"])][r["method"]] = value
    data = {}
    for budget in sorted({k[0] for k in rows}):
        reps = sorted(rep for b, rep in rows if b == budget)
        data[budget] = {m: np.array([rows[budget, rep][m] for rep in reps])
                        for m in rows[budget, reps[0]]}
    return data, factor


def empirical_draws(curves, data, factor, method, reference):
    """Joint resampling of coupled replication rows; independent streams
    (rollouts) resample independently. Budgets resample independently.
    Verifies each retained column reproduces the summary MSE exactly."""
    rng = np.random.default_rng(RAW_SEED)
    a, b = [], []
    for budget in sorted(data):
        arr = data[budget]
        x, y = arr[method], arr[reference]
        assert len(x) == len(y)
        for m, v in ((method, x), (reference, y)):
            assert abs(v.mean() - curves[m][budget * factor][0] ** 2) < 1e-12, m
        ix = rng.integers(0, len(x), size=(RAW_DRAWS, len(x)))
        iy = (rng.integers(0, len(y), size=(RAW_DRAWS, len(y)))
              if (method in INDEPENDENT_STREAMS) != (reference in INDEPENDENT_STREAMS)
              else ix)
        a.append(np.sqrt(x[ix].mean(axis=1)))
        b.append(np.sqrt(y[iy].mean(axis=1)))
    return np.array(a).T, np.array(b).T


def gaussian_draws(curves, method, reference, rng):
    _, mm, sm = curve_arrays(curves, method)
    _, mr, sr = curve_arrays(curves, reference)
    a = mm + sm * rng.standard_normal((GAUSS_DRAWS, mm.size))
    b = mr + sr * rng.standard_normal((GAUSS_DRAWS, mr.size))
    clipped = int((a <= 0).sum() + (b <= 0).sum())
    return (np.sqrt(np.maximum(a, 1e-300)), np.sqrt(np.maximum(b, 1e-300)),
            clipped)


def main() -> None:
    tasks = {"cliffwalking": tabular_curves("cliffwalking"),
             "inventory": tabular_curves("inventory"),
             "finqa_phi_ordinary": finqa_curves()}
    pairs = dict(PAIRS)
    for task, run in FINQA_EXT_RUNS.items():
        if (ROOT / f"results/{run}_b48/panel_replications.csv").exists():
            tasks[task] = finqa_curves(run, include_extension=True)
            pairs[task] = FINQA_EXT_PAIRS
    out = {"protocol": {
        "version": 2,
        "targets_per_pair": NTAU,
        "target_selection": "geometric grid over the common attained log-RMSE "
                            "range, trimmed 2% of the log-span at each end; "
                            "retrospective",
        "crossing_rule": "first crossing, log-log interpolation between "
                         "adjacent tested budgets, no extrapolation",
        "censoring": "left crossing bounded by [0, smallest tested budget], "
                     "right by [largest tested budget, infinity]; every draw "
                     "contributes ratio bounds; none discarded",
        "resampling": {"empirical": {"draws": RAW_DRAWS, "seed": RAW_SEED,
                                     "unit": "replication rows, jointly for "
                                             "coupled methods, independently "
                                             "for rollout streams and across "
                                             "budgets"},
                       "gaussian_summary_only": {"draws": GAUSS_DRAWS,
                                                 "seed": GAUSS_SEED,
                                                 "note": "MSE ~ N(mse, mse_se) "
                                                         "clipped at zero; used "
                                                         "only where replication "
                                                         "files were not retained"}},
        "interpretation": "pointwise exploratory percentile envelopes, "
                          "conditional on the tested grid, target selection, "
                          "and interpolation convention; not simultaneous "
                          "confidence bands; observed grid brackets are in "
                          "results/cost_to_accuracy.json"}}
    grng = np.random.default_rng(GAUSS_SEED)
    for task, curves in tasks.items():
        out[task] = {}
        data, factor = load_replications(task)
        for method, reference in pairs[task]:
            key = f"{method}_vs_{reference}"
            taus = target_grid(curves, method, reference)
            if taus is None:
                bm, mm, _ = curve_arrays(curves, method)
                br, mr, _ = curve_arrays(curves, reference)
                out[task][key] = {
                    "note": "no overlapping attained RMSE range",
                    "method_rmse_range": [float(np.sqrt(mm).min()),
                                          float(np.sqrt(mm).max())],
                    "reference_rmse_range": [float(np.sqrt(mr).min()),
                                             float(np.sqrt(mr).max())]}
                print(f"{task}: {method} vs {reference}: no overlap "
                      f"(budget-censored)")
                continue
            have_raw = (data is not None and
                        all(m in data[min(data)] for m in (method, reference)))
            if have_raw:
                a, b = empirical_draws(curves, data, factor, method, reference)
                r = analyze_draws(curves, method, reference, a, b, taus,
                                  "empirical replication bootstrap")
            else:
                a, b, clipped = gaussian_draws(curves, method, reference, grng)
                r = analyze_draws(curves, method, reference, a, b, taus,
                                  "gaussian summary-only sensitivity")
                r["clipped_mse_cells"] = clipped
            out[task][key] = r
            pr = r["point_ratio_range"]
            print(f"{task}: {method} vs {reference} [{r['uncertainty_source']}]"
                  f": point ratios {pr[0]:.2f}-{pr[1]:.2f}, envelopes "
                  f"below/above one {r['n_envelope_method_cheaper']}/"
                  f"{r['n_envelope_reference_cheaper']} of {r['n_targets']}, "
                  f"max censored fraction {r['max_censored_fraction']:.3f}")
    (ROOT / "results/cost_multi_target.json").write_text(
        json.dumps(out, indent=1) + "\n")
    print("written results/cost_multi_target.json")


if __name__ == "__main__":
    main()
