#!/usr/bin/env python3
"""Fitted-model rollouts at the tail levels .05 and .01 (Appendix F.4).

For every case, budget, and replication of the inventory raw-profit study at
that tail level, applies the categorical estimator to exactly the
trajectories of its complete rollouts (same seed and rollout stream); their
empirical CVaR must reproduce the complete-rollout errors. The tail level
comes from REBUTTAL_PROFIT_CONFIG.

Usage:
  REBUTTAL_PROFIT_CONFIG=configs/rebuttal_inventory_profit_alpha01.json \
      python scripts/rebuttal_fitted_rollouts_alpha.py run [processes] | analyze
Writes <output directory of the config>/fitted_rollouts.csv and fitted_rollouts_analysis.json.
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
if "REBUTTAL_PROFIT_CONFIG" not in os.environ:
    raise SystemExit("set REBUTTAL_PROFIT_CONFIG to the alpha .05 or .01 config")

import rebuttal_cliffwalking_attribution as A  # noqa: E402
import rebuttal_inventory_profit as B  # noqa: E402  (loads the D config from the environment)
from tis.engine import compute_influences, solve_stop_loss  # noqa: E402
from tis.llm_study.runner import _empirical_cvar  # noqa: E402
from tis.model import model_from_counts  # noqa: E402

assert B.SEED in (20261110, 20261111), "not the alpha .05 or .01 configuration"
METHODS = ("tis", "tis_anchored", "learned_mean", "occ_plus_mean", "learned_occupancy", "uniform")


def rollout_with_counts(model, runs: int, rng: np.random.Generator):
    """Stage-state trajectories with per-block outcome counts.

    Same draw order as scripts/tabular_rollout.py::rollout_returns (untied
    branch) and scripts/rebuttal_checkpoint_access.py::rollout_with_counts.
    """
    counts = {g: np.zeros(model.kernels[g].size, dtype=int) for g in model.query_groups}
    state = np.full(runs, model.initial_state, dtype=int)
    totals = np.zeros(runs)
    for step in range(model.horizon):
        h = model.horizon - step
        successor = np.empty_like(state)
        for current in np.unique(state):
            mask = state == current
            group = (h, int(current))
            kernel = model.kernels[group]
            outcomes = rng.choice(kernel.size, size=int(mask.sum()), p=kernel.probabilities)
            totals[mask] += kernel.rewards[outcomes]
            successor[mask] = kernel.next_states[outcomes]
            if group in counts:
                counts[group] += np.bincount(outcomes, minlength=kernel.size)
        state = successor
    return totals, counts


def job(args):
    case_index, budget, start, stop = args
    case = B.CASES[case_index]
    model = B.build_case(case["p_base"], case["loss"], case["policy"])
    assert model.group_mode == "untied_layer_state"
    C = float(compute_influences(model, B.ALPHA).solution.cvar)
    runs = model.group_count * budget // model.horizon
    rows = []
    for rep in range(start, stop):
        returns, counts = rollout_with_counts(model, runs, B._rng(case_index, budget, rep, 6))
        filled, empty = A.complete_rows(model, counts, "uniform")
        estimate = solve_stop_loss(model_from_counts(model, filled), B.ALPHA).cvar
        base = {"case": case_index, "budget": budget, "replication": rep, "charged_queries": runs * model.horizon,
                "empty_blocks": len(empty)}
        rows.append({**base, "method": "rollout_model", "error_profit": B.SPAN * (estimate - C)})
        rows.append({**base, "method": "rollout_empirical_check",
                     "error_profit": B.SPAN * (_empirical_cvar(returns, B.ALPHA) - C)})
    return rows


def run(processes: int) -> None:
    reps = int(B.CONFIG["replications"])
    jobs = [(c["index"], b, s, min(s + 25, reps)) for c in B.CASES
            for b in B.CONFIG["budgets_per_block"] for s in range(0, reps, 25)]
    rows = []
    with ProcessPoolExecutor(max_workers=processes) as pool:
        for chunk in pool.map(job, jobs):
            rows.extend(chunk)
    rows.sort(key=lambda r: (r["case"], r["budget"], r["replication"], r["method"]))
    with (B.OUT / "fitted_rollouts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows")


def analyze() -> None:
    errors = defaultdict(dict)
    for path in (B.OUT / "raw.csv", B.OUT / "fitted_rollouts.csv"):
        for row in csv.DictReader(path.open()):
            errors[(int(row["case"]), int(row["budget"]), row["method"])][int(row["replication"])] = \
                float(row["error_profit"])
    cases = [c["index"] for c in B.CASES]
    budgets = B.CONFIG["budgets_per_block"]
    reps = range(int(B.CONFIG["replications"]))
    worst = max(abs(errors[(c, b, "rollout_empirical_check")][r] - errors[(c, b, "complete_rollout")][r])
                for c in cases for b in budgets for r in reps)

    def sq(key):
        return np.array([errors[key][r] ** 2 for r in reps])

    def z(a, r, paired):
        sa, sr = sq(a), sq(r)
        if paired:
            d = sa - sr
            return float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size)))
        return float((sa.mean() - sr.mean()) / np.sqrt(sa.var(ddof=1) / sa.size + sr.var(ddof=1) / sr.size))

    out = {"alpha": B.ALPHA, "check_empirical_reproduces_D_max_abs_diff": float(worst), "contrasts": {}}
    pairs = [(m, "rollout_model", False) for m in METHODS] + [("rollout_model", "complete_rollout", True)]
    for b in budgets:
        for a, r, paired in pairs:
            zs = [z((c, b, a), (c, b, r), paired) for c in cases]
            ratios = [sq((c, b, a)).mean() / sq((c, b, r)).mean() for c in cases]
            out["contrasts"][f"{a}_vs_{r}|{b}"] = {
                "family_ratio": float(np.mean([sq((c, b, a)).mean() for c in cases])
                                      / np.mean([sq((c, b, r)).mean() for c in cases])),
                "median_ratio": float(np.median(ratios)), "min_ratio": float(min(ratios)),
                "max_ratio": float(max(ratios)), "resolved_better": int(sum(x <= -2 for x in zs)),
                "resolved_worse": int(sum(x >= 2 for x in zs)), "paired": paired}
    (B.OUT / "fitted_rollouts_analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"alpha {B.ALPHA}: empirical CVaR reproduces D's complete rollouts (max |diff| {worst:.1e})")
    for k, v in out["contrasts"].items():
        print(f"  {k:40s} family {v['family_ratio']:.3f} median {v['median_ratio']:.3f} "
              f"better {v['resolved_better']} worse {v['resolved_worse']}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif step == "analyze":
        analyze()
    else:
        raise SystemExit("usage: REBUTTAL_PROFIT_CONFIG=... rebuttal_fitted_rollouts_alpha.py run [processes] | analyze")
