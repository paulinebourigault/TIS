#!/usr/bin/env python3
"""Complete-rollout CVaR rows for the tabular tasks.

Adds the missing matched-cost rollout comparison to CliffWalking,
inventory, and the controlled replacement task, without modifying the
verified runner: an independent comparison with its own seed
(streams are not coupled to the existing runs, so uncertainty is
reported from independent replications).

Accounting: one rollout consumes one environment sample per step
(sampling the known policy is free), so a task with horizon H yields
floor(total_budget / H) trajectories at the same total sampling cost
the conditional methods receive. Termination: CliffWalking's goal state
is absorbing through its retained self-loop kernel and cliff entries
reset to the start inside the kernel law, so every trajectory runs
exactly H steps; the untied tasks have no absorbing states. The
empirical CVaR of the sampled returns is compared with the same exact
categorical truth the existing summaries report (cross-checked against
their truth column).

Writes results/tabular_rollout/summary.csv.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tis.engine import compute_influences  # noqa: E402
from tis.environments import make_environment  # noqa: E402
from tis.llm_study.runner import _empirical_cvar  # noqa: E402

MASTER_SEED = 20260922
TASKS = [
    # (environment, alpha, budgets, budget_unit, replications)
    ("cliffwalking_primary", 0.1, [50, 100, 200, 400], "per_group", 500),
    ("inventory_untied_v1", 0.1, [150, 300, 600, 1200], "per_group", 300),
    ("controlled_untied_v1", 0.1, [12500, 25000, 50000, 100000], "total", 400),
]


def rollout_returns(model, runs: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate `runs` H-step trajectories through the true laws."""
    horizon = model.horizon
    untied = model.group_mode == "untied_layer_state"
    state = np.full(runs, model.initial_state, dtype=int)
    totals = np.zeros(runs)
    for step in range(horizon):
        h = horizon - step  # layers count down: h steps remaining
        successor = np.empty_like(state)
        for current in np.unique(state):
            mask = state == current
            count = int(mask.sum())
            if untied:
                kernel = model.kernels[(h, int(current))]
                outcomes = rng.choice(kernel.probabilities.size, size=count,
                                      p=kernel.probabilities)
                totals[mask] += kernel.rewards[outcomes]
                successor[mask] = kernel.next_states[outcomes]
            else:
                probs = np.asarray(model.policy[h, int(current)], dtype=float)
                probs = probs / probs.sum()
                actions = rng.choice(probs.size, size=count, p=probs)
                sub_successor = np.empty(count, dtype=int)
                sub_rewards = np.empty(count)
                for action in np.unique(actions):
                    amask = actions == action
                    kernel = model.kernels[(int(current), int(action))]
                    outs = rng.choice(kernel.probabilities.size,
                                      size=int(amask.sum()),
                                      p=kernel.probabilities)
                    sub_rewards[amask] = kernel.rewards[outs]
                    sub_successor[amask] = kernel.next_states[outs]
                totals[mask] += sub_rewards
                successor[mask] = sub_successor
        state = successor
    return totals


def _reference_truth(env_name: str):
    """Cross-check against the truth column of the committed summaries."""
    source = {"cliffwalking_primary": "results/public_paper/summary.csv",
              "inventory_untied_v1": "results/replacement_paper/summary.csv",
              "controlled_untied_v1": "results/replacement_paper/summary.csv"}[env_name]
    try:
        for r in csv.DictReader(open(ROOT / source)):
            if r["environment"] == env_name:
                return float(r["truth"])
    except FileNotFoundError:
        return None
    return None


def main() -> None:
    rows = []
    raw_rows = []
    for env_name, alpha, budgets, unit, replications in TASKS:
        model = make_environment(env_name)
        truth = float(compute_influences(model, alpha).solution.cvar)
        ref = _reference_truth(env_name)
        if ref is not None:
            assert abs(truth - ref) < 1e-9, (env_name, truth, ref)
        groups = model.group_count
        for budget in budgets:
            total = budget * groups if unit == "per_group" else budget
            runs = total // model.horizon
            errors = np.empty(replications)
            for rep in range(replications):
                env_tag = int(hashlib.sha256(env_name.encode()).hexdigest()[:8], 16)
                seed = np.random.SeedSequence([MASTER_SEED, env_tag, total, rep])
                returns = rollout_returns(model, runs,
                                          np.random.default_rng(seed))
                errors[rep] = _empirical_cvar(returns, alpha) - truth
                raw_rows.append({"environment": env_name, "total_budget": total,
                                 "replication": rep, "method": "complete_rollout",
                                 "truth": truth, "error": float(errors[rep])})
            sq = errors ** 2
            rows.append({
                "environment": env_name, "alpha": alpha,
                "budget_per_group": budget if unit == "per_group" else "",
                "total_budget": total, "method": "complete_rollout",
                "replications": replications, "runs_per_replication": runs,
                "truth": truth, "mse": float(sq.mean()),
                "mse_se": float(sq.std(ddof=1) / np.sqrt(len(sq))),
                "bias": float(errors.mean()),
            })
            print(f"{env_name} total={total}: rollout mse {sq.mean():.4g} "
                  f"(se {sq.std(ddof=1)/np.sqrt(len(sq)):.2g})")
    out = ROOT / "results" / "tabular_rollout"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # Per-replication errors are retained so bootstrap analyses
    # reproduce from the released files.
    with (out / "raw.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
        w.writeheader()
        w.writerows(raw_rows)
    (out / "manifest.json").write_text(json.dumps(
        {"master_seed": MASTER_SEED,
         "accounting": "one environment sample per trajectory step; "
                       "floor(total/H) trajectories; policy sampling free",
         "independent_streams": True}, indent=1) + "\n")
    print(out / "summary.csv")


if __name__ == "__main__":
    main()
