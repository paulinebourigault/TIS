#!/usr/bin/env python3
"""Runtime instrumentation: per-method pipeline replay timing.

Times each method's own pipeline separately, end to end per replication:

- uniform: allocation + main-sample categorical estimation (no pilot);
- learned occupancy (minimal): pilot sampling + empirical model +
  forward occupancy solve only (no CVaR adjoint or influence bundle) +
  allocation + estimation;
- plain TIS: pilot sampling + full influence solve + allocation +
  estimation;
- anchored TIS: pilot sampling + full influence solve + occupancy and
  anchor scores + allocation + estimation;
- complete rollout: trajectory simulation + empirical CVaR at matched
  total transition budget.

BLAS/OMP threads are pinned to 1 before NumPy import. Frozen-law replay
timing measures computation only; it does not establish live model-call
costs, and query counts are never merged with these times. The FinQA
setting aggregates over the full 50-question held-out Phi ordinary panel
(per-question grids range from 13 to the panel maximum atom count, which
is recorded), so it is a panel total per replication, not question 0
alone. Writes results/method_timing.json.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np  # noqa: E402  (after thread pinning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tis.allocation import integer_allocation, pilot_size, regularized_weights  # noqa: E402
from tis.engine import compute_influences, solve_occupancy, solve_stop_loss  # noqa: E402
from tis.environments import make_environment  # noqa: E402
from tis.finqa.models import finqa_question_model  # noqa: E402
from tis.llm_study.runner import _empirical_cvar, _mapping, _rng  # noqa: E402
from tis.runner import _anchored_score, _occupancy_scores  # noqa: E402
from tis.model import counts_from_streams, coupled_streams, model_from_counts, sample_counts  # noqa: E402
from tabular_rollout import rollout_returns  # noqa: E402

WARMUP, REPS = 2, 10
ALPHA = 0.1


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def build_settings():
    settings = []
    for env, budget in (("cliffwalking_primary", 400), ("inventory_untied_v1", 1200)):
        settings.append((f"{env}_b{budget}", [make_environment(env)], budget))
    art = json.loads((ROOT / "frozen/finqa_kernels_heldout2_phi_ordinary.json").read_text())
    banks = {int(b["panel_index"]): b["candidates"]
             for b in json.loads((ROOT / "frozen/finqa_banks_heldout2.json").read_text())["banks"]}
    models = [finqa_question_model(q, banks[int(q["panel_index"])], 3)
              for q in art["questions"]]
    settings.append((f"finqa_phi_ordinary_panel{len(models)}_b200", models, 200))
    return settings


def estimate(model, counts, rng):
    streams = coupled_streams(model, _mapping(model, counts), rng)
    empirical = counts_from_streams(model, streams, _mapping(model, counts))
    solve_stop_loss(model_from_counts(model, empirical), ALPHA)


def run_pipeline(model, budget, method, rep):
    """One full replication of one method's pipeline; returns elapsed seconds."""
    groups = model.group_count
    total = groups * budget
    exploration = total ** (-0.25)
    m_pilot = pilot_size(total, groups, 4.0, 8, 2)
    main_total = total - groups * m_pilot
    stage = {"uniform": 0, "learned_occupancy": 1, "tis": 2,
             "tis_anchored": 3, "complete_rollout": 4}[method]
    t0 = time.perf_counter()
    if method == "uniform":
        counts = integer_allocation(total, np.full(groups, 1.0 / groups), 2)
        estimate(model, counts,
                 _rng(20260923, model.horizon, 0, total, stage, rep, 1))
    elif method == "complete_rollout":
        returns = rollout_returns(model, total // model.horizon,
                                  _rng(20260923, model.horizon, 0, total, stage, rep, 6))
        _empirical_cvar(returns, ALPHA)
    else:
        rng = _rng(20260923, model.horizon, 0, total, stage, rep, 0)
        pilot_counts = sample_counts(model, {g: m_pilot for g in model.query_groups}, rng)
        empirical_model = model_from_counts(model, pilot_counts)
        if method == "learned_occupancy":
            # Minimal path: forward visitation only, no CVaR adjoint.
            score = _occupancy_scores(model, solve_occupancy(empirical_model))
        else:
            pilot = compute_influences(empirical_model, ALPHA)
            if method == "tis":
                score = np.asarray(pilot.tail_scales)
            else:
                occ = _occupancy_scores(model, pilot.occupancy)
                score = _anchored_score(np.asarray(pilot.tail_scales), occ)
        weights = regularized_weights(score, exploration)
        counts = integer_allocation(main_total, weights, 2)
        estimate(model, counts,
                 _rng(20260923, model.horizon, 0, total, stage, rep, 1))
    return time.perf_counter() - t0


METHODS = ("uniform", "learned_occupancy", "tis", "tis_anchored", "complete_rollout")


def time_setting(name, models, budget):
    times = {m: [] for m in METHODS}
    for rep in range(WARMUP + REPS):
        for method in METHODS:
            elapsed = sum(run_pipeline(model, budget, method, rep)
                          for model in models)
            if rep >= WARMUP:
                times[method].append(elapsed)
    summary = {m: {"mean_s": float(np.mean(v)), "sd_s": float(np.std(v, ddof=1))}
               for m, v in times.items()}
    summary["scope"] = ("panel total over all questions per replication"
                        if len(models) > 1 else "single environment per replication")
    if len(models) > 1:
        atoms = [len(m.grid) if hasattr(m, "grid") else None for m in models]
        atoms = [a for a in atoms if a]
        if atoms:
            summary["grid_atoms_min_max"] = [min(atoms), max(atoms)]
        summary["questions"] = len(models)
    return summary


def main() -> None:
    out = {"protocol": {
        "version": 2, "warmup": WARMUP, "reps": REPS, "alpha": ALPHA,
        "pipelines": "each method timed separately end to end per "
                     "replication, including its own pilot sampling where "
                     "the method uses one; learned occupancy uses the "
                     "minimal forward-visitation solve without the CVaR "
                     "adjoint or influence bundle",
        "hardware": {"cpu": _cpu_model(), "platform": platform.platform(),
                     "python": platform.python_version(),
                     "numpy": np.__version__,
                     "blas_omp_threads": {v: os.environ.get(v) for v in
                                          ("OPENBLAS_NUM_THREADS",
                                           "OMP_NUM_THREADS",
                                           "MKL_NUM_THREADS")}},
        "note": "frozen-law replay computation only; query counts reported "
                "separately and never merged; not live model-call costs"}}
    for name, models, budget in build_settings():
        out[name] = time_setting(name, models, budget)
        row = {m: out[name][m]["mean_s"] for m in METHODS}
        print(name + ": " + ", ".join(f"{m} {v*1e3:.1f} ms" for m, v in row.items()))
    (ROOT / "results/method_timing.json").write_text(json.dumps(out, indent=1) + "\n")
    print("written results/method_timing.json")


if __name__ == "__main__":
    main()
