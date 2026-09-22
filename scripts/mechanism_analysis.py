#!/usr/bin/env python3
"""Mechanism analysis: influence versus occupancy, pilot noise, and what
anchoring changes.

Everything here replays existing frozen artifacts; no model is queried
and no simulation is repeated. For each analyzed setting the script

1. computes the exact influence scales sigma_g and exact occupancy o_g
   per query group from the frozen kernels, and summarizes their
   alignment (correlation of normalized shares, and the influence mass
   carried by the bottom-occupancy half of the groups);
2. regenerates every pilot draw exactly as the runner did (identical
   seed rule, pilot size, exploration floor, integer allocation) and
   evaluates the leading variance functional V(w) = sum_g sigma_g^2/w_g
   at the realized learned-TIS, learned-occupancy, anchored, and uniform
   weights, using the exact sigma_g as reference information;
3. converts V to the leading MSE prediction V(w)/n with each method's
   actual main-stage count (pilot charged exactly as in the runs), and
   compares its per-question mean with the observed simulation MSE;
4. measures, per pilot, the largest single-group contribution share
   max_g (sigma_g^2/w_g)/V and how the anchor reweights the group that
   plain TIS starves most.

The identity V(w) - S^2 = S^2 sum_g (p_g - w_g)^2 / w_g with
p_g = sigma_g / S explains the numbers: a pilot that misplaces weight on
a high-sigma group pays quadratically. Rollouts are excluded: their
trajectory samples do not follow this independent-group functional.

Settings (chosen transparently: the documented MMLU depth failure, one
FinQA cell where the anchor trails occupancy, one where it leads):
  A. MMLU Qwen3-4B closed grid, H=6, alpha=.1, 400 queries per kernel.
  B. FinQA held-out Qwen, ordinary workflow, 200 queries per kernel.
  C. FinQA held-out Phi, ordinary workflow, 200 queries per kernel.
All panel questions of each setting are included; zero-influence
questions are counted and excluded from V (V is undefined there at
first order).

Writes results/mechanism_analysis.json and figures/
fig_mechanism.pdf (+ .png render).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tis.allocation import integer_allocation, pilot_size, regularized_weights  # noqa: E402
from tis.engine import compute_influences  # noqa: E402
from tis.finqa.models import finqa_question_model  # noqa: E402
from tis.llm_study.models import shared_question_model  # noqa: E402
from tis.llm_study.runner import _anchored_score, _occupancy_scores, _rng  # noqa: E402
from tis.model import model_from_counts, sample_counts  # noqa: E402

ALPHA = 0.1
METHODS = ("tis", "learned_occupancy", "tis_anchored")


def observed_mse(path: Path, budget_key: str, budget: int,
                 extra: dict | None = None) -> dict[tuple[int, str], float]:
    out: dict[tuple[int, str], float] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row[budget_key]) != budget:
                continue
            if extra and any(row[k] != v for k, v in extra.items()):
                continue
            out[(int(row["panel_index"]), row["method"])] = float(row["mse"])
    return out


def analyze_setting(name, models, seeds, budget, config, observed, replications):
    """models: {panel_index: FiniteHorizonModel}; seeds(q, rep) -> Generator."""
    exploration_exp = float(config["exploration_exponent"])
    minimum_main = int(config["minimum_main_per_group"])
    rows = {"name": name, "questions": [], "pilot_summary": {}}
    align_shares, align_corr = [], []
    per_method_pred = {m: [] for m in METHODS + ("uniform",)}
    v_ratio_dists = {m: [] for m in METHODS}
    top_contrib = {m: [] for m in METHODS}
    starved_lift = []
    zero_influence = 0

    for qindex, model in sorted(models.items()):
        truth = compute_influences(model, ALPHA)
        sigma = np.asarray(truth.tail_scales, dtype=float)
        S = float(sigma.sum())
        occ = _occupancy_scores(model, truth)
        occ_share = occ / occ.sum()
        groups = model.group_count
        total = groups * budget
        exploration = total ** (-exploration_exp)
        m_pilot = pilot_size(total, groups, float(config["pilot_multiplier"]),
                             int(config["minimum_pilot_per_group"]), minimum_main)
        main_total = total - groups * m_pilot
        if S <= 0:
            zero_influence += 1
            continue
        p = sigma / S
        order = np.argsort(occ_share)
        bottom_half = order[: groups // 2]
        align_shares.append(float(p[bottom_half].sum()))
        align_corr.append(float(np.corrcoef(p, occ_share)[0, 1]))

        v_uniform_pred = (groups * float((sigma ** 2).sum())) / total
        per_method_pred["uniform"].append((qindex, v_uniform_pred))
        v_star = S * S  # unconstrained oracle variance functional

        pred_acc = {m: [] for m in METHODS}
        for rep in range(replications):
            pilot_counts = sample_counts(
                model, {g: m_pilot for g in model.query_groups or ()},
                seeds(qindex, rep))
            pilot = compute_influences(model_from_counts(model, pilot_counts), ALPHA)
            occ_pilot = _occupancy_scores(model, pilot)
            scores = {
                "tis": np.asarray(pilot.tail_scales, dtype=float),
                "learned_occupancy": occ_pilot,
                "tis_anchored": _anchored_score(
                    np.asarray(pilot.tail_scales, dtype=float), occ_pilot),
            }
            w_by = {}
            for method in METHODS:
                weights = regularized_weights(scores[method], exploration)
                counts = integer_allocation(main_total, weights, minimum_main)
                w = counts / counts.sum()
                w_by[method] = w
                contrib = sigma ** 2 / w
                v = float(contrib.sum())
                pred_acc[method].append(v / main_total)
                v_ratio_dists[method].append(v / v_star)
                top_contrib[method].append(float(contrib.max() / v))
            g_star = int(np.argmax(sigma ** 2 / w_by["tis"]))
            starved_lift.append(
                float(w_by["tis_anchored"][g_star] / w_by["tis"][g_star]))
        for method in METHODS:
            per_method_pred[method].append((qindex, float(np.mean(pred_acc[method]))))

    rows["alignment"] = {
        "median_influence_share_of_bottom_occupancy_half": float(np.median(align_shares)),
        "share_p90": float(np.percentile(align_shares, 90)),
        "median_share_correlation": float(np.median(align_corr)),
    }
    rows["zero_influence_questions"] = zero_influence
    for method, dist in v_ratio_dists.items():
        arr = np.asarray(dist)
        grid = np.linspace(0.0, 100.0, 513)
        rows["pilot_summary"][method] = {
            "V_over_oracle_median": float(np.median(arr)),
            "V_over_oracle_p90": float(np.percentile(arr, 90)),
            "V_over_oracle_max": float(arr.max()),
            "V_over_oracle_quantiles": [float(x) for x in np.percentile(arr, grid)],
            "top_group_contribution_median": float(np.median(top_contrib[method])),
            "top_group_contribution_p90": float(np.percentile(top_contrib[method], 90)),
        }
    rows["anchor_lift_of_tis_starved_group"] = {
        "median": float(np.median(starved_lift)),
        "p90": float(np.percentile(starved_lift, 90)),
    }
    calib = []
    for method in METHODS + ("uniform",):
        for qindex, pred in per_method_pred[method]:
            obs = observed.get((qindex, method))
            if obs is not None and obs > 0 and pred > 0:
                calib.append({"panel_index": qindex, "method": method,
                              "predicted_leading_mse": pred, "observed_mse": obs})
    ratios = [c["observed_mse"] / c["predicted_leading_mse"] for c in calib]
    rows["prediction_calibration"] = {
        "pairs": len(calib),
        "log10_ratio_median": float(np.median(np.log10(ratios))),
        "log10_ratio_p10": float(np.percentile(np.log10(ratios), 10)),
        "log10_ratio_p90": float(np.percentile(np.log10(ratios), 90)),
    }
    rows["_calibration_points"] = calib
    rows["_scatter"] = {"p": [], "occ": []}
    for qindex, model in sorted(models.items()):
        truth = compute_influences(model, ALPHA)
        sigma = np.asarray(truth.tail_scales, dtype=float)
        if sigma.sum() <= 0:
            continue
        occ = _occupancy_scores(model, truth)
        rows["_scatter"]["p"].extend((sigma / sigma.sum()).tolist())
        rows["_scatter"]["occ"].extend((occ / occ.sum()).tolist())
    return rows


def main() -> None:
    if "--figure-only" in sys.argv:
        out = json.loads((ROOT / "results/mechanism_analysis.json").read_text())
        make_figure(out, out["scatter"], out["calibration_points"])
        return
    results = []

    # --- Setting A: MMLU Qwen closed grid, H=6, b400 ---
    config = json.loads((ROOT / "configs/llm_study_closed.json").read_text())
    artifact = json.loads((ROOT / "frozen/qwen_kernels.json").read_text())
    horizon, budget = 6, 400
    models = {int(q["panel_index"]): shared_question_model(q, horizon, closed_grid=True)
              for q in artifact["questions"]}
    groups0 = next(iter(models.values())).group_count
    obs = observed_mse(ROOT / "results/llm_study_closed/question_summary.csv",
                       "budget_per_shared_group", budget,
                       {"horizon": str(horizon), "alpha": "0.1"})
    seeds = lambda q, r: _rng(int(config["master_seed"]), horizon, 0,
                              groups0 * budget, q, r, 0)
    results.append(analyze_setting(
        "MMLU Qwen3-4B H=6 b400", models, seeds, budget, config, obs,
        int(config["replications"])))

    # --- Settings B, C: FinQA held-out, ordinary workflow, b200 ---
    for gen, kfile, rdir in (
        ("Qwen", "finqa_kernels_heldout2_ordinary.json", "finqa_heldout_ordinary"),
        ("Phi", "finqa_kernels_heldout2_phi_ordinary.json", "finqa_heldout_ordinary_phi"),
    ):
        fconfig = json.loads((ROOT / "configs/finqa_heldout_ordinary.json").read_text())
        art = json.loads((ROOT / "frozen" / kfile).read_text())
        banks = {int(b["panel_index"]): b["candidates"]
                 for b in json.loads((ROOT / "frozen/finqa_banks_heldout2.json").read_text())["banks"]}
        budget = 200
        models = {int(q["panel_index"]): finqa_question_model(q, banks[int(q["panel_index"])], 3)
                  for q in art["questions"]}
        obs = observed_mse(ROOT / f"results/{rdir}/question_summary.csv",
                           "budget_per_group", budget)
        groups0 = next(iter(models.values())).group_count
        seeds = lambda q, r, _c=fconfig, _g=groups0: _rng(
            int(_c["master_seed"]), 3, 0, _g * budget, q, r, 0)
        results.append(analyze_setting(
            f"FinQA {gen} ordinary b200", models, seeds, budget, fconfig, obs,
            int(fconfig["replications"])))

    out = {"settings": []}
    scatter, calib_all = [], []
    for r in results:
        scatter.append({"name": r["name"], **r.pop("_scatter")})
        calib_all.append({"name": r["name"], "points": r.pop("_calibration_points")})
        out["settings"].append(r)
    out["scatter"] = scatter
    out["calibration_points"] = calib_all
    (ROOT / "results/mechanism_analysis.json").write_text(
        json.dumps(out, indent=1, sort_keys=True) + "\n")

    make_figure(out, scatter, calib_all)
    print(json.dumps({r["name"]: {"alignment": r["alignment"],
                                  "pilot": r["pilot_summary"],
                                  "lift": r["anchor_lift_of_tis_starved_group"],
                                  "calibration": r["prediction_calibration"]}
                      for r in out["settings"]}, indent=1))


def make_figure(out, scatter, calib_all) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullFormatter

    TICKS, LABEL, LEGEND, TITLE = 18, 20, 18, 22
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    colors = {"MMLU Qwen3-4B H=6 b400": "#1f4e9c",
              "FinQA Qwen ordinary b200": "#2ca02c",
              "FinQA Phi ordinary b200": "#ff7f0e"}
    short = {"MMLU Qwen3-4B H=6 b400": "MMLU Qwen $H{=}6$",
             "FinQA Qwen ordinary b200": "FinQA Qwen",
             "FinQA Phi ordinary b200": "FinQA Phi"}

    ax = axes[1]
    corr = {r["name"]: r["alignment"]["median_share_correlation"]
            for r in out["settings"]}
    for s in scatter:
        p = np.asarray(s["p"]); occ = np.asarray(s["occ"])
        keep = (p > 1e-8) & (occ > 1e-8)
        ax.scatter(occ[keep], p[keep], s=13, alpha=0.3, color=colors[s["name"]],
                   edgecolors="none")
    lims = (1e-6, 1.0)
    ax.plot(lims, lims, ls=":", color="black", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lims); ax.set_ylim(*lims)
    ax.set_xlabel("exact occupancy share\n(visit frequency)", fontsize=LABEL - 2)
    ax.set_ylabel("exact influence share\n(importance for the tail)", fontsize=LABEL - 2)
    ax.tick_params(labelsize=TICKS)
    ax.set_title("(b)", fontsize=TITLE, fontweight="bold", loc="left")

    ax = axes[0]
    mmlu = out["settings"][0]
    for method, label, color in (("tis", "plain TIS", "#000000"),
                                 ("learned_occupancy", "learned occupancy", "#2ca02c"),
                                 ("tis_anchored", "anchored TIS", "#1f4e9c")):
        s = mmlu["pilot_summary"][method]
        q = np.asarray(s.get("V_over_oracle_quantiles", []))
        if q.size:
            frac = np.linspace(0.0, 1.0, q.size)
            ax.plot(q, frac, color=color, lw=2.6 if method == "tis_anchored" else 2.0,
                    ls="--" if method == "tis" else "-", label=label)
    ax.axvline(1.0, color="black", ls=":", lw=1.2)
    ax.text(1.07, 0.55, "oracle", color="black", fontsize=LEGEND - 2, ha="left",
            rotation=90, va="bottom")
    ax.annotate("unlucky pilots:\none starved group",
                xy=(150, 0.80), xytext=(7, 0.38), fontsize=LEGEND - 2,
                arrowprops=dict(arrowstyle="->", color="black", lw=1.4),
                color="black")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 2e3)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"realized variance $V(\widehat w)\,/\,S^2$", fontsize=LABEL)
    ax.set_ylabel("fraction of pilots below", fontsize=LABEL)
    ax.tick_params(labelsize=TICKS)
    ax.legend(fontsize=LEGEND - 2, frameon=False, loc="lower right")
    ax.set_title("(a)", fontsize=TITLE, fontweight="bold", loc="left")

    ax = axes[2]
    for c in calib_all:
        pts = c["points"]
        xs = [p["predicted_leading_mse"] for p in pts]
        ys = [p["observed_mse"] for p in pts]
        ax.scatter(xs, ys, s=26, alpha=0.6, color=colors[c["name"]],
                   edgecolors="none")
    lo, hi = 1e-12, 1e-1
    ax.plot([lo, hi], [lo, hi], ls=":", color="black", lw=1.2)
    ax.text(3e-5, 4e-6, "$y=x$", color="black", fontsize=LEGEND - 1,
            ha="left", va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    below = sum(1 for c in calib_all for p in c["points"]
                if p["observed_mse"] < lo or p["predicted_leading_mse"] < lo)
    ax.text(0.03, 0.96, f"{below} degenerate cells\nbelow range",
            transform=ax.transAxes, fontsize=LEGEND - 2, va="top")
    ax.set_xlabel(r"predicted $V(\widehat w)/n$", fontsize=LABEL)
    ax.set_ylabel("observed MSE", fontsize=LABEL)
    ax.tick_params(labelsize=TICKS)
    ax.set_title("(c)", fontsize=TITLE, fontweight="bold", loc="left")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="", marker="o", ms=9, color=colors[k])
               for k in colors]
    fig.legend(handles, [short[k] for k in colors], loc="lower center",
               ncol=3, fontsize=LEGEND, frameon=False,
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.8,
               handletextpad=0.3)
    fig.tight_layout(w_pad=2.4, rect=(0, 0.09, 1, 1))
    outfig = ROOT / "figures/fig_mechanism.pdf"
    fig.savefig(outfig, bbox_inches="tight")
    fig.savefig(outfig.with_suffix(".png"), dpi=110, bbox_inches="tight")
    print(outfig)


if __name__ == "__main__":
    main()
