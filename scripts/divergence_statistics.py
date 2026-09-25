#!/usr/bin/env python3
"""Population tail-mean divergence for every language-model setting.

For each question, the total-variation distance between the normalized
population tail-influence scales and mean-influence scales,
0.5 * sum_g |p_g - m_g|, computed from the frozen kernels. Questions whose
tail or mean scales sum to zero are excluded (their count is reported).
The setting statistic is the median over questions: the first 25 panel
questions for MMLU-Pro, all 50 held-out questions for FinQA. These are the
distances quoted in the paper and recorded in the declarations under
frozen/ before the corresponding runs.

Usage: python scripts/divergence_statistics.py [processes]
Writes results/divergence_statistics.json.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tis.engine import compute_influences  # noqa: E402
from tis.finqa.models import finqa_question_model  # noqa: E402
from tis.finqa.toolfault import toolfault_question_model  # noqa: E402
from tis.llm_study.models import shared_question_model  # noqa: E402

ALPHA = 0.1
MMLU_QUESTIONS = 25
GENERATORS = {"qwen": "Qwen3-4B", "robustness": "Phi-4-mini", "robustness2": "Granite-4.2-8B",
              "scale1": "Mistral-24B", "scale2": "Qwen3-32B", "scale3": "GLM-4-32B"}


def settings() -> list[dict]:
    out = []
    for kern in GENERATORS:
        out.append({"name": f"mmlu_brier_{kern}_h6", "kernels": f"{kern}_kernels", "horizon": 6,
                    "utility": "brier", "policy": "default"})
        out.append({"name": f"mmlu_confident_{kern}_h6", "kernels": f"{kern}_kernels", "horizon": 6,
                    "utility": "confident_error", "policy": "default"})
    for kern in ("hs_qwen", "hs_robustness", "hs_scale2"):
        for utility in ("brier", "confident_error"):
            out.append({"name": f"mmlu_{'confident' if utility != 'brier' else 'brier'}_{kern}_h6",
                        "kernels": f"{kern}_kernels", "horizon": 6, "utility": utility, "policy": "default"})
    for utility in ("brier", "confident_error"):
        out.append({"name": f"mmlu_{'confident' if utility != 'brier' else 'brier'}_cautious_qwen_h6",
                    "kernels": "cautious_qwen_kernels", "horizon": 6, "utility": utility, "policy": "cautious"})
    for kern in ("qwen", "robustness", "scale3"):
        for horizon in (8, 10):
            out.append({"name": f"mmlu_confident_{kern}_h{horizon}", "kernels": f"{kern}_kernels",
                        "horizon": horizon, "utility": "confident_error", "policy": "default"})
    for tag, gen in (("", "qwen"), ("phi_", "phi")):
        for workflow in ("ordinary", "unitcheck"):
            for horizon in (3, 6):
                out.append({"name": f"finqa_{gen}_{workflow}_h{horizon}",
                            "kernels": f"finqa_kernels_heldout2_{tag}{workflow}", "horizon": horizon,
                            "finqa": True})
    for workflow in ("", "unitcheck_"):
        for gen in ("qwen", "phi"):
            for p in (0.01, 0.1):
                out.append({"name": f"finqa_toolfault_{workflow}{gen}_p{p}",
                            "kernels": f"finqa_kernels_heldout2_toolfault_{workflow}{gen}", "horizon": 3,
                            "finqa": True, "fault_probability": p})
    return out


def _tv(tail: np.ndarray, mean: np.ndarray) -> float | None:
    if tail.sum() <= 0 or mean.sum() <= 0:
        return None
    return float(0.5 * np.abs(tail / tail.sum() - mean / mean.sum()).sum())


def evaluate(setting: dict) -> tuple[str, dict]:
    art = json.loads((ROOT / "frozen" / f"{setting['kernels']}.json").read_text())
    values = []
    if setting.get("finqa"):
        banks = {int(b["panel_index"]): b["candidates"] for b in
                 json.loads((ROOT / "frozen/finqa_banks_heldout2.json").read_text())["banks"]}
        for q in art["questions"]:
            bank = banks[int(q["panel_index"])]
            if "fault_probability" in setting:
                model = toolfault_question_model(q, bank, setting["horizon"], setting["fault_probability"])
            else:
                model = finqa_question_model(q, bank, setting["horizon"])
            truth = compute_influences(model, ALPHA)
            values.append(_tv(np.asarray(truth.tail_scales), np.asarray(truth.mean_scales)))
    else:
        for q in art["questions"][:MMLU_QUESTIONS]:
            model = shared_question_model(q, setting["horizon"], closed_grid=True,
                                          utility=setting["utility"], policy=setting["policy"])
            truth = compute_influences(model, ALPHA)
            values.append(_tv(np.asarray(truth.tail_scales), np.asarray(truth.mean_scales)))
    kept = np.array([v for v in values if v is not None])
    return setting["name"], {"median": float(np.median(kept)), "questions_used": int(kept.size),
                             "questions_excluded": len(values) - int(kept.size),
                             "fraction_zero": float(np.mean(kept < 1e-9)),
                             "settings": {k: v for k, v in setting.items() if k != "name"}}


def main() -> None:
    processes = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    with ProcessPoolExecutor(max_workers=processes) as pool:
        results = dict(pool.map(evaluate, settings()))
    for name, entry in results.items():
        print(f"{name:38s} median {entry['median']:.3f}  zero {entry['fraction_zero']:.2f}  "
              f"n {entry['questions_used']}")
    (ROOT / "results/divergence_statistics.json").write_text(json.dumps(results, indent=1) + "\n")
    print("written results/divergence_statistics.json")


if __name__ == "__main__":
    main()
