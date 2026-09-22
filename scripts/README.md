# Script index

Every script is self-contained, reads only committed configs/results/frozen
artifacts, and states its protocol in its module docstring. Grouped by role.

## Analysis (CPU, reproduce paper numbers from committed inputs)

| Script | Purpose | Writes |
|---|---|---|
| `build_tabular_reconciled.py` | Deterministic builder for the merged tabular table; `--check` verifies the committed file cell-by-cell | `results/tabular_reconciled.json` |
| `controlled_separation.py` | Controlled-separation experiment: analytic verification gate, then the frozen simulation | `results/controlled_separation/` |
| `tabular_rollout.py` | Matched-cost complete-rollout rows for the tabular tasks | `results/tabular_rollout/` |
| `mechanism_analysis.py` | Pilot replay: realized-variance ECDFs, influence/visitation alignment, prediction calibration; `--figure-only` replots from the JSON | `results/mechanism_analysis.json`, mechanism figure |
| `llm_paper_analysis.py` | Canonical MMLU study numbers (closed grid) | `results/llm_paper_numbers_closed.json` |
| `llm_extension_analysis.py` | Learned-baseline and ablation extension numbers | `results/llm_extension_numbers.json` |
| `strongest_baseline_table.py` | Strongest-baseline comparison table | table fragments |
| `llm_final_analysis.py`, `llm_cost_report.py`, `workflow_choice_analysis.py` | MMLU decision/cost side analyses | JSONs under `results/` |
| `finqa_analysis.py` | FinQA panel/paired/decision report; refuses to overwrite the authoritative report when raw decision data are absent | `results/finqa_paper_numbers.json` |
| `finqa_supplement_tables.py` | FinQA supplement tables from committed panels | table fragments |
| `finqa_metric_audit.py` | Percent-convention audit and unit-aware severity variant | `results/finqa_metric_audit.json` |
| `finqa_skeptical_analysis.py` | Skeptical-workflow follow-up analysis | JSONs under `results/` |
| `finqa_strong_baselines.py` | Per-workflow paired MSE ratios: plain/anchored TIS vs occupancy/rollouts, auto-including the declared 400/800 extension | `results/finqa_strong_baselines.json` |
| `finqa_paircheck.py` | Workflow-decision pairing sensitivity (identical-index vs 500 independent permutations) | `results/finqa_decision_paircheck/` |
| `finqa_token_cost.py` | Prompt-token replay of allocations (accounting, not live cost) | token report |
| `cost_to_accuracy.py` | Single-benchmark RMSE crossing brackets | `results/cost_to_accuracy.json` |
| `cost_multi_target.py` | Censor-preserving multi-target relative query cost with empirical bootstraps | `results/cost_multi_target.json` |
| `method_timing.py` | Per-method pipeline replay timing, single pinned CPU thread | `results/method_timing.json` |
| `robustness_panels.py` | Perturbed-instance ratios and pilot-sensitivity grid | `results/robustness_panels/` |
| `floor_diagnostic.py`, `floor_holdout_validation.py` | Exploration-floor sensitivity and retrospective split analysis | JSONs under `results/` |
| `grid_representation_check.py` | Categorical-grid exactness checks | report |
| `make_source_manifest.py` | Source-hash manifest for a run directory | `manifest.json` entries |

## Figures

| Script | Figure |
|---|---|
| `make_figures.py` | Tabular MSE curves, controlled-MRP appendix figure, robustness supplement |
| `make_finqa_figure.py` | FinQA held-out figure (estimation panels and workflow choice) |
| `mechanism_analysis.py --figure-only` | Mechanism figure |

## Run launchers (host-specific)

`reproduce_*.sh` wrap the `tis` CLI for the tabular suites.
`run_finqa_*.sh`, `run_finqa_unit_aware.py`, `run_llm_*.sh`, and
`run_anchored.sh` launch the simulation campaigns from frozen kernels
(CPU only). None of these are needed to reproduce a reported number from
the committed artifacts.
