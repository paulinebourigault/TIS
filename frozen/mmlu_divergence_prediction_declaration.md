# Declared test: does the tail-mean divergence predict the blending outcome out of sample?

Declared 2026-09-24, before computing the divergence statistic for, or running
any blending simulation on, the five settings below.

**Question.** On the nine MMLU-Pro Brier cells and the confident-error cells run
so far, the anchor's advantage over the occupancy-mean blend tracked the median
total-variation distance between normalized tail and mean influence shares
(threshold .24 for wins, at most .18 for no consistent advantage). Several of
those distances were computed after the runs. This test applies the same rule
to settings never used for blending, with predictions recorded before running.

**Settings (all with existing frozen kernels; no new model calls).**
1. High-stakes panel, Qwen3-4B (hs_qwen_kernels), confident-error utility.
2. High-stakes panel, Phi-4-mini (hs_robustness_kernels), confident-error utility.
3. High-stakes panel, Qwen3-32B (hs_scale2_kernels), confident-error utility.
4. Primary panel, Qwen3-4B, cautious review policy (cautious_qwen_kernels), Brier utility.
5. Primary panel, Qwen3-4B, cautious review policy (cautious_qwen_kernels), confident-error utility.

**Statistic.** For each setting, the median over the first 25 panel questions of
0.5 * sum_g |p_g - m_g|, with p and m the normalized tail and mean influence
shares of the population model (closed grid, H = 6, alpha = 0.1, the setting's
utility and review policy). Same code as the values reported for the earlier
cells.

**Simulation protocol.** Closed grid; H = 6; alpha = 0.1; budgets 100, 200, 400
per shared group; 300 replications; master seed 20260725; standard pilot,
floor, rounding, charged pilots. Methods on coupled streams: uniform, learned
occupancy, learned mean, plain TIS, anchored TIS, occ_plus_uniform,
occ_plus_mean, oracle+floor (reference), complete rollouts.

**Outcome.** Paired contrast of anchored TIS against occ_plus_mean at each of
the three budgets; "resolved" means |z| >= 2 (negative favors the anchor).

**Prediction rule (fixed now).**
- Median divergence >= .24: predict WIN, i.e. the anchor is resolved better in
  at least 2 of the 3 budgets and resolved worse in none.
- Median divergence <= .18: predict NO ADVANTAGE, i.e. the anchor is resolved
  better in at most 1 of the 3 budgets.
- Strictly between .18 and .24: no prediction; the outcome is reported.

A setting's prediction is correct if its outcome satisfies the predicted
condition. Every setting is reported whatever the result. The predictions are
appended below after computing the statistic and before any simulation.

## Predictions

Computed 2026-09-24 after the rule above was committed and before any
simulation of these settings. Questions with zero tail or mean influence are
excluded from the median (n shown).

| Setting | Median divergence (n) | Prediction |
|---|---|---|
| 1. High-stakes Qwen3-4B, confident-error | .318 (24) | WIN |
| 2. High-stakes Phi-4-mini, confident-error | .169 (24) | NO ADVANTAGE |
| 3. High-stakes Qwen3-32B, confident-error | .216 (23) | none (between thresholds) |
| 4. Qwen3-4B cautious policy, Brier | .195 (25) | none (between thresholds) |
| 5. Qwen3-4B cautious policy, confident-error | .269 (25) | WIN |
