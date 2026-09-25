# Declared analysis: a pilot-estimated divergence as a practical selection rule

Declared 2026-09-25, before computing any pilot-estimated divergence.
No new model calls and no new allocation simulations: the rule is evaluated
against blending runs already on disk.

**Motivation.** The tail-mean divergence used so far is computed from
population kernels, which a practitioner does not have. The allocation
methods already draw a uniform pilot; the same pilot yields estimated tail
and mean influence scales at no extra query cost.

**Statistic.** For a setting (configuration) and budget, for each question,
take the pilot drawn for replication 0 by the runner (same seed coordinates,
same pilot size), compute tail and mean influence scales in the pilot-fitted
model, and the total-variation distance between their normalized shares.
Questions whose pilot tail or mean scales sum to zero are excluded. The
setting-level statistic is the median over questions.

**Rule (fixed now).** Use anchored TIS if the pilot median is at least .21
(the midpoint of the population thresholds .18 and .24); otherwise use the
occupancy-mean blend.

**Evaluation.** For every setting and budget with both anchored TIS and
occ_plus_mean results on disk: the rule is scored as a *bad choice* when it
selects the design with resolved higher MSE (paired |z| >= 2) than the other,
and as *correct* otherwise. Reported: number of bad choices; MSE of the chosen
design relative to the better of the two; the rank correlation between pilot
and population medians; and, for stability, how often replications 1-4 give
the same decision as replication 0.

**Settings.** MMLU-Pro: Brier blending (6 generators + 3 high-stakes),
confident-error blending (6), out-of-sample prediction test (5), long-horizon
(6, when complete). FinQA: held-out blending H = 3 (4), tool-fault (12),
long-horizon H = 6 (4, when complete). All are reported whatever the result.
The pilot-based threshold is not tuned after seeing results; if the pilot
statistic is biased relative to the population one, that is reported.
