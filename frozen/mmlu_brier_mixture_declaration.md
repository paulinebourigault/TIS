# Declared ablation: blending controls on the MMLU-Pro studies (Brier utility)

Declared 2026-09-24, before any run of these configurations. Retrospective
with respect to the already-examined MMLU panels; labeled as such.

**Motivation (population-only, recorded before any method comparison).**
When a question's failure probability is below alpha, the alpha-quantile sits
at the top of the return support, CVaR is an affine function of the mean,
and tail influence equals mean influence up to scale. On the FinQA review
panels this holds for most Qwen questions (median total-variation distance
between normalized tail and mean influence shares = 0.000), matching the
FinQA ablation in which the mean-influence blend matched the anchor. On the
MMLU-Pro panels the same statistic is large (median 0.09-0.33 across
generators and panels, 25-question samples). The theory therefore predicts
that a tail-specific advantage over the mean-influence blend is possible on
MMLU-Pro but not on FinQA.

**Protocol.** Primary MMLU-Pro panel, six generators (frozen kernels:
qwen, robustness, robustness2, scale1, scale2, scale3), and the high-stakes
panel, three generators (hs_qwen, hs_robustness, hs_scale2); closed grid;
Brier utility; default policy; H = 6; alpha = 0.1; budgets 100, 200, 400
per shared group; 300 replications; standard pilot, floor, rounding,
charged pilots. Methods on coupled streams: uniform, learned occupancy,
learned mean, plain TIS, anchored TIS, occ_plus_uniform, occ_plus_mean,
oracle+floor (reference), complete rollouts. No new model calls.

**Prediction and interpretation, recorded in advance.**
- If anchored TIS beats occ_plus_mean on these panels, the tail score adds
  value in LLM workflows exactly where tail and mean influence diverge,
  consistent with the FinQA contrast.
- If occ_plus_mean matches or wins here too, the prediction fails and the
  paper says that tail specificity is not supported for LLM workflows.
All nine generator-panel cells are reported regardless of direction.
