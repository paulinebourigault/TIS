# Declared ablation: mixture controls for the anchored design

Declared 2026-09-23, before any mixture-control simulation ran. This is a
labeled retrospective ablation on the already-examined held-out panels; it
cannot inherit the frozen study's prospective status.

**Question.** Does the anchor's gain come specifically from the learned
tail-influence component, or would equally regularized alternatives match
it? With component weights already carrying the original exploration floor
(w = (1-lambda) p + lambda u) and no second floor after mixing:

| Design | Fresh-query shares before rounding |
|---|---|
| anchored TIS (existing) | 0.5 w_tail + 0.5 w_occ |
| occ_plus_uniform (new) | 0.5 u + 0.5 w_occ |
| occ_plus_mean (new) | 0.5 w_mean + 0.5 w_occ |

occ_plus_uniform equals occupancy with effective floor (1+lambda)/2 and
coincides with anchored TIS exactly whenever the tail pilot falls back to
uniform, so it is the most direct "stronger exploration/shrinkage"
alternative explanation. occ_plus_mean tests whether a smoother influence
score suffices.

**Protocol.** Frozen FinQA kernels, both generators, both workflows;
budgets 100, 200, 400, 800 queries per kernel; 300 replications; the
existing pilot sizes, exploration floor, integer rounding, final
estimator, and charged pilots, all unchanged. Methods run together for
paired conditional-query streams: uniform, learned occupancy, learned
mean, plain TIS, anchored TIS, both mixture controls, oracle+floor
(population reference), and complete rollouts on their independent
stream. Outputs land in results/finqa_heldout_<wf><gen>_mix/ and touch
no frozen result.

**Committed interpretation, recorded in advance.**
- If anchored TIS beats both mixtures: the workflow evidence specifically
  supports learning tail influence.
- If it matches occ_plus_uniform: the anchor's gain is described as
  occupancy/exploration regularization, not tail learning.
- If occ_plus_mean wins: the practical case for a tail-specific workflow
  allocator is weak, and the paper says so.
All prespecified comparisons are reported for every cell, regardless of
direction.
