# Declared post-hoc budget extension: FinQA held-out, budgets 400 and 800

Declared 2026-09-21, before any extended-budget simulation ran.

**Motivation (stated before results exist).** Across the frozen budgets
25-200, the Phi ordinary paired ratios move monotonically in the anchor's
favor (anchor/occupancy 1.10 -> .88, anchor/rollout 1.96 -> .84), and the
censor-preserving multi-target cost analysis finds the strictest attainable
accuracy targets favor the anchor. This extension tests whether that trend
continues at budgets 400 and 800 queries per kernel. It is outcome-motivated
in the sense that the trend suggested it; therefore it is a labeled post-hoc
sensitivity extension, not part of the frozen prospective protocol, and it
cannot inherit the held-out study's prospective status.

**Commitments.**
1. All four generator-workflow cells run (Qwen/Phi x ordinary/unit-check),
   not only the favorable one, and all results are reported regardless of
   direction, including the expectation-contradicting outcomes.
2. Everything else is unchanged from the frozen configs: kernels, banks,
   panel, seeds rule (master seed 20260919; the seed tuple includes the
   budget, so streams at new budgets are fresh and deterministic), methods,
   replications (300), pilot rule, floor, rounding, metric.
3. Expected outcomes, recorded now: on Phi, the anchor's advantage over
   occupancy and rollouts widens or holds; on Qwen, occupancy and rollouts
   remain ahead of the anchor. If instead the Phi trend reverses, that is
   reported with equal prominence.
4. Outputs land in results/finqa_heldout_<wf><gen>_b48/ and never modify
   the frozen-budget results.

Configs: configs/finqa_heldout_<wf><gen>_b48.json (budgets_per_group
[400, 800], output-relevant fields otherwise identical).
