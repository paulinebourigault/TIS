# FinQA development-gate report (protocol v2, screened panels)

Recorded 2026-09-12/13, before any held-out artifact existed. Inputs:
`frozen/finqa_{panel,banks,scan,kernels,audit}_dev2*` (commit be68788,
Phi variants 5350b98) and the four development simulation runs in
`results/finqa_dev_{ordinary,unitcheck}{,_phi}/` (100 replications,
budgets 25/50/100/200, six methods, alpha = .1, H = 3).

## Gate criteria and findings

**1. Candidate validity — pass.** 142/160 admitted-bank candidates parse
to a number (invalid candidates keep severity utility 0 by design).

**2. Tail informativeness — pass (enforced by the screen).** The v1 gate
found 16/20 constant-utility banks; the v2 screen (spread >= .25 in the
seeded scan order; 170 scanned, 20 admitted, 11.8% acceptance) yields
spreads .28-.90 with 11/20 banks containing a near-zero-utility
candidate. Every scanned bank, admitted or not, is in
`frozen/finqa_scan_dev2.jsonl`.

**3. Positive margins — pass.** Qwen 20/20 (min .0999); Phi 20/20
(min .0008, one near-zero margin noted).

**4. Headroom — pass.** Qwen: population uniform-to-oracle median ~70x;
realized at budgets 100-200, oracle .006-.013x uniform panel MSE,
learned occupancy .016-.051, anchored TIS .041-.093, complete rollouts
.016-.040, plain TIS 1.2-1.4 (the prespecified-floor fragility again).
Phi: population headroom 10-13x; anchored TIS is the best learned
method at every budget (.16-.42x uniform) and beats uniform with paired
z = +7.9 to +9.0 already at development scale; plain TIS .9-2.2x.

**5. Workflow separation — one equivalence, one difference.**
Qwen: the two review templates leave the reachable dynamics unchanged -
per-question terminal-law CVaRs agree to 5e-8 on all 20 questions.
Kernel differences exist (32% of non-root states at TV > .1) but sit
off the reachable paths or move confidence rather than the candidate;
a confidence-weighted severity was checked and would separate only
1/20 questions, so no estimand revision is warranted. The workflow
decision for Qwen resolves to an equivalence: identical tail severity,
choose the cheaper template. Phi: the workflows genuinely separate -
truths differ on 13/20 questions (max .156; panel values 1.7274
ordinary vs 1.7406 unitcheck). The held-out decision analysis therefore
contains both outcomes an evaluator faces: a resolved equivalence and a
resolved difference.

**6. Known degeneracy (budgets 25-50, Qwen only) — recorded.** Qwen's
kernels are near-deterministic (57-59% of state kernels put > .99 mass
on one outcome). At 25-50 queries per kernel the empirical law is
almost always the exact point mass, uniform's estimates are
deterministic and essentially exact, and every concentration method
pays its exploration floors (2-query kernels occasionally miss the
dominant outcome). Uniform is unbeatable below this concentration
threshold; the effect inverts by budget 100. Reported as the
budget-calibration boundary; budgets kept for honesty. Phi's flatter
kernels show no such regime.

## Held-out sizing

Development paired per-replication panel-MSE differences (Qwen,
20 questions x 100 replications): anchored-vs-uniform z ~ 2.0,
anchored-vs-occupancy z ~ 2.9-3.2, anchored-vs-rollout z ~ 2.4-2.6;
uniform-vs-TIS unresolved (z -0.6 to -1.7). Scaling z by
sqrt(questions x replications): 50 questions x 300 replications
projects the headline to z ~ 5.5 and the cross-method comparisons to
z ~ 8. A 100-question panel is not needed.

## Frozen decision (frozen/finqa_freeze2.json, commit 9807a78)

Held-out panel: 50 questions, identical screen, scan order seeded
20260913, excluding all 170 development-scanned items. Generators:
Qwen3-4B and Phi-4-mini, both workflows each. Simulations: 300
replications, master seed 20260919, budgets unchanged. The held-out
GPU chain (job 26692) reads this record and started only after it
appeared.
