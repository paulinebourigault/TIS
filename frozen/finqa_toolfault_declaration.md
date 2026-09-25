# Declared experiment: FinQA review with a faulty calculator tool

Declared 2026-09-24, before any calibration or simulation of this design.
New prompts and new real model calibrations on the already-examined
held-out FinQA panel; the design is fixed here in full before any data.

**Question.** Where a rarely visited state drives the worst outcomes
without moving the average, does a tail-specific allocator help in a real
LLM workflow? Here the rare state is a review step whose calculator tool
reported a wrong value.

**Design.** The ordinary FinQA review workflow (horizon 3: select, then two
reviews), with one addition: every review prompt states the result of an
automated calculator that re-ran the prior candidate's computation. The
calculator reports the candidate's own final answer, except that with
fault probability p it reports that value multiplied by 100 (a decimal or
percent slip). The model sees the report and answers with the usual two
constrained tokens. States: root; 72 reviews with a correct report; 72
reviews with a faulted report (145 prompt kernels per question). Each
model call is followed by an independent fault draw, so a kernel outcome is
(model output, fault flag). Rewards and the severity utility are the base
workflow's, unchanged. Code: src/tis/finqa/toolfault.py.

**Calibration.** Both generators (Qwen3-4B, Phi-4-mini) at their pinned
revisions and decoder settings; exact 72-outcome laws for all 145 states
of the 50 held-out questions; generation audit on sampled states
including faulted ones. A two-question smoke test on the development
panel checks the pipeline only; no method comparison is run on it.

**Simulation.** Fault probabilities p in {0.01, 0.03, 0.10}; alpha 0.1;
budgets 100, 200, 400 queries per kernel; 300 replications; the standard
pilot, floor, rounding, and charged pilots. Methods on coupled streams:
uniform, learned occupancy, learned mean, plain TIS, anchored TIS,
occ_plus_uniform, occ_plus_mean, oracle+floor (reference), complete
rollouts (independent stream).

**Primary output.** Per generator and p: panel-MSE ratios with paired z of
plain and anchored TIS against occupancy, learned mean, both mixtures, and
rollouts. Population diagnostics: exact CVaR, margins, uniform-to-oracle
ratio, and the share of tail influence versus visitation carried by the
faulted states.

**Interpretation, recorded in advance.**
- Tail designs beat both mixtures -> tail-specific value in a real LLM
  workflow with rare tool faults.
- Anchor ~ occ_plus_uniform -> regularization, not tail learning.
- occ_plus_mean wins -> tail specificity not supported here either; the
  paper says so.
All cells reported regardless of direction; no case filtered on outcome.

## Addendum (declared 2026-09-24, before any unit-check calibration)

The identical design is also run with the unit-check review workflow
(reviewer instructed to audit units, signs, and scale), same generators,
fault model, probabilities, budgets, replications, and methods
(configs/finqa_toolfault_unitcheck_*.json). Additional question: does the
unit-audit reviewer lower the lower-tail severity CVaR under calculator
faults relative to the ordinary reviewer? Reported as exact population
CVaR per generator and fault probability for both workflows, alongside
the same estimation contrasts. All cells reported regardless of direction.
