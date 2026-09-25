# Declared ablation: mixture controls on MMLU under the confident-error utility

Declared 2026-09-24, before any run of this configuration. Retrospective
with respect to the already-examined MMLU panels; labeled as such.

**Question.** On FinQA (aligned tail and mean influence), an equal blend of
learned mean influence and occupancy matched or beat the anchor. Under the
confident-error utility (correct: (1+c)/2; wrong: (1-c)^2/2), overconfident
errors dominate the lower tail without changing per-answer ordering, so
tail influence can diverge from mean influence. Does the tail score beat
the identically regularized mean blend there?

**Protocol.** Frozen MMLU-Pro kernels of all six generators
(Qwen3-4B, Phi-4-mini, Granite-4.2-8B, Mistral-24B, Qwen3-32B, GLM-4-32B);
closed grid; utility confident_error; H = 6; alpha = 0.1; budgets 100, 200,
400, 800 per shared group; 300 replications; pilot, floor, rounding, and
charged pilots as in the primary study. Methods on coupled streams:
uniform, learned occupancy, learned mean, plain TIS, anchored TIS,
occ_plus_uniform, occ_plus_mean, oracle+floor (reference), complete
rollouts (independent stream). Mixtures blend floored component weights
50/50 with no second floor. No new model calls. Outputs:
results/llm_confident_mix_<generator>/.

**Interpretation, recorded in advance** (same rules as the FinQA ablation):
- anchor beats both mixtures -> tail-specific value in an LLM setting;
- anchor ~ occ_plus_uniform -> regularization, not tail learning;
- occ_plus_mean wins -> tail specificity not supported for LLM workflows
  even under a tail-divergent utility; the paper says so.
All six generators and all budgets are reported regardless of direction.
