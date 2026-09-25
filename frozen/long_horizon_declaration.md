# Declared test: longer review loops (MMLU-Pro H = 8, 10; FinQA H = 6)

Declared 2026-09-24, after computing the divergence statistic below and
before any simulation at these horizons. No new model calls: prompts carry
no stage index, so the frozen kernels define every horizon.

**Question.** Longer loops reuse each prompt kernel more often, which the
theory says favors shared-kernel allocation over rollouts, and they give the
pilot more chances to miss rare outcomes, which favors the anchor over plain
TIS. Does the anchor's standing against rollouts and against the mean blend
move as predicted when the loop grows?

**Settings.**
- MMLU-Pro primary panel, confident-error utility, default policy, closed
  grid, alpha = 0.1, H in {8, 10}: Qwen3-4B (qwen_kernels), Phi-4-mini
  (robustness_kernels), GLM-4-32B (scale3_kernels). Six cells. Reference: the
  existing H = 6 confident-error blending runs (same seed, same methods).
- FinQA held-out panel, H = 6, alpha = 0.1: Qwen3-4B and Phi-4-mini, ordinary
  and unit-check workflows. Four cells. Reference: the existing H = 3
  blending runs.
- Budgets 100, 200, 400 queries per kernel; 300 replications; the seeds of the
  reference runs; methods uniform, learned occupancy, learned mean, plain TIS,
  anchored TIS, occ_plus_uniform, occ_plus_mean, oracle+floor, complete
  rollouts. Standard pilot, floor, rounding, charged pilots.

**Divergence statistic** (median over questions of the total-variation
distance between normalized tail and mean influence shares; MMLU: first 25
questions; FinQA: all 50):

| Setting | reference H | new H |
|---|---|---|
| MMLU Qwen3-4B, confident-error | .321 (H=6) | .255 (H=8), .264 (H=10) |
| MMLU Phi-4-mini, confident-error | .175 (H=6) | .191 (H=8), .193 (H=10) |
| MMLU GLM-4-32B, confident-error | .317 (H=6) | .355 (H=8), .389 (H=10) |
| FinQA Qwen ordinary | .000 (H=3) | .000 (H=6) |
| FinQA Qwen unit check | .000 (H=3) | .000 (H=6) |
| FinQA Phi ordinary | .000 (H=3) | .016 (H=6) |
| FinQA Phi unit check | .000 (H=3) | .098 (H=6) |

**Predictions (fixed now). "Resolved" means |z| >= 2 on paired
replications; negative favors the anchor.**
- P1 (divergence rule, same as the out-of-sample test). Distance >= .24
  predicts WIN against occ_plus_mean: resolved better in at least 2 of 3
  budgets and resolved worse in none. Distance <= .18 predicts NO ADVANTAGE:
  resolved better in at most 1 of 3. In between: no prediction. This gives
  WIN for MMLU Qwen3-4B and GLM-4-32B at H = 8 and 10 (four cells), no
  prediction for MMLU Phi-4-mini (.19), and NO ADVANTAGE for all four FinQA
  H = 6 cells.
- P2 (kernel reuse). At 400 queries per kernel, the anchor/rollout MSE ratio
  is lower at the longest new horizon than at the reference horizon, in each
  setting. References: MMLU H = 6: Qwen3-4B .80, Phi-4-mini .94,
  GLM-4-32B .90 (compared with H = 10). FinQA H = 3: Qwen ordinary 1.62,
  Qwen unit check 1.74, Phi ordinary .77, Phi unit check .88 (compared with
  H = 6).
- P3 (pilot risk). At 400 queries per kernel, plain TIS/anchor MSE ratio is
  higher at H = 10 than at H = 6 for each MMLU generator (references 18.98,
  7.35, 30.12).

Each prediction is scored per setting and every cell is reported whatever
the direction. If P2 fails, the paper does not claim that longer loops
favor conditional sampling.
