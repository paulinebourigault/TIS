"""Structured language-model workflow study."""

LLM_METHODS = (
    "uniform",
    "occupancy",
    "mean_influence",
    "answer_entropy",
    "oracle_tail",
    "tis",
    "mc_ucb_frozen",
    "adaptive_neyman",
    "untied_tis",
)

LLM_EXTENSION_METHODS = (
    "learned_mean",
    "learned_occupancy",
    "tis_no_cov",
    "oracle_no_cov",
)

# The extension run's method panel: anchors shared with the primary panel
# (whose coupled-stream errors it must reproduce exactly) plus the extension.
LLM_EXTENSION_PANEL = (
    "uniform",
    "oracle_tail",
    "tis",
) + LLM_EXTENSION_METHODS

# Complete-rollout baseline: empirical CVaR from floor(N/H) independent full
# trajectories through the same conditional laws, at matched total query
# cost. The anchors again reproduce the primary panel exactly.
LLM_ROLLOUT_PANEL = (
    "uniform",
    "tis",
    "complete_rollout",
)

