from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from .allocation import (
    design_constant,
    integer_allocation,
    normalized_weights,
    regularized_weights,
)
from .engine import compute_influences, solve_mean, solve_probability_form, solve_stop_loss
from .environments import make_environment


def validate_environment(name: str, alpha: float) -> dict[str, object]:
    model = make_environment(name)
    bundle = compute_influences(model, alpha)
    eta = solve_probability_form(model)
    root_probability = eta[model.horizon, model.initial_state]
    probability_error = float(np.max(np.abs(root_probability - bundle.solution.probabilities)))

    group_index = int(np.argmax(bundle.tail_scales))
    group = bundle.groups[group_index]
    outcome = int(np.argmax(np.abs(bundle.tail_values[group_index])))
    epsilon = 1e-6
    kernel = model.kernels[group]
    contaminated = (1.0 - epsilon) * kernel.probabilities
    contaminated[outcome] += epsilon
    perturbed = model.replace_query_probabilities(
        {
            retained: contaminated if retained == group else model.kernels[retained].probabilities
            for retained in model.query_groups or ()
        }
    )
    finite_difference = (
        solve_stop_loss(perturbed, alpha).cvar - bundle.solution.cvar
    ) / epsilon
    influence_derivative = float(bundle.tail_values[group_index][outcome])

    finite_difference_mean = (
        solve_mean(perturbed)[model.horizon, model.initial_state]
        - bundle.value[model.horizon, model.initial_state]
    ) / epsilon
    mean_derivative = float(bundle.mean_values[group_index][outcome])

    component = bundle.tail_components[group_index]
    probabilities = kernel.probabilities
    covariance = component.T @ (probabilities[:, None] * component)
    variance_of_sum = float(np.ones(model.horizon) @ covariance @ np.ones(model.horizon))
    diagonal_sum = float(np.trace(covariance))

    oracle_weights = normalized_weights(bundle.tail_scales)
    oracle_value = float(bundle.tail_scales.sum() ** 2)
    uniform_value = design_constant(
        bundle.tail_scales, np.full(model.group_count, 1.0 / model.group_count)
    )
    feasible_oracle_weights = regularized_weights(bundle.tail_scales, 1e-6)
    counts = integer_allocation(
        100 * model.group_count, feasible_oracle_weights, 2
    )

    checks = {
        "probability_stop_loss_agreement": bool(probability_error < 1e-9),
        "tail_finite_difference": bool(
            abs(finite_difference - influence_derivative) < 2e-4
        ),
        "mean_finite_difference": bool(
            abs(finite_difference_mean - mean_derivative) < 2e-5
        ),
        "shared_variance_identity": bool(
            abs(variance_of_sum - bundle.tail_scales[group_index] ** 2) < 1e-10
        ),
        "dropped_covariance_identity": bool(
            abs(diagonal_sum - bundle.no_cov_scales[group_index] ** 2) < 1e-10
        ),
        "oracle_no_worse_than_uniform": bool(
            oracle_value <= uniform_value + 1e-10
        ),
        "integer_budget": bool(
            int(counts.sum()) == 100 * model.group_count
        ),
        "positive_margin": bool(bundle.solution.margin > 0),
    }
    if model.group_mode == "untied_layer_state":
        checks["untied_scale_factorization"] = bool(
            np.max(
                np.abs(
                    bundle.tail_scales
                    - bundle.reachability_scales * bundle.local_scales
                )
            )
            < 1e-12
        )
    return {
        "environment": name,
        "alpha": alpha,
        "truth": bundle.solution.cvar,
        "margin": bundle.solution.margin,
        "groups": model.group_count,
        "probability_error": probability_error,
        "tail_finite_difference": finite_difference,
        "tail_influence": influence_derivative,
        "mean_finite_difference": finite_difference_mean,
        "mean_influence": mean_derivative,
        "checks": checks,
        "passed": all(checks.values()),
    }


def validate_population_reference(
    name: str,
    alpha: float,
    expected_groups: int,
    expected_truth: float,
    expected_margin: float,
    expected_ratio: float,
    tolerance: float = 5e-7,
    ratio_tolerance: float = 1e-4,
) -> dict[str, object]:
    """Validate frozen public specifications against manuscript constants."""
    model = make_environment(name)
    bundle = compute_influences(model, alpha)
    ratio = (
        model.group_count
        * float(np.sum(bundle.tail_scales**2))
        / float(bundle.tail_scales.sum() ** 2)
    )
    checks = {
        "group_count": model.group_count == expected_groups,
        "truth": abs(bundle.solution.cvar - expected_truth) < tolerance,
        "margin": abs(bundle.solution.margin - expected_margin) < tolerance,
        "uniform_to_oracle": abs(ratio - expected_ratio) < ratio_tolerance,
        "positive_margin": bundle.solution.margin > 0,
        "nonzero_tail_signal": float(bundle.tail_scales.sum()) > 0,
    }
    return {
        "environment": name,
        "alpha": alpha,
        "truth": bundle.solution.cvar,
        "margin": bundle.solution.margin,
        "groups": model.group_count,
        "uniform_to_oracle": ratio,
        "expected": {
            "truth": expected_truth,
            "margin": expected_margin,
            "groups": expected_groups,
            "uniform_to_oracle": expected_ratio,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_validation() -> list[dict[str, object]]:
    reports = [
        validate_environment("controlled_shared", 0.1),
        validate_environment("controlled_untied_v1", 0.1),
        validate_environment("inventory_untied_v1", 0.1),
        validate_environment("frozenlake_8x8", 0.2),
        validate_population_reference(
            "frozenlake_8x8", 0.2, 50, 0.3147769, 0.0629554, 3.92589
        ),
        validate_population_reference(
            "cliffwalking_primary", 0.1, 149, 11.8709354, 0.0139085, 84.6973
        ),
        validate_population_reference(
            "cliffwalking_safe", 0.05, 149, 14.2554974, 0.0100618, 116.3908
        ),
        validate_population_reference(
            "cliffwalking_primary", 0.2, 149, 12.7966208, 0.0041269, 77.5748
        ),
        validate_population_reference(
            "rainy_taxi", 0.1, 37, 9.0541409, 0.0121392, 2.33938
        ),
        validate_population_reference(
            "controlled_untied_v1",
            0.1,
            21,
            0.9086716125743333,
            0.010685396723788224,
            1.7233238733483527,
            tolerance=1e-12,
            ratio_tolerance=1e-12,
        ),
        validate_population_reference(
            "inventory_untied_v1",
            0.1,
            41,
            1.1512371171695401,
            0.04362019546849408,
            1.8235977736549085,
            tolerance=1e-12,
            ratio_tolerance=1e-12,
        ),
    ]
    if not all(report["passed"] for report in reports):
        raise RuntimeError(json.dumps(reports, indent=2))
    return reports
