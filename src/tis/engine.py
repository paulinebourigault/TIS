from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import FiniteHorizonModel, Group


@dataclass(frozen=True)
class BellmanSolution:
    stop_loss: np.ndarray
    cdf: np.ndarray
    probabilities: np.ndarray
    quantile_index: int
    quantile: float
    cvar: float
    margin: float


@dataclass(frozen=True)
class InfluenceBundle:
    groups: tuple[Group, ...]
    tail_values: tuple[np.ndarray, ...]
    tail_components: tuple[np.ndarray, ...]
    tail_scales: np.ndarray
    no_cov_scales: np.ndarray
    reachability_scales: np.ndarray
    local_scales: np.ndarray
    mean_values: tuple[np.ndarray, ...]
    mean_components: tuple[np.ndarray, ...]
    mean_scales: np.ndarray
    adjoint: np.ndarray
    occupancy: np.ndarray
    value: np.ndarray
    solution: BellmanSolution


def _target_vector(
    model: FiniteHorizonModel,
    stop_loss: np.ndarray,
    h: int,
    reward: float,
    next_state: int,
) -> np.ndarray:
    shifted = model.grid - reward
    if h == 1:
        return np.maximum(shifted, 0.0)
    return np.interp(
        shifted,
        model.grid,
        stop_loss[h - 1, next_state],
        left=0.0,
        right=float(stop_loss[h - 1, next_state, -1]),
    )


_INTERP_CACHE: dict[tuple[bytes, bytes], tuple[np.ndarray, ...]] = {}
_INTERP_CACHE_LIMIT = 256


def _interp_structure(
    grid: np.ndarray, rewards: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared piecewise-linear interpolation structure for shifted thresholds.

    For every reward r and grid threshold q, the Bellman target evaluates the
    continuation stop loss at q - r. Because the stop loss is piecewise linear
    on the grid, that evaluation is `(1-w)*V[lo] + w*V[hi]` on the bracketing
    grid cell, zero when q - r <= 0, and the top grid value when q - r >= H.
    The bracketing indices depend only on (grid, rewards), so they are cached
    across the empirical models of a Monte Carlo run.
    """
    key = (grid.tobytes(), rewards.tobytes())
    cached = _INTERP_CACHE.get(key)
    if cached is not None:
        return cached  # type: ignore[return-value]
    shifted = grid[None, :] - rewards[:, None]
    below = shifted <= 0.0
    above = shifted >= grid[-1]
    lo = np.searchsorted(grid, shifted, side="right") - 1
    lo = np.clip(lo, 0, len(grid) - 2)
    hi = lo + 1
    weight = (shifted - grid[lo]) / (grid[hi] - grid[lo])
    weight = np.clip(weight, 0.0, 1.0)
    weight[above] = 1.0
    interior = ~(below | above)
    result = (lo, hi, weight, below, interior)
    if len(_INTERP_CACHE) >= _INTERP_CACHE_LIMIT:
        _INTERP_CACHE.clear()
    _INTERP_CACHE[key] = result
    return result


def _kernel_targets(
    model: FiniteHorizonModel, previous: np.ndarray, kernel
) -> np.ndarray:
    """Rows of sampled Bellman targets, one per kernel outcome.

    Row i equals `interp(grid - rewards[i], grid, previous[next_states[i]])`
    with the stop-loss boundary conventions (zero left of the grid, top value
    right of it).
    """
    lo, hi, weight, below, _ = _interp_structure(model.grid, kernel.rewards)
    values = previous[kernel.next_states]
    gathered_lo = np.take_along_axis(values, lo, axis=1)
    gathered_hi = np.take_along_axis(values, hi, axis=1)
    targets = gathered_lo + weight * (gathered_hi - gathered_lo)
    targets[below] = 0.0
    return targets


def _stationary_policy(model: FiniteHorizonModel) -> bool:
    if model.group_mode == "untied_layer_state":
        return False
    return bool(
        np.allclose(
            model.policy[1:],
            model.policy[1][None, :, :],
            atol=0.0,
            rtol=0.0,
        )
    )


def _rows_by_kernel_signature(
    model: FiniteHorizonModel, h: int
) -> dict[tuple[int, int], list[tuple[int, float, object]]]:
    """Group Bellman rows whose kernels share reward/next-state arrays.

    Kernels produced by ``with_probabilities`` keep the same reward and
    next-state array objects, so every empirical model of a Monte Carlo run
    reuses the group structure. Rows in one group share the sampled-target
    matrix; only their outcome probabilities differ.
    """
    grouped: dict[tuple[int, int], list[tuple[int, float, object]]] = {}
    for state in range(model.state_count):
        for weight, _, kernel in model.row_components(h, state):
            key = (id(kernel.rewards), id(kernel.next_states))
            grouped.setdefault(key, []).append((state, weight, kernel))
    return grouped


def _apply_stop_loss(
    model: FiniteHorizonModel, h: int, previous: np.ndarray
) -> np.ndarray:
    """Apply one Bellman layer without materialising a dense SK-by-SK matrix."""
    result = np.zeros_like(previous)
    for rows in _rows_by_kernel_signature(model, h).values():
        targets = _kernel_targets(model, previous, rows[0][2])
        states = np.asarray([state for state, _, _ in rows], dtype=int)
        weighted = np.vstack(
            [weight * kernel.probabilities for _, weight, kernel in rows]
        )
        np.add.at(result, states, weighted @ targets)
    return result


def _first_step_stop_loss(model: FiniteHorizonModel) -> np.ndarray:
    result = np.zeros((model.state_count, len(model.grid)), dtype=float)
    for state in range(model.state_count):
        for weight, _, kernel in model.row_components(1, state):
            base = np.maximum(
                model.grid[None, :] - kernel.rewards[:, None], 0.0
            )
            result[state] += weight * (kernel.probabilities @ base)
    return result


def _mean_operator(
    model: FiniteHorizonModel, h: int
) -> tuple[np.ndarray, np.ndarray]:
    reward_vector = np.zeros(model.state_count, dtype=float)
    transition = np.zeros((model.state_count, model.state_count), dtype=float)
    for state in range(model.state_count):
        for weight, _, kernel in model.row_components(h, state):
            reward_vector[state] += weight * float(
                np.sum(kernel.probabilities * kernel.rewards)
            )
            np.add.at(
                transition[state],
                kernel.next_states,
                weight * kernel.probabilities,
            )
    return reward_vector, transition


def solve_stop_loss(
    model: FiniteHorizonModel, alpha: float
) -> BellmanSolution:
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0,1)")
    hmax, states, grid_size = model.horizon, model.state_count, len(model.grid)
    stop_loss = np.zeros((hmax + 1, states, grid_size), dtype=float)
    stop_loss[1] = _first_step_stop_loss(model)
    for h in range(2, hmax + 1):
        stop_loss[h] = _apply_stop_loss(model, h, stop_loss[h - 1])

    root_u = stop_loss[hmax, model.initial_state]
    differences = np.diff(root_u) / np.diff(model.grid)
    cdf = np.empty(grid_size, dtype=float)
    cdf[:-1] = differences
    cdf[-1] = 1.0
    cdf = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
    probabilities = np.diff(np.concatenate(([0.0], cdf)))
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= probabilities.sum()
    cdf = np.cumsum(probabilities)
    quantile_index = int(np.searchsorted(cdf, alpha, side="left"))
    quantile = float(model.grid[quantile_index])
    cvar = quantile - float(root_u[quantile_index]) / alpha
    previous = 0.0 if quantile_index == 0 else float(cdf[quantile_index - 1])
    margin = min(alpha - previous, float(cdf[quantile_index]) - alpha)
    return BellmanSolution(
        stop_loss=stop_loss,
        cdf=cdf,
        probabilities=probabilities,
        quantile_index=quantile_index,
        quantile=quantile,
        cvar=cvar,
        margin=margin,
    )


def _add_interpolation_transpose(
    destination: np.ndarray,
    grid: np.ndarray,
    shifted: np.ndarray,
    weights: np.ndarray,
) -> None:
    valid = shifted > 0
    if not np.any(valid):
        return
    values = shifted[valid]
    mass = weights[valid]
    hi = np.searchsorted(grid, values, side="left")
    hi = np.clip(hi, 1, len(grid) - 1)
    exact = np.isclose(grid[hi], values, atol=1e-12, rtol=0)
    if np.any(exact):
        np.add.at(destination, hi[exact], mass[exact])
    between = ~exact
    if np.any(between):
        hidx = hi[between]
        lidx = hidx - 1
        vals = values[between]
        denom = grid[hidx] - grid[lidx]
        high_weight = (vals - grid[lidx]) / denom
        np.add.at(destination, lidx, mass[between] * (1.0 - high_weight))
        np.add.at(destination, hidx, mass[between] * high_weight)


def solve_adjoint(
    model: FiniteHorizonModel, solution: BellmanSolution
) -> np.ndarray:
    adjoint = np.zeros_like(solution.stop_loss)
    adjoint[
        model.horizon, model.initial_state, solution.quantile_index
    ] = 1.0
    for h in range(model.horizon, 1, -1):
        for rows in _rows_by_kernel_signature(model, h).values():
            active = [
                (state, weight, kernel)
                for state, weight, kernel in rows
                if np.any(adjoint[h, state])
            ]
            if not active:
                continue
            kernel = active[0][2]
            lo, hi, cell_weight, below, _ = _interp_structure(
                model.grid, kernel.rewards
            )
            weighted = np.vstack(
                [weight * kernel.probabilities for _, weight, kernel in active]
            )
            layer_weights = np.vstack([adjoint[h, state] for state, _, _ in active])
            mass = weighted.T @ layer_weights
            mass = np.where(below, 0.0, mass)
            destination = adjoint[h - 1]
            np.add.at(
                destination,
                (kernel.next_states[:, None], lo),
                mass * (1.0 - cell_weight),
            )
            np.add.at(
                destination,
                (kernel.next_states[:, None], hi),
                mass * cell_weight,
            )
    return adjoint


def solve_mean(model: FiniteHorizonModel) -> np.ndarray:
    value = np.zeros((model.horizon + 1, model.state_count), dtype=float)
    stationary_operator = _mean_operator(model, 1) if _stationary_policy(model) else None
    for h in range(1, model.horizon + 1):
        reward_vector, transition = (
            stationary_operator
            if stationary_operator is not None
            else _mean_operator(model, h)
        )
        value[h] = reward_vector + transition @ value[h - 1]
    return value


def solve_occupancy(model: FiniteHorizonModel) -> np.ndarray:
    occupancy = np.zeros((model.horizon + 1, model.state_count), dtype=float)
    occupancy[model.horizon, model.initial_state] = 1.0
    stationary_operator = _mean_operator(model, 1) if _stationary_policy(model) else None
    for h in range(model.horizon, 1, -1):
        _, transition = (
            stationary_operator
            if stationary_operator is not None
            else _mean_operator(model, h)
        )
        occupancy[h - 1] = transition.T @ occupancy[h]
    return occupancy


def compute_influences(
    model: FiniteHorizonModel, alpha: float
) -> InfluenceBundle:
    solution = solve_stop_loss(model, alpha)
    adjoint = solve_adjoint(model, solution)
    value = solve_mean(model)
    occupancy = solve_occupancy(model)
    target_memo: dict[tuple[int, int, int], np.ndarray] = {}
    tail_values: list[np.ndarray] = []
    tail_components: list[np.ndarray] = []
    tail_scales: list[float] = []
    no_cov_scales: list[float] = []
    reachability_scales: list[float] = []
    local_scales: list[float] = []
    mean_values: list[np.ndarray] = []
    mean_components: list[np.ndarray] = []
    mean_scales: list[float] = []

    for group in model.query_groups or ():
        kernel = model.kernels[group]
        tail_by_layer = np.zeros((kernel.size, model.horizon), dtype=float)
        mean_by_layer = np.zeros((kernel.size, model.horizon), dtype=float)
        if model.group_mode == "untied_layer_state":
            selected_layers = ((group[0], group[1], 1.0),)
        else:
            state, action = group
            selected_layers = tuple(
                (h, state, float(model.policy[h, state, action]))
                for h in range(1, model.horizon + 1)
                if model.policy[h, state, action] > 0
            )
        reachability = 0.0
        for h, state, row_weight in selected_layers:
            memo_key = (h, id(kernel.rewards), id(kernel.next_states))
            targets = target_memo.get(memo_key)
            if targets is None:
                if h == 1:
                    targets = np.maximum(
                        model.grid[None, :] - kernel.rewards[:, None], 0.0
                    )
                else:
                    targets = _kernel_targets(
                        model, solution.stop_loss[h - 1], kernel
                    )
                target_memo[memo_key] = targets
            centered_targets = targets - np.sum(
                kernel.probabilities[:, None] * targets, axis=0
            )
            tail_by_layer[:, h - 1] = (
                -row_weight
                / alpha
                * (centered_targets @ adjoint[h, state])
            )
            reachability += row_weight * float(adjoint[h, state].sum())

            raw_mean_targets = (
                kernel.rewards + value[h - 1, kernel.next_states]
            )
            centered_mean = raw_mean_targets - float(
                np.sum(kernel.probabilities * raw_mean_targets)
            )
            mean_by_layer[:, h - 1] = (
                occupancy[h, state] * row_weight * centered_mean
            )

        tail = tail_by_layer.sum(axis=1)
        mean = mean_by_layer.sum(axis=1)
        tail_variance = float(np.sum(kernel.probabilities * tail**2))
        diagonal_variance = float(
            np.sum(kernel.probabilities[:, None] * tail_by_layer**2)
        )
        mean_variance = float(np.sum(kernel.probabilities * mean**2))
        tail_values.append(tail)
        tail_components.append(tail_by_layer)
        tail_scales.append(np.sqrt(max(tail_variance, 0.0)))
        no_cov_scales.append(np.sqrt(max(diagonal_variance, 0.0)))
        reachability_scales.append(reachability)
        local_scales.append(
            np.sqrt(max(tail_variance, 0.0)) / reachability
            if reachability > 0
            else 0.0
        )
        mean_values.append(mean)
        mean_components.append(mean_by_layer)
        mean_scales.append(np.sqrt(max(mean_variance, 0.0)))

    return InfluenceBundle(
        groups=tuple(model.query_groups or ()),
        tail_values=tuple(tail_values),
        tail_components=tuple(tail_components),
        tail_scales=np.asarray(tail_scales),
        no_cov_scales=np.asarray(no_cov_scales),
        reachability_scales=np.asarray(reachability_scales),
        local_scales=np.asarray(local_scales),
        mean_values=tuple(mean_values),
        mean_components=tuple(mean_components),
        mean_scales=np.asarray(mean_scales),
        adjoint=adjoint,
        occupancy=occupancy,
        value=value,
        solution=solution,
    )


def categorical_projection(
    grid: np.ndarray, support: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    result = np.zeros(len(grid), dtype=float)
    for value, probability in zip(support, probabilities):
        if value <= grid[0]:
            result[0] += probability
        elif value >= grid[-1]:
            result[-1] += probability
        else:
            hi = int(np.searchsorted(grid, value, side="left"))
            if np.isclose(grid[hi], value, atol=1e-12, rtol=0):
                result[hi] += probability
            else:
                lo = hi - 1
                w_hi = (value - grid[lo]) / (grid[hi] - grid[lo])
                result[lo] += probability * (1.0 - w_hi)
                result[hi] += probability * w_hi
    return result


def solve_probability_form(model: FiniteHorizonModel) -> np.ndarray:
    eta = np.zeros(
        (model.horizon + 1, model.state_count, len(model.grid)), dtype=float
    )
    eta[0, :, 0] = 1.0
    for h in range(1, model.horizon + 1):
        for state in range(model.state_count):
            row = np.zeros(len(model.grid), dtype=float)
            for weight, _, kernel in model.row_components(h, state):
                for reward, next_state, probability in zip(
                    kernel.rewards, kernel.next_states, kernel.probabilities
                ):
                    row += weight * probability * categorical_projection(
                        model.grid,
                        float(reward) + model.grid,
                        eta[h - 1, int(next_state)],
                    )
            eta[h, state] = row
    return eta
