from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator, Literal, Mapping

import numpy as np


Group = tuple[int, int]
GroupMode = Literal["shared_state_action", "untied_layer_state"]


@dataclass(frozen=True)
class Kernel:
    rewards: np.ndarray
    next_states: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        rewards = np.asarray(self.rewards, dtype=float)
        next_states = np.asarray(self.next_states, dtype=int)
        probabilities = np.asarray(self.probabilities, dtype=float)
        if not (rewards.ndim == next_states.ndim == probabilities.ndim == 1):
            raise ValueError("Kernel arrays must be one-dimensional")
        if not (len(rewards) == len(next_states) == len(probabilities)):
            raise ValueError("Kernel arrays must have equal lengths")
        if len(rewards) == 0 or np.any(probabilities < 0):
            raise ValueError("Kernel probabilities must be nonnegative and nonempty")
        total = float(probabilities.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Kernel probabilities must have positive finite mass")
        probabilities = probabilities / total
        object.__setattr__(self, "rewards", rewards)
        object.__setattr__(self, "next_states", next_states)
        object.__setattr__(self, "probabilities", probabilities)

    @property
    def size(self) -> int:
        return len(self.probabilities)

    def with_probabilities(self, probabilities: np.ndarray) -> "Kernel":
        return Kernel(self.rewards, self.next_states, probabilities)


@dataclass(frozen=True)
class FiniteHorizonModel:
    name: str
    horizon: int
    initial_state: int
    grid: np.ndarray
    policy: np.ndarray
    kernels: Mapping[Group, Kernel]
    query_groups: tuple[Group, ...] | None = None
    metadata: Mapping[str, object] | None = None
    group_mode: GroupMode = "shared_state_action"

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid, dtype=float)
        policy = np.asarray(self.policy, dtype=float)
        if grid.ndim != 1 or len(grid) < 2 or np.any(np.diff(grid) <= 0):
            raise ValueError("Grid must be a strictly increasing vector")
        if abs(grid[0]) > 1e-12 or abs(grid[-1] - self.horizon) > 1e-10:
            raise ValueError("Grid must contain finite-horizon endpoints 0 and H")
        if policy.ndim != 3 or policy.shape[0] != self.horizon + 1:
            raise ValueError("Policy shape must be (H+1, states, actions)")
        if np.any(policy < -1e-14):
            raise ValueError("Policy probabilities must be nonnegative")
        if not np.allclose(policy[1:].sum(axis=2), 1.0, atol=1e-10):
            raise ValueError("Each active policy row must sum to one")
        states, actions = policy.shape[1], policy.shape[2]
        if self.group_mode not in {"shared_state_action", "untied_layer_state"}:
            raise ValueError(f"Unknown group mode {self.group_mode!r}")
        for first, second in self.kernels:
            if self.group_mode == "shared_state_action":
                valid = 0 <= first < states and 0 <= second < actions
            else:
                valid = 1 <= first <= self.horizon and 0 <= second < states
            if not valid:
                raise ValueError("Kernel group outside model dimensions")
        for kernel in self.kernels.values():
            if np.any(kernel.next_states < 0) or np.any(kernel.next_states >= states):
                raise ValueError("Kernel next state outside state space")
            if np.any(kernel.rewards < -1e-12) or np.any(kernel.rewards > 1 + 1e-12):
                raise ValueError("Finite-horizon rewards must lie in [0,1]")
        if self.group_mode == "shared_state_action":
            for h in range(1, self.horizon + 1):
                for state in range(states):
                    for action in np.flatnonzero(policy[h, state] > 0):
                        if (state, int(action)) not in self.kernels:
                            raise ValueError(
                                f"Missing active kernel {(state, int(action))}"
                            )
        else:
            missing = [
                (h, state)
                for h in range(1, self.horizon + 1)
                for state in range(states)
                if (h, state) not in self.kernels
            ]
            if missing:
                raise ValueError(f"Missing untied layer-state kernel {missing[0]}")
        groups = self.query_groups
        if groups is None:
            if self.group_mode == "shared_state_action":
                active = {
                    (s, a)
                    for h in range(1, self.horizon + 1)
                    for s in range(states)
                    for a in range(actions)
                    if policy[h, s, a] > 0
                }
            else:
                active = set(self.kernels)
            groups = tuple(sorted(active))
        else:
            groups = tuple(groups)
        if len(set(groups)) != len(groups):
            raise ValueError("Query groups must be unique")
        if not set(groups).issubset(self.kernels):
            raise ValueError("Every query group must have a kernel")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "query_groups", groups)

    @property
    def state_count(self) -> int:
        return self.policy.shape[1]

    @property
    def action_count(self) -> int:
        return self.policy.shape[2]

    @property
    def group_count(self) -> int:
        return len(self.query_groups or ())

    def row_components(
        self, h: int, state: int
    ) -> Iterator[tuple[float, Group, Kernel]]:
        """Yield the laws entering one Bellman row.

        Shared groups are action-conditional and receive their policy weight.
        An untied layer-state group is already the policy-mixture law, so it
        enters exactly once with weight one.
        """
        if self.group_mode == "untied_layer_state":
            group = (h, state)
            yield 1.0, group, self.kernels[group]
            return
        for action in range(self.action_count):
            weight = float(self.policy[h, state, action])
            if weight:
                group = (state, action)
                yield weight, group, self.kernels[group]

    def replace_query_probabilities(
        self, probabilities: Mapping[Group, np.ndarray]
    ) -> "FiniteHorizonModel":
        kernels = dict(self.kernels)
        for group in self.query_groups or ():
            if group not in probabilities:
                raise ValueError(f"Missing empirical probabilities for {group}")
            kernels[group] = kernels[group].with_probabilities(probabilities[group])
        return replace(self, kernels=kernels)


def model_from_counts(
    model: FiniteHorizonModel, counts: Mapping[Group, np.ndarray]
) -> FiniteHorizonModel:
    probabilities: dict[Group, np.ndarray] = {}
    for group in model.query_groups or ():
        values = np.asarray(counts[group], dtype=float)
        if values.shape != model.kernels[group].probabilities.shape:
            raise ValueError(f"Count shape mismatch for {group}")
        if np.any(values < 0) or values.sum() <= 0:
            raise ValueError(f"Counts for {group} must have positive mass")
        probabilities[group] = values / values.sum()
    return model.replace_query_probabilities(probabilities)


def sample_counts(
    model: FiniteHorizonModel,
    sample_sizes: Mapping[Group, int],
    rng: np.random.Generator,
) -> dict[Group, np.ndarray]:
    result: dict[Group, np.ndarray] = {}
    for group in model.query_groups or ():
        n = int(sample_sizes[group])
        if n <= 0:
            raise ValueError("Sample sizes must be positive")
        result[group] = rng.multinomial(n, model.kernels[group].probabilities)
    return result


def coupled_streams(
    model: FiniteHorizonModel,
    maximum_sizes: Mapping[Group, int],
    rng: np.random.Generator,
) -> dict[Group, np.ndarray]:
    streams: dict[Group, np.ndarray] = {}
    groups = tuple(model.query_groups or ())
    # Draw every group seed before any outcomes. Increasing one method's prefix
    # can then extend that group's stream without shifting any other group.
    seeds = rng.integers(0, np.iinfo(np.uint64).max, size=len(groups), dtype=np.uint64)
    for group, seed in zip(groups, seeds):
        n = int(maximum_sizes[group])
        group_rng = np.random.default_rng(seed)
        streams[group] = group_rng.choice(
            model.kernels[group].size,
            size=n,
            replace=True,
            p=model.kernels[group].probabilities,
        )
    return streams


def counts_from_streams(
    model: FiniteHorizonModel,
    streams: Mapping[Group, np.ndarray],
    sample_sizes: Mapping[Group, int],
) -> dict[Group, np.ndarray]:
    counts: dict[Group, np.ndarray] = {}
    for group in model.query_groups or ():
        n = int(sample_sizes[group])
        stream = np.asarray(streams[group])
        if n > len(stream):
            raise ValueError("Requested prefix exceeds coupled stream")
        counts[group] = np.bincount(
            stream[:n], minlength=model.kernels[group].size
        ).astype(int)
    return counts
