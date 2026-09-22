from __future__ import annotations

from collections import deque
import math
from typing import Iterable

import numpy as np

from .model import FiniteHorizonModel, Kernel


def _kernel(outcomes: Iterable[tuple[float, int, float]]) -> Kernel:
    """Coalesce identical observable (reward, next-state) outcomes."""
    mass: dict[tuple[float, int], float] = {}
    for reward, next_state, probability in outcomes:
        key = (float(reward), int(next_state))
        mass[key] = mass.get(key, 0.0) + float(probability)
    ordered = sorted(mass.items())
    return Kernel(
        rewards=np.asarray([key[0] for key, _ in ordered], dtype=float),
        next_states=np.asarray([key[1] for key, _ in ordered], dtype=int),
        probabilities=np.asarray([probability for _, probability in ordered]),
    )


def controlled_shared() -> FiniteHorizonModel:
    """Seed-specified shared-kernel benchmark."""
    horizon, states, actions = 5, 5, 2
    grid = np.arange(0.0, horizon + 0.25, 0.25)
    policy = np.zeros((horizon + 1, states, actions), dtype=float)
    for h in range(1, horizon + 1):
        for state in range(states):
            preferred = (state + h) % 2
            policy[h, state, preferred] = 0.72
            policy[h, state, 1 - preferred] = 0.28

    rng = np.random.default_rng(314159)
    reward_atoms = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    kernels: dict[tuple[int, int], Kernel] = {}
    for state in range(states):
        for action in range(actions):
            probabilities = rng.dirichlet(
                np.array([0.7, 1.2, 1.8, 0.9]) + 0.15 * state
            )
            rewards = rng.choice(reward_atoms, size=4, replace=True)
            rewards[0] = 0.0 if (state + action) % 2 == 0 else 0.25
            rewards[-1] = 1.0 if action == 1 else 0.75
            next_states = rng.integers(0, states, size=4)
            next_states[0] = state
            next_states[-1] = (state + action + 1) % states
            kernels[(state, action)] = Kernel(
                rewards, next_states, probabilities
            )
    return FiniteHorizonModel(
        name="controlled_shared",
        horizon=horizon,
        initial_state=0,
        grid=grid,
        policy=policy,
        kernels=kernels,
        metadata={
            "kind": "new controlled benchmark",
            "parameter_seed": 314159,
            "grid_step": 0.25,
        },
    )


def controlled_untied_v1() -> FiniteHorizonModel:
    """Predeclared v1 layer-state controlled instance.

    This is a one-pass seeded construction, not a search over seeds or
    parameters. Every layer-state law has full positive support over the
    Cartesian product of three rewards and five successor states.
    """
    horizon, states = 5, 5
    policy = np.zeros((horizon + 1, states, 1), dtype=float)
    policy[1:, :, 0] = 1.0
    reward_atoms = np.asarray([0.0, 0.5, 1.0])
    rng = np.random.default_rng(314159)
    kernels: dict[tuple[int, int], Kernel] = {}
    realized_parameters: dict[str, object] = {}
    for h in range(1, horizon + 1):
        for state in range(states):
            reward_concentration = np.asarray(
                [
                    0.80 + 0.10 * h,
                    1.05 + 0.08 * state,
                    0.90 + 0.06 * (h + state),
                ]
            )
            transition_concentration = np.asarray(
                [
                    0.75 + 0.04 * h + 0.03 * ((state + successor) % states)
                    for successor in range(states)
                ]
            )
            reward_probability = rng.dirichlet(reward_concentration)
            transition_probability = rng.dirichlet(transition_concentration)
            outcomes = [
                (
                    float(reward),
                    successor,
                    float(reward_probability[reward_index]
                          * transition_probability[successor]),
                )
                for reward_index, reward in enumerate(reward_atoms)
                for successor in range(states)
            ]
            kernels[(h, state)] = _kernel(outcomes)
            realized_parameters[f"{h},{state}"] = {
                "reward_concentration": reward_concentration.tolist(),
                "transition_concentration": transition_concentration.tolist(),
                "reward_probabilities": reward_probability.tolist(),
                "transition_probabilities": transition_probability.tolist(),
            }
    query_groups = ((horizon, 0),) + tuple(
        (h, state)
        for h in range(1, horizon)
        for state in range(states)
    )
    return FiniteHorizonModel(
        name="controlled_untied_v1",
        horizon=horizon,
        initial_state=0,
        grid=np.arange(0.0, horizon + 0.25, 0.5),
        policy=policy,
        kernels=kernels,
        query_groups=query_groups,
        group_mode="untied_layer_state",
        metadata={
            "kind": "predeclared controlled instance (one-pass specification)",
            "instance_version": 1,
            "declaration": "one pass; no rejection, parameter search, or outcome matching",
            "parameter_seed": 314159,
            "numpy_generator": "numpy.random.default_rng (PCG64)",
            "draw_order": "ascending remaining horizon h, then ascending state; reward Dirichlet then transition Dirichlet",
            "states": list(range(states)),
            "reward_support": reward_atoms.tolist(),
            "grid_step": 0.5,
            "queried_blocks": "root (5,0), then all five states at h=1,2,3,4",
            "realized_parameters": realized_parameters,
        },
    )


INVENTORY_CAPACITY = 6
INVENTORY_BASE_STOCK = (4, 4, 5, 5, 5, 4, 4, 5)
INVENTORY_POISSON_MEAN = (1.2, 1.6, 2.1, 2.6, 2.4, 1.9, 1.5, 1.1)


def _censored_poisson_probabilities(rate: float) -> np.ndarray:
    probabilities = np.asarray(
        [math.exp(-rate) * rate**demand / math.factorial(demand)
         for demand in range(INVENTORY_CAPACITY)],
        dtype=float,
    )
    return np.concatenate((probabilities, [1.0 - float(probabilities.sum())]))


def _inventory_reward(
    sales: int, order: int, ending_before_disruption: int, lost: int, disrupted: bool
) -> float:
    profit = (
        4.0 * sales
        - 1.5 * order
        - 0.5 * ending_before_disruption
        - 2.0 * lost
    )
    reward = 0.0 if profit <= 2.0 else (0.5 if profit < 8.0 else 1.0)
    if disrupted:
        reward = max(0.0, reward - 0.5)
    return reward


def inventory_untied_v1() -> FiniteHorizonModel:
    """Mechanically specified seasonal inventory replacement instance."""
    horizon, states = 8, INVENTORY_CAPACITY + 1
    policy = np.zeros((horizon + 1, states, 1), dtype=float)
    policy[1:, :, 0] = 1.0
    kernels: dict[tuple[int, int], Kernel] = {}
    for h in range(1, horizon + 1):
        season = horizon - h
        target = INVENTORY_BASE_STOCK[season]
        demand_probabilities = _censored_poisson_probabilities(
            INVENTORY_POISSON_MEAN[season]
        )
        for state in range(states):
            order = min(INVENTORY_CAPACITY - state, max(0, target - state))
            available = state + order
            disruption_probability = 0.08 if available == INVENTORY_CAPACITY else 0.06
            outcomes: list[tuple[float, int, float]] = []
            for demand, demand_probability in enumerate(demand_probabilities):
                sales = min(available, demand)
                lost = max(demand - available, 0)
                ending = max(available - demand, 0)
                for disrupted, event_probability in (
                    (False, 1.0 - disruption_probability),
                    (True, disruption_probability),
                ):
                    next_state = max(ending - int(disrupted), 0)
                    reward = _inventory_reward(
                        sales, order, ending, lost, disrupted
                    )
                    outcomes.append(
                        (
                            reward,
                            next_state,
                            float(demand_probability * event_probability),
                        )
                    )
            kernels[(h, state)] = _kernel(outcomes)

    reachable_by_layer: dict[int, set[int]] = {horizon: {0}}
    for h in range(horizon, 1, -1):
        following: set[int] = set()
        for state in reachable_by_layer[h]:
            kernel = kernels[(h, state)]
            following.update(
                int(next_state)
                for next_state, probability in zip(
                    kernel.next_states, kernel.probabilities
                )
                if probability > 0
            )
        reachable_by_layer[h - 1] = following
    query_groups = tuple(
        (h, state)
        for h in range(1, horizon + 1)
        for state in sorted(reachable_by_layer.get(h, set()))
    )
    return FiniteHorizonModel(
        name="inventory_untied_v1",
        horizon=horizon,
        initial_state=0,
        grid=np.arange(0.0, horizon + 0.25, 0.5),
        policy=policy,
        kernels=kernels,
        query_groups=query_groups,
        group_mode="untied_layer_state",
        metadata={
            "kind": "predeclared inventory instance (mechanical construction)",
            "instance_version": 1,
            "declaration": "mechanical construction; no search or outcome matching",
            "state": "on-hand inventory 0 through 6",
            "period_index": "season t = H-h, so h=8 is the first period",
            "base_stock_by_period": list(INVENTORY_BASE_STOCK),
            "poisson_mean_by_period": list(INVENTORY_POISSON_MEAN),
            "demand_support": "0,...,5 with Poisson tail P(D>=6) represented by censored value 6",
            "ordering": "order=max(0,target-state), capped at capacity; then demand; then disruption",
            "disruption_probability": "0.08 when post-order stock is 6, otherwise 0.06",
            "disruption_effect": "removes one remaining unit if available and lowers reward by one 0.5 category",
            "profit_before_quantization": "4*sales - 1.5*order - 0.5*ending_before_disruption - 2*lost",
            "quantization": "profit<=2 -> 0; 2<profit<8 -> 0.5; profit>=8 -> 1; disruption then subtracts 0.5 with floor 0",
            "reward_support": [0.0, 0.5, 1.0],
            "grid_step": 0.5,
            "query_rule": "all structurally reachable layer-state groups from root (8,0), including h=1",
            "reachable_states_by_h": {
                str(h): sorted(states_at_h)
                for h, states_at_h in sorted(reachable_by_layer.items())
            },
        },
    )


FROZEN_LAKE_8X8 = (
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
)


def _frozenlake_move(state: int, action: int, size: int = 8) -> int:
    row, column = divmod(state, size)
    if action == 0:  # left
        column = max(column - 1, 0)
    elif action == 1:  # down
        row = min(row + 1, size - 1)
    elif action == 2:  # right
        column = min(column + 1, size - 1)
    elif action == 3:  # up
        row = max(row - 1, 0)
    else:
        raise ValueError(f"invalid FrozenLake action {action}")
    return row * size + column


def frozenlake_8x8() -> FiniteHorizonModel:
    horizon, size, actions = 200, 8, 4
    states = size * size
    terminal = {
        r * size + c
        for r, row in enumerate(FROZEN_LAKE_8X8)
        for c, symbol in enumerate(row)
        if symbol in {"H", "G"}
    }
    goal = next(
        r * size + c
        for r, row in enumerate(FROZEN_LAKE_8X8)
        for c, symbol in enumerate(row)
        if symbol == "G"
    )
    kernels: dict[tuple[int, int], Kernel] = {}
    for state in range(states):
        for action in range(actions):
            if state in terminal:
                kernels[(state, action)] = _kernel([(0.0, state, 1.0)])
                continue
            realized = ((action - 1) % 4, action, (action + 1) % 4)
            outcomes = []
            for actual in realized:
                next_state = _frozenlake_move(state, actual, size)
                outcomes.append((float(next_state == goal), next_state, 1.0 / 3.0))
            kernels[(state, action)] = _kernel(outcomes)

    discount = 0.99
    value = np.zeros(states, dtype=float)
    selected = np.zeros(states, dtype=int)
    for _ in range(100_000):
        candidate = np.zeros((states, actions), dtype=float)
        for state in range(states):
            if state in terminal:
                continue
            for action in range(actions):
                kernel = kernels[(state, action)]
                candidate[state, action] = np.sum(
                    kernel.probabilities
                    * (kernel.rewards + discount * value[kernel.next_states])
                )
        updated = candidate.max(axis=1)
        updated[list(terminal)] = 0.0
        selected = np.argmax(candidate, axis=1)
        if np.max(np.abs(updated - value)) < 1e-14:
            value = updated
            break
        value = updated
    else:
        raise RuntimeError("FrozenLake planning value iteration did not converge")

    policy = np.zeros((horizon + 1, states, actions), dtype=float)
    policy[1:, np.arange(states), selected] = 1.0

    reachable = {0}
    frontier = {0}
    for _ in range(horizon):
        following: set[int] = set()
        for state in frontier:
            if state in terminal:
                continue
            kernel = kernels[(state, int(selected[state]))]
            following.update(
                int(next_state)
                for next_state, probability in zip(
                    kernel.next_states, kernel.probabilities
                )
                if probability > 0
            )
        new_states = following - reachable
        reachable.update(following)
        frontier = new_states
        if not frontier:
            break
    query_groups = tuple(
        (state, int(selected[state]))
        for state in sorted(reachable)
        if state not in terminal
    )
    return FiniteHorizonModel(
        name="frozenlake_8x8",
        horizon=horizon,
        initial_state=0,
        grid=np.array([0.0, 1.0, float(horizon)]),
        policy=policy,
        kernels=kernels,
        query_groups=query_groups,
        metadata={
            "public_environment": "Gymnasium FrozenLake-v1, map_name=8x8",
            "map": list(FROZEN_LAKE_8X8),
            "action_codes": {"left": 0, "down": 1, "right": 2, "up": 3},
            "slippery_probabilities": [1 / 3, 1 / 3, 1 / 3],
            "planning_discount": discount,
            "planning_tolerance": 1e-14,
            "tie_break": "smallest action index via numpy.argmax",
            "preferred_action_by_state": selected.tolist(),
            "terminal_kernels": "known and excluded from query groups",
        },
    )


# Gymnasium CliffWalking-v1 action order: up, right, down, left.
CLIFF_UP, CLIFF_RIGHT, CLIFF_DOWN, CLIFF_LEFT = range(4)

# These literal tables are part of the experiment specification. Rows are
# flattened row-major over the public 4-by-12 state numbering.
CLIFF_PRIMARY_ACTIONS = (
    *(CLIFF_DOWN for _ in range(12)),
    *(CLIFF_DOWN for _ in range(12)),
    *(CLIFF_RIGHT for _ in range(11)), CLIFF_DOWN,
    *(CLIFF_UP for _ in range(12)),
)
CLIFF_SAFE_ACTIONS = (
    *(CLIFF_DOWN for _ in range(12)),
    *(CLIFF_RIGHT for _ in range(11)), CLIFF_DOWN,
    *(CLIFF_UP for _ in range(12)),
    *(CLIFF_UP for _ in range(12)),
)


def _cliff_move(state: int, action: int) -> int:
    row, column = divmod(state, 12)
    drdc = ((-1, 0), (0, 1), (1, 0), (0, -1))
    dr, dc = drdc[action]
    row = min(3, max(0, row + dr))
    column = min(11, max(0, column + dc))
    return row * 12 + column


def _cliffwalking(policy_name: str) -> FiniteHorizonModel:
    horizon, states, actions = 20, 48, 4
    start, goal = 36, 47
    cliff = set(range(37, 47))
    preferred = np.asarray(
        CLIFF_PRIMARY_ACTIONS if policy_name == "primary" else CLIFF_SAFE_ACTIONS,
        dtype=int,
    )
    epsilon = 0.05
    policy_row = np.full((states, actions), epsilon / actions, dtype=float)
    policy_row[np.arange(states), preferred] += 1.0 - epsilon
    policy_row[goal] = 0.0
    policy_row[goal, CLIFF_UP] = 1.0
    policy = np.zeros((horizon + 1, states, actions), dtype=float)
    policy[1:] = policy_row

    kernels: dict[tuple[int, int], Kernel] = {}
    for state in range(states):
        for action in range(actions):
            if state == goal:
                kernels[(state, action)] = _kernel([(1.0, goal, 1.0)])
                continue
            outcomes: list[tuple[float, int, float]] = []
            for actual in ((action - 1) % 4, action, (action + 1) % 4):
                proposed = _cliff_move(state, actual)
                if proposed in cliff:
                    outcomes.append((0.0, start, 1.0 / 3.0))
                else:
                    outcomes.append((0.99, proposed, 1.0 / 3.0))
            kernels[(state, action)] = _kernel(outcomes)

    query_groups = tuple(
        [(state, action) for state in range(states) if state not in cliff | {goal}
         for action in range(actions)]
        + [(goal, CLIFF_UP)]
    )
    grid = np.asarray(
        sorted({0.99 * ordinary + absorbing
                for ordinary in range(horizon + 1)
                for absorbing in range(horizon + 1 - ordinary)}),
        dtype=float,
    )
    return FiniteHorizonModel(
        name=f"cliffwalking_{policy_name}",
        horizon=horizon,
        initial_state=start,
        grid=grid,
        policy=policy,
        kernels=kernels,
        query_groups=query_groups,
        metadata={
            "public_environment": "Gymnasium CliffWalking-v1, is_slippery=True",
            "shape": [4, 12],
            "start_state": start,
            "goal_state": goal,
            "cliff_states": sorted(cliff),
            "action_codes": {"up": 0, "right": 1, "down": 2, "left": 3},
            "slippery_realized_actions": "(a-1, a, a+1) modulo 4",
            "slippery_probabilities": [1 / 3, 1 / 3, 1 / 3],
            "reward_map": {"-100": 0.0, "-1": 0.99, "0_padding": 1.0},
            "policy_name": policy_name,
            "epsilon": epsilon,
            "epsilon_convention": "epsilon-greedy: 1-epsilon+epsilon/4 preferred",
            "preferred_action_by_state": preferred.tolist(),
            "policy_table_shape": [4, 12],
            "terminal_kernel": "known absorbing reward-1 kernel; one retained group",
        },
    )


def cliffwalking_primary() -> FiniteHorizonModel:
    return _cliffwalking("primary")


def cliffwalking_safe() -> FiniteHorizonModel:
    return _cliffwalking("safe")


TAXI_MAP = (
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
)
TAXI_LOCS = ((0, 0), (0, 4), (4, 0), (4, 3))


def taxi_encode(row: int, column: int, passenger: int, destination: int) -> int:
    return ((row * 5 + column) * 5 + passenger) * 4 + destination


def taxi_decode(state: int) -> tuple[int, int, int, int]:
    destination = state % 4
    state //= 4
    passenger = state % 5
    state //= 5
    column = state % 5
    row = state // 5
    return row, column, passenger, destination


def _taxi_dry_move(row: int, column: int, action: int) -> tuple[int, int]:
    if action == 0:
        return min(row + 1, 4), column
    if action == 1:
        return max(row - 1, 0), column
    if action == 2 and TAXI_MAP[1 + row][2 * column + 2] == ":":
        return row, min(column + 1, 4)
    if action == 3 and TAXI_MAP[1 + row][2 * column] == ":":
        return row, max(column - 1, 0)
    return row, column


def _taxi_distances(target: tuple[int, int]) -> dict[tuple[int, int], int]:
    distances = {target: 0}
    queue = deque([target])
    while queue:
        current = queue.popleft()
        for row in range(5):
            for column in range(5):
                source = (row, column)
                if source in distances:
                    continue
                if any(_taxi_dry_move(row, column, action) == current for action in range(4)):
                    distances[source] = distances[current] + 1
                    queue.append(source)
    if len(distances) != 25:
        raise RuntimeError("Taxi dry map is not connected")
    return distances


TAXI_DISTANCE = tuple(_taxi_distances(location) for location in TAXI_LOCS)


def _taxi_policy_action(full_state: int) -> int:
    row, column, passenger, destination = taxi_decode(full_state)
    target_index = destination if passenger == 4 else passenger
    target = TAXI_LOCS[target_index]
    if (row, column) == target:
        return 5 if passenger == 4 else 4
    distances = TAXI_DISTANCE[target_index]
    current_distance = distances[(row, column)]
    improving = [
        action
        for action in range(4)
        if _taxi_dry_move(row, column, action) != (row, column)
        and distances[_taxi_dry_move(row, column, action)] == current_distance - 1
    ]
    if not improving:
        raise RuntimeError(f"no dry shortest-path action for Taxi state {full_state}")
    return min(improving)


def _taxi_rainy_positions(row: int, column: int, action: int) -> list[tuple[float, int, int]]:
    intended = _taxi_dry_move(row, column, action)
    if intended == (row, column):
        return [(1.0, row, column)]
    relative = {
        0: (2, 3),  # south: lateral east/west
        1: (3, 2),  # north: lateral west/east
        2: (1, 0),  # east: lateral north/south
        3: (0, 1),  # west: lateral south/north
    }
    left_action, right_action = relative[action]
    left = _taxi_dry_move(row, column, left_action)
    right = _taxi_dry_move(row, column, right_action)
    return [
        (0.8, intended[0], intended[1]),
        (0.1, left[0], left[1]),
        (0.1, right[0], right[1]),
    ]


def _taxi_outcomes_full(full_state: int, action: int) -> list[tuple[float, int, float]]:
    row, column, passenger, destination = taxi_decode(full_state)
    if action <= 3:
        return [
            (-1.0, taxi_encode(new_row, new_column, passenger, destination), probability)
            for probability, new_row, new_column in _taxi_rainy_positions(row, column, action)
        ]
    if action == 4:
        legal = passenger < 4 and (row, column) == TAXI_LOCS[passenger]
        new_passenger = 4 if legal else passenger
        return [((-1.0 if legal else -10.0), taxi_encode(row, column, new_passenger, destination), 1.0)]
    if action == 5 and passenger == 4 and (row, column) == TAXI_LOCS[destination]:
        return [(20.0, taxi_encode(row, column, destination, destination), 1.0)]
    if action == 5 and passenger == 4 and (row, column) in TAXI_LOCS:
        new_passenger = TAXI_LOCS.index((row, column))
        return [(-1.0, taxi_encode(row, column, new_passenger, destination), 1.0)]
    return [(-10.0, full_state, 1.0)]


def _taxi_dry_completion(full_state: int) -> int:
    steps = 0
    while True:
        row, column, passenger, destination = taxi_decode(full_state)
        if passenger == destination and (row, column) == TAXI_LOCS[destination]:
            return steps
        action = _taxi_policy_action(full_state)
        if action <= 3:
            row, column = _taxi_dry_move(row, column, action)
            full_state = taxi_encode(row, column, passenger, destination)
        elif action == 4:
            full_state = taxi_encode(row, column, 4, destination)
        else:
            full_state = taxi_encode(row, column, destination, destination)
        steps += 1
        if steps > 100:
            raise RuntimeError("Taxi dry policy failed to terminate")


def rainy_taxi() -> FiniteHorizonModel:
    horizon, actions = 200, 6
    initial_states = [
        taxi_encode(row, column, passenger, destination)
        for row in range(5)
        for column in range(5)
        for passenger in range(4)
        for destination in range(4)
        if passenger != destination
    ]
    dry_lengths = {state: _taxi_dry_completion(state) for state in initial_states}
    maximum_dry = max(dry_lengths.values())
    maximizers = sorted(state for state, length in dry_lengths.items() if length == maximum_dry)
    root_full = maximizers[0]

    reachable = {root_full}
    queue = deque([root_full])
    terminal_full: set[int] = set()
    while queue:
        state = queue.popleft()
        row, column, passenger, destination = taxi_decode(state)
        if passenger == destination and (row, column) == TAXI_LOCS[destination]:
            terminal_full.add(state)
            continue
        action = _taxi_policy_action(state)
        for _, next_state, probability in _taxi_outcomes_full(state, action):
            if probability > 0 and next_state not in reachable:
                reachable.add(next_state)
                queue.append(next_state)

    full_states = tuple(sorted(reachable))
    compact = {state: index for index, state in enumerate(full_states)}
    states = len(full_states)
    policy_row = np.zeros((states, actions), dtype=float)
    kernels: dict[tuple[int, int], Kernel] = {}
    query_groups: list[tuple[int, int]] = []
    action_table: dict[str, int] = {}
    for full_state in full_states:
        state = compact[full_state]
        if full_state in terminal_full:
            action = 0
            policy_row[state, action] = 1.0
            kernels[(state, action)] = _kernel([(1.0 / 21.0, state, 1.0)])
            action_table[str(full_state)] = action
            continue
        action = _taxi_policy_action(full_state)
        policy_row[state, action] = 1.0
        action_table[str(full_state)] = action
        transformed = [
            ((raw_reward + 1.0) / 21.0, compact[next_full], probability)
            for raw_reward, next_full, probability in _taxi_outcomes_full(full_state, action)
        ]
        kernels[(state, action)] = _kernel(transformed)
        query_groups.append((state, action))

    policy = np.zeros((horizon + 1, states, actions), dtype=float)
    policy[1:] = policy_row
    active_grid = np.arange(221, dtype=float) / 21.0
    grid = np.concatenate((active_grid, [float(horizon)]))
    return FiniteHorizonModel(
        name="rainy_taxi",
        horizon=horizon,
        initial_state=compact[root_full],
        grid=grid,
        policy=policy,
        kernels=kernels,
        query_groups=tuple(query_groups),
        metadata={
            "public_environment": "Gymnasium Taxi-v4, is_rainy=True, fickle_passenger=False",
            "map": list(TAXI_MAP),
            "locations": [list(location) for location in TAXI_LOCS],
            "action_codes": {
                "south": 0, "north": 1, "east": 2, "west": 3,
                "pickup": 4, "dropoff": 5,
            },
            "rainy_probabilities": [0.8, 0.1, 0.1],
            "blocked_intended_move": "suppresses lateral drift",
            "root_official_state": root_full,
            "root_decoded": list(taxi_decode(root_full)),
            "maximum_dry_completion_steps": maximum_dry,
            "maximum_dry_initial_states": maximizers,
            "compact_to_official_state": list(full_states),
            "preferred_action_by_official_state": action_table,
            "reward_map": "(raw_reward + 1) / 21, including zero-reward padding",
            "terminal_kernels": "known and excluded from query groups",
        },
    )


ENVIRONMENTS = {
    "controlled_shared": controlled_shared,
    "controlled_untied_v1": controlled_untied_v1,
    "inventory_untied_v1": inventory_untied_v1,
    "frozenlake_8x8": frozenlake_8x8,
    "cliffwalking_primary": cliffwalking_primary,
    "cliffwalking_safe": cliffwalking_safe,
    "rainy_taxi": rainy_taxi,
}


def make_environment(name: str) -> FiniteHorizonModel:
    try:
        return ENVIRONMENTS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown environment {name!r}") from exc
