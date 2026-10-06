#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Point-reference route-and-self-loop contraction with physical-time feedback.

The empirical measure, objective, and gradient use exact residence time. The
contraction reference is not the previous block measure: it is the Dirac
occupation of the agent's current cell at the start of the iteration. A
continuous-position A* planner uses actual first-order transition times and the
source/target residence split as edge costs. After reaching the destination,
the agent executes one stationary self-loop (u = 0) for the exact duration
required to contract relative to the starting-cell point measure.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ==================================================
# Parameters
# ==================================================
rows = 16
cols = 16
number_of_nodes = rows * cols

rho = 1 - 1e-7
number_of_blocks = 40_000
maximum_horizon = 500
random_seed = 13

# Destination must be strictly better than the final block threshold.
# This slack pays for transit error.
rho_endpoint = 0.9

# Destination score:
#   distance_weight * normalized graph distance
# + endpoint_weight * normalized endpoint oracle error.
distance_weight = 6.0
endpoint_weight = 2.0

# A* edge cost:
#   travel_edge_weight
# + path_error_weight * normalized positive excess error.
travel_edge_weight = 1.0
path_error_weight = 4.0
astar_heuristic_weight = 1.
# Aggressive self-loop release rule.
self_loop_near_min_fraction = 0.35 ## Condition for aggressive self-loop: point-reference error <= fraction * warm_gap
self_loop_minimum_distance = 15 ## Condition for aggressive self-loop: nearest minimizer distance >= minimum_distance
long_block_warning_time = 5.0
maximum_predicted_self_loop_time = 0.1
route_improvement_tolerance = 1e-12

objective_tolerance = 1e-15
warm_gap_tolerance = 1e-12
contraction_tolerance = 1e-12

# Online dynamic execution. These values do not feed back into the discrete
# optimizer. They only determine the continuous trajectory and trajectory G.
u_max = 0.35
self_loop_time = 0.1
entry_fraction = 0.02
patrol_margin_fraction = 0.08
cell_width = 1.0 / cols
cell_height = 1.0 / rows


# ==================================================
# Rectangular grid graph with self-loops
# ==================================================
def node_index(row: int, col: int) -> int:
    return row * cols + col


def node_coordinates(node: int) -> Tuple[int, int]:
    return divmod(node, cols)


neighbors: List[List[int]] = []
for row in range(rows):
    for col in range(cols):
        node = node_index(row, col)
        node_neighbors = [node]
        if row > 0:
            node_neighbors.append(node_index(row - 1, col))
        if row < rows - 1:
            node_neighbors.append(node_index(row + 1, col))
        if col > 0:
            node_neighbors.append(node_index(row, col - 1))
        if col < cols - 1:
            node_neighbors.append(node_index(row, col + 1))
        neighbors.append(node_neighbors)


def cell_bounds(node: int) -> Tuple[float, float, float, float]:
    row, col = node_coordinates(node)
    return (
        col * cell_width,
        (col + 1) * cell_width,
        row * cell_height,
        (row + 1) * cell_height,
    )


def entry_waypoint(position: np.ndarray, current: int, nxt: int) -> np.ndarray:
    """Return a point slightly inside an adjacent next cell."""
    r0, c0 = node_coordinates(current)
    r1, c1 = node_coordinates(nxt)
    eps = entry_fraction * min(cell_width, cell_height)
    waypoint = np.asarray(position, dtype=float).copy()
    if c1 == c0 + 1:
        waypoint[0] = (c0 + 1) * cell_width + eps
        waypoint[1] = np.clip(waypoint[1], r1 * cell_height + eps, (r1 + 1) * cell_height - eps)
    elif c1 == c0 - 1:
        waypoint[0] = c0 * cell_width - eps
        waypoint[1] = np.clip(waypoint[1], r1 * cell_height + eps, (r1 + 1) * cell_height - eps)
    elif r1 == r0 + 1:
        waypoint[1] = (r0 + 1) * cell_height + eps
        waypoint[0] = np.clip(waypoint[0], c1 * cell_width + eps, (c1 + 1) * cell_width - eps)
    elif r1 == r0 - 1:
        waypoint[1] = r0 * cell_height - eps
        waypoint[0] = np.clip(waypoint[0], c1 * cell_width + eps, (c1 + 1) * cell_width - eps)
    else:
        raise ValueError(f"Nonadjacent move {current}->{nxt}")
    return waypoint


def transition_residence(position, current, nxt, speed):
    """Execute one transition and return source and target residence times."""
    waypoint = entry_waypoint(position, current, nxt)
    displacement = waypoint - np.asarray(position, dtype=float)
    length = float(np.linalg.norm(displacement))
    if length <= 1e-15:
        return waypoint, 0.0, 0.0
    r0, c0 = node_coordinates(current)
    r1, c1 = node_coordinates(nxt)
    if c1 != c0:
        boundary = max(c0, c1) * cell_width
        alpha = (boundary - float(position[0])) / float(displacement[0])
    else:
        boundary = max(r0, r1) * cell_height
        alpha = (boundary - float(position[1])) / float(displacement[1])
    alpha = float(np.clip(alpha, 0.0, 1.0))
    duration = length / speed
    return waypoint, alpha * duration, (1.0 - alpha) * duration


def uniform_patrol(position, cell, duration, speed, rng):
    """Realize one holding sample as an exact-duration in-cell patrol."""
    if duration <= 0.0:
        return np.asarray(position, dtype=float).copy(), []
    xmin, xmax, ymin, ymax = cell_bounds(cell)
    margin = patrol_margin_fraction * min(cell_width, cell_height)
    low = np.array([xmin + margin, ymin + margin])
    high = np.array([xmax - margin, ymax - margin])
    point = np.minimum(np.maximum(np.asarray(position, dtype=float), low), high)
    remaining = float(duration)
    points = []
    while remaining > contraction_tolerance:
        target_point = rng.uniform(low, high)
        displacement = target_point - point
        length = float(np.linalg.norm(displacement))
        if length <= 1e-15:
            continue
        travel_time = length / speed
        if travel_time <= remaining + contraction_tolerance:
            point = target_point
            remaining -= travel_time
        else:
            point = point + (remaining / travel_time) * displacement
            remaining = 0.0
        points.append(point.copy())
    return point, points


def execute_block_online(position, current, block, speed, hold_time, rng):
    """Execute one generated block and return exact physical residence times."""
    residence = np.zeros(number_of_nodes, dtype=float)
    points = []
    travel_time = 0.0
    patrol_time = 0.0
    position = np.asarray(position, dtype=float).copy()
    current = int(current)
    for sample in block:
        sample = int(sample)
        if sample == current:
            position, patrol_points = uniform_patrol(position, current, hold_time, speed, rng)
            residence[current] += hold_time
            patrol_time += hold_time
            points.extend(patrol_points)
        else:
            waypoint, source_time, target_time = transition_residence(
                position, current, sample, speed
            )
            residence[current] += source_time
            residence[sample] += target_time
            travel_time += source_time + target_time
            position = waypoint
            current = sample
            points.append(position.copy())
    return position, current, points, residence, travel_time, patrol_time


def objective(mu: np.ndarray) -> float:
    return float(np.sum((mu - target) ** 2))


def first_variation(mu: np.ndarray) -> np.ndarray:
    return 2.0 * (mu - target)


# ==================================================
# Nonuniform target distribution
# ==================================================
centroids = np.zeros((number_of_nodes, 2), dtype=float)
for node in range(number_of_nodes):
    row, col = divmod(node, cols)
    centroids[node, 0] = (col + 0.5) / cols
    centroids[node, 1] = (row + 0.5) / rows


def gaussian_density(points, mean, covariance):
    difference = points - mean
    inverse_covariance = np.linalg.inv(covariance)
    exponent = np.einsum(
        "ni,ij,nj->n", difference, inverse_covariance, difference
    )
    return np.exp(-0.5 * exponent)


density_1 = gaussian_density(
    centroids,
    np.array([0.25, 0.75]),
    np.array([[0.012, 0.0], [0.0, 0.018]]),
)
density_2 = gaussian_density(
    centroids,
    np.array([0.75, 0.30]),
    np.array([[0.020, 0.008], [0.008, 0.015]]),
)
density_3 = gaussian_density(
    centroids,
    np.array([0.70, 0.80]),
    np.array([[0.008, 0.0], [0.0, 0.008]]),
)

target = 0.50 * density_1 + 0.35 * density_2 + 0.15 * density_3
background_weight = 0.02
target = (1.0 - background_weight) * target + background_weight
target /= np.sum(target)

assert np.all(target >= 0.0)
assert np.isclose(np.sum(target), 1.0)


@dataclass
class DestinationChoice:
    node: int
    distance: int
    endpoint_error: float
    normalized_endpoint_error: float
    score: float


@dataclass
class BlockRecord:
    block: int
    mode: str
    total_samples: int
    block_length: int
    travel_length: int
    holding_length: int
    destination: int
    destination_row: int
    destination_col: int
    objective_before: float
    objective_after: float
    history_gap: float
    previous_block_error: float
    new_block_error: float
    contraction_ratio: float
    contracted: bool
    gamma: float
    astar_cost: float
    physical_time: float
    physical_block_time: float
    physical_travel_time: float
    physical_patrol_time: float
    discrete_objective: float
    trajectory_objective: float
    point_reference_error: float
    required_self_loop_time: float
    physical_contracted: bool
    aggressive_self_loop: bool
    planned_goal: int
    current_to_goal_distance: int
    current_error: float
    goal_error: float
    oracle_error_difference: float
    objective_change: float
    route_average_error: float
    route_rejected_for_hold: bool


def manhattan_distance(a: int, b: int) -> int:
    ar, ac = node_coordinates(a)
    br, bc = node_coordinates(b)
    return abs(ar - br) + abs(ac - bc)


def block_measure(block: Sequence[int]) -> np.ndarray:
    if not block:
        raise ValueError("A block must contain at least one sample.")
    counts = np.bincount(np.asarray(block, dtype=np.int64), minlength=number_of_nodes)
    return counts.astype(float) / len(block)


def choose_destination(
    current: int,
    g: np.ndarray,
    warm_gap: float,
    endpoint_rho: float,
) -> DestinationChoice:
    """Choose a nearby endpoint with error <= endpoint_rho * warm_gap."""
    s = float(np.min(g))
    errors = g - s
    feasible = errors <= endpoint_rho * warm_gap + contraction_tolerance
    candidates = np.flatnonzero(feasible)

    if candidates.size == 0:
        candidates = np.flatnonzero(np.isclose(g, s, atol=1e-15, rtol=0.0))

    current_row, current_col = node_coordinates(current)
    candidate_rows = candidates // cols
    candidate_cols = candidates % cols
    distances = (
        np.abs(candidate_rows - current_row)
        + np.abs(candidate_cols - current_col)
    ).astype(float)

    max_distance = max(1.0, float(rows + cols - 2))
    normalized_distances = distances / max_distance
    normalized_errors = errors[candidates] / max(warm_gap, warm_gap_tolerance)

    scores = (
        distance_weight * normalized_distances
        + endpoint_weight * normalized_errors
    )
    pos = int(np.argmin(scores))
    node = int(candidates[pos])

    return DestinationChoice(
        node=node,
        distance=int(distances[pos]),
        endpoint_error=float(errors[node]),
        normalized_endpoint_error=float(normalized_errors[pos]),
        score=float(scores[pos]),
    )


def astar_variational_path(
    start: int,
    goal: int,
    g: np.ndarray,
    position: np.ndarray,
    speed: float,
) -> Tuple[List[int], float]:
    """A* using actual transition time and residence-weighted oracle error.

    A state is (cell, incoming_cell). Its label stores the predicted continuous
    entry position. For u -> v, the cost is

        travel_edge_weight * (t_u + t_v)
        + path_error_weight * [t_u (g[u]-g[goal])_+
                               + t_v (g[v]-g[goal])_+] / scale.
    """
    if start == goal:
        return [start], 0.0
    if speed <= 0.0:
        raise ValueError("speed must be positive")

    goal_value = float(g[goal])
    scale = max(float(np.max(g) - np.min(g)), 1e-15)
    nominal_cell_time = min(cell_width, cell_height) / speed

    def heuristic(cell: int) -> float:
        # Lower bound for the explicit nonnegative travel-time part.
        return (
            astar_heuristic_weight
            * travel_edge_weight
            * manhattan_distance(cell, goal)
            * nominal_cell_time
        )

    start_state = (int(start), -1)
    best = {start_state: 0.0}
    predecessor = {}
    state_position = {start_state: np.asarray(position, dtype=float).copy()}
    queue = [(heuristic(start), 0.0, start_state)]
    goal_state = None

    while queue:
        _, cost, state = heapq.heappop(queue)
        current, _ = state
        if cost > best.get(state, np.inf) + contraction_tolerance:
            continue
        if current == goal:
            goal_state = state
            break

        for nxt in neighbors[current]:
            if nxt == current:
                continue
            nxt = int(nxt)
            next_state = (nxt, current)
            next_position, source_time, target_time = transition_residence(
                state_position[state], current, nxt, speed
            )
            transition_time = source_time + target_time
            error_mass = (
                source_time * max(0.0, float(g[current]) - goal_value)
                + target_time * max(0.0, float(g[nxt]) - goal_value)
            )
            edge_cost = (
                travel_edge_weight * transition_time
                + path_error_weight * error_mass / scale
            )
            candidate = cost + edge_cost
            if candidate < best.get(next_state, np.inf) - contraction_tolerance:
                best[next_state] = candidate
                predecessor[next_state] = state
                state_position[next_state] = next_position
                heapq.heappush(
                    queue,
                    (candidate + heuristic(nxt), candidate, next_state),
                )

    if goal_state is None:
        raise RuntimeError(f"No path from {start} to {goal}.")

    states = [goal_state]
    while states[-1] != start_state:
        states.append(predecessor[states[-1]])
    states.reverse()
    return [state[0] for state in states], float(best[goal_state])


def build_block_to_block_contraction(
    path: Sequence[int],
    destination: int,
    g: np.ndarray,
    s: float,
    warm_gap: float,
    contraction_rho: float,
    horizon_cap: Optional[int],
) -> Tuple[List[int], float, bool, int, int]:
    """Follow route and hold until h_k(p_new) <= rho * h_k(p_previous)."""
    route_samples = list(path[1:])
    travel_length = len(route_samples)

    if not route_samples:
        route_samples = [destination]  # one self-loop sample

    if horizon_cap is not None and len(route_samples) > horizon_cap:
        route_samples = route_samples[:horizon_cap]

    block = list(route_samples)
    total_g = float(np.sum(g[np.asarray(block, dtype=np.int64)]))

    def error() -> float:
        return total_g / len(block) - s

    threshold = contraction_rho * warm_gap
    contracted = error() <= threshold + contraction_tolerance

    while not contracted:
        if horizon_cap is not None and len(block) >= horizon_cap:
            break
        block.append(destination)
        total_g += float(g[destination])
        contracted = error() <= threshold + contraction_tolerance

    holding_length = max(0, len(block) - len(route_samples))
    return block, float(error()), bool(contracted), travel_length, holding_length


def run_algorithm(
    blocks: int,
    max_horizon: Optional[int],
    contraction_rho: float,
    endpoint_rho: float,
    seed: int,
    initial_state: Optional[int],
    verbose_every: int,
    near_min_fraction: float,
    minimum_goal_distance: int,
    long_block_time: float,
    max_predicted_self_loop_time: float,
    route_better_tolerance: float,
):
    if not (0.0 <= endpoint_rho < contraction_rho < 1.0):
        raise ValueError("Require 0 <= rho_endpoint < rho < 1.")
    if not (0.0 <= near_min_fraction <= 1.0):
        raise ValueError("near_min_fraction must lie in [0, 1].")
    if minimum_goal_distance < 0:
        raise ValueError("minimum_goal_distance must be nonnegative.")
    if long_block_time <= 0.0:
        raise ValueError("long_block_time must be positive.")
    if max_predicted_self_loop_time <= 0.0:
        raise ValueError("max_predicted_self_loop_time must be positive.")
    if route_better_tolerance < 0.0:
        raise ValueError("route_better_tolerance must be nonnegative.")

    rng = np.random.default_rng(seed)
    if initial_state is None:
        initial_state = int(rng.integers(number_of_nodes))

    counts = np.zeros(number_of_nodes, dtype=np.int64)
    counts[initial_state] = 1
    total_samples = 1
    current = initial_state
    trajectory = [initial_state]

    # Separate online dynamic trajectory and physical residence-time measure.
    dynamic_position = centroids[initial_state].copy()
    dynamic_trajectory = [dynamic_position.copy()]
    physical_occupation = np.zeros(number_of_nodes, dtype=float)
    physical_occupation[initial_state] = self_loop_time
    physical_time = self_loop_time
    physical_time_history = [physical_time]
    trajectory_objective_history = [objective(physical_occupation / physical_time)]

    # No previous-block measure is used. The reference at each step is the
    # Dirac occupation of the current/last-position cell.
    previous_block = [initial_state]
    previous_block_measure = None

    records: List[BlockRecord] = []
    sample_history = [total_samples]
    objective_history = [objective(physical_occupation / physical_time)]

    for block_index in range(blocks):
        eta = physical_occupation / physical_time
        objective_before = objective(eta)
        if objective_before <= objective_tolerance:
            break

        g = first_variation(eta)
        s = float(np.min(g))
        history_gap = float(np.dot(g, eta) - s)
        if history_gap <= warm_gap_tolerance:
            break

        # Point reference: the occupation of the current/last position cell.
        # Its frozen-gradient error is simply g[current] - s.
        point_reference_error = max(0.0, float(g[current] - s))
        warm_gap = point_reference_error

        minimizers = np.flatnonzero(
            np.isclose(g, s, atol=contraction_tolerance, rtol=0.0)
        )
        if minimizers.size == 0:
            minimizers = np.array([int(np.argmin(g))], dtype=np.int64)
        minimizer_distances = np.asarray(
            [manhattan_distance(current, int(node)) for node in minimizers],
            dtype=np.int64,
        )
        nearest_pos = int(np.argmin(minimizer_distances))
        nearest_minimizer = int(minimizers[nearest_pos])
        nearest_minimizer_distance = int(minimizer_distances[nearest_pos])
        near_minimum_threshold = near_min_fraction * max(
            history_gap, warm_gap_tolerance
        )
        aggressive_self_loop = (
            point_reference_error <= near_minimum_threshold + contraction_tolerance
            and nearest_minimizer_distance >= minimum_goal_distance
        )
        start_cell = current
        route_rejected_for_hold = False
        route_average_error = point_reference_error

        if aggressive_self_loop:
            # The current point is already sufficiently good, while reaching a
            # true minimizer would require a comparatively long route. Perform
            # a stationary release self-loop and refresh the physical gradient.
            planned_goal = nearest_minimizer
            destination = current
            path = [current]
            astar_cost = 0.0
            physical_route = []
            physical_block_residence = np.zeros(number_of_nodes, dtype=float)
            required_self_loop_time = self_loop_time
            physical_travel_time = 0.0
            dynamic_points = []
            dynamic_current = current
            mode = "aggressive_self_loop_near_minimum"
        elif warm_gap <= warm_gap_tolerance:
            planned_goal = current
            destination = current
            path = [current]
            astar_cost = 0.0
            physical_route = []
            physical_block_residence = np.zeros(number_of_nodes, dtype=float)
            required_self_loop_time = self_loop_time
            physical_travel_time = 0.0
            dynamic_points = []
            dynamic_current = current
            mode = "release_self_loop_zero_point_gap"
        else:
            choice = choose_destination(
                current=current,
                g=g,
                warm_gap=warm_gap,
                endpoint_rho=endpoint_rho,
            )
            destination = choice.node
            planned_goal = destination
            path, astar_cost = astar_variational_path(
                current, destination, g, dynamic_position, u_max
            )
            physical_route = list(path[1:])
            if max_horizon is not None and len(physical_route) > max_horizon:
                raise RuntimeError(
                    "The physical A* path exceeds --max-horizon; truncating it "
                    "would not reach the certified destination."
                )

            block_start_position = dynamic_position.copy()
            if physical_route:
                (
                    dynamic_position,
                    dynamic_current,
                    dynamic_points,
                    physical_block_residence,
                    physical_travel_time,
                    _unused_patrol_time,
                ) = execute_block_online(
                    dynamic_position,
                    current,
                    physical_route,
                    u_max,
                    self_loop_time,
                    rng,
                )
            else:
                dynamic_current = current
                dynamic_points = []
                physical_block_residence = np.zeros(number_of_nodes, dtype=float)
                physical_travel_time = 0.0

            if dynamic_current != destination:
                raise RuntimeError("Physical route ended in the wrong cell.")

            route_time = float(physical_block_residence.sum())
            threshold = contraction_rho * point_reference_error
            destination_error = float(g[destination] - s)
            route_error_mass = float(np.dot(g - s, physical_block_residence))
            excess_mass = route_error_mass - threshold * route_time

            if excess_mass <= contraction_tolerance:
                required_self_loop_time = 0.0
            else:
                denominator = threshold - destination_error
                if denominator <= contraction_tolerance:
                    raise RuntimeError(
                        "No finite destination self-loop can contract relative "
                        "to the current-cell point reference."
                    )
                required_self_loop_time = excess_mass / denominator
                required_self_loop_time *= 1.0 + 32.0 * np.finfo(float).eps

            if route_time <= contraction_tolerance:
                required_self_loop_time = max(
                    required_self_loop_time, self_loop_time
                )

            route_average_error = (
                route_error_mass / route_time
                if route_time > contraction_tolerance
                else destination_error
            )
            route_is_strictly_better = (
                route_average_error
                < point_reference_error - route_better_tolerance
            )
            route_rejected_for_hold = (
                not route_is_strictly_better
                or not np.isfinite(required_self_loop_time)
                or required_self_loop_time
                > max_predicted_self_loop_time + contraction_tolerance
            )

            if route_rejected_for_hold:
                # Reject the proposed route before it affects the physical
                # trajectory. Stay at the current point for one short release
                # self-loop, update eta, and recompute the gradient.
                dynamic_position = block_start_position
                dynamic_current = current
                dynamic_points = []
                physical_route = []
                destination = current
                physical_block_residence = np.zeros(
                    number_of_nodes, dtype=float
                )
                physical_travel_time = 0.0
                required_self_loop_time = min(
                    self_loop_time, max_predicted_self_loop_time
                )
                aggressive_self_loop = True
                mode = "release_self_loop_rejected_route"
            else:
                mode = "point_reference_contract"

        # Execute the self-loop as a stationary dwell: u = 0.
        physical_block_residence[destination] += required_self_loop_time
        physical_patrol_time = required_self_loop_time
        physical_block_time = float(physical_block_residence.sum())
        if physical_block_time <= contraction_tolerance:
            raise RuntimeError("Physical block has zero duration.")

        new_block_measure = physical_block_residence / physical_block_time
        new_block_error = float(np.dot(g, new_block_measure) - s)
        physical_contracted = (
            new_block_error
            <= contraction_rho * point_reference_error + contraction_tolerance
            if point_reference_error > warm_gap_tolerance
            else new_block_error <= contraction_tolerance
        )
        if (
            point_reference_error > warm_gap_tolerance
            and not physical_contracted
            and not aggressive_self_loop
        ):
            raise RuntimeError(
                "Point-reference physical contraction failed: "
                f"error={new_block_error:.16e}, "
                f"threshold={contraction_rho * point_reference_error:.16e}."
            )

        # Update the optimizer itself with actual physical residence time.
        physical_occupation += physical_block_residence
        physical_time += physical_block_time
        dynamic_trajectory.extend(dynamic_points)
        eta_after = physical_occupation / physical_time
        objective_after = objective(eta_after)
        trajectory_objective_after = objective_after
        objective_change = objective_after - objective_before
        current_to_goal_distance = manhattan_distance(start_cell, planned_goal)
        current_error = float(g[start_cell] - s)
        goal_error = float(g[planned_goal] - s)
        oracle_error_difference = current_error - goal_error

        if physical_block_time > long_block_time:
            print(
                "LONG BLOCK "
                f"block={block_index + 1} "
                f"time={physical_block_time:.6f}s "
                f"route_time={physical_travel_time:.6f}s "
                f"self_loop={required_self_loop_time:.6f}s "
                f"distance={current_to_goal_distance} "
                f"current_error={current_error:.6e} "
                f"goal_error={goal_error:.6e} "
                f"oracle_difference={oracle_error_difference:.6e} "
                f"route_average_error={route_average_error:.6e} "
                f"objective_change={objective_change:+.6e} "
                f"mode={mode}",
                flush=True,
            )

        # Retain a compact discrete path only as a diagnostic output.
        block = list(physical_route) + [destination]
        nodes = np.asarray(block, dtype=np.int64)
        np.add.at(counts, nodes, 1)
        trajectory.extend(block)
        total_samples += len(block)
        current = destination
        travel_length = len(physical_route)
        holding_length = 1 if required_self_loop_time > 0.0 else 0
        contracted = physical_contracted
        gamma = physical_block_time / physical_time
        contraction_ratio = (
            new_block_error / point_reference_error
            if point_reference_error > warm_gap_tolerance
            else np.nan
        )

        row, col = node_coordinates(destination)

        records.append(
            BlockRecord(
                block=block_index,
                mode=mode,
                total_samples=total_samples,
                block_length=len(block),
                travel_length=travel_length,
                holding_length=holding_length,
                destination=destination,
                destination_row=row,
                destination_col=col,
                objective_before=objective_before,
                objective_after=objective_after,
                history_gap=history_gap,
                previous_block_error=warm_gap,
                new_block_error=new_block_error,
                contraction_ratio=contraction_ratio,
                contracted=contracted,
                gamma=gamma,
                astar_cost=astar_cost,
                physical_time=physical_time,
                physical_block_time=physical_block_time,
                physical_travel_time=physical_travel_time,
                physical_patrol_time=physical_patrol_time,
                discrete_objective=objective(counts / total_samples),
                trajectory_objective=trajectory_objective_after,
                point_reference_error=point_reference_error,
                required_self_loop_time=required_self_loop_time,
                physical_contracted=physical_contracted,
                aggressive_self_loop=aggressive_self_loop,
                planned_goal=planned_goal,
                current_to_goal_distance=current_to_goal_distance,
                current_error=current_error,
                goal_error=goal_error,
                oracle_error_difference=oracle_error_difference,
                objective_change=objective_change,
                route_average_error=route_average_error,
                route_rejected_for_hold=route_rejected_for_hold,
            )
        )

        previous_block = block
        # The next reference is again the Dirac mass of the new current cell,
        # not this complete block measure.
        previous_block_measure = None
        sample_history.append(total_samples)
        objective_history.append(objective_after)
        physical_time_history.append(physical_time)
        trajectory_objective_history.append(trajectory_objective_after)

        if verbose_every > 0 and (block_index + 1) % verbose_every == 0:
            strict_so_far = [
                r for r in records
                if not r.aggressive_self_loop
                and "zero_point_gap" not in r.mode
            ]
            failures = sum(not r.contracted for r in strict_so_far)
            holds = sum("zero_point_gap" in r.mode for r in records)
            aggressive_releases = sum(r.aggressive_self_loop for r in records)
            print(
                f"block={block_index + 1:7d} "
                f"T={physical_time:12.6f} "
                f"G_physical={objective_after:.6e} "
                f"G_count={objective(counts / total_samples):.6e} "
                f"N={len(block):4d} ratio={contraction_ratio!s:>10} "
                f"holds={holds} aggressive_releases={aggressive_releases} "
                f"failures={failures}"
            )

    return (
        counts / total_samples,
        trajectory,
        records,
        np.asarray(sample_history, dtype=np.int64),
        np.asarray(objective_history, dtype=float),
        np.asarray(dynamic_trajectory, dtype=float),
        np.asarray(physical_time_history, dtype=float),
        np.asarray(trajectory_objective_history, dtype=float),
        physical_occupation / physical_time,
    )


def save_records(records: Sequence[BlockRecord], path: Path) -> None:
    if not records:
        return
    fields = list(records[0].__dict__.keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def save_plots(
    output_dir: Path,
    sample_history: np.ndarray,
    objective_history: np.ndarray,
    records: Sequence[BlockRecord],
    final_eta: np.ndarray,
    contraction_rho: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(sample_history, objective_history, label=r"$G(\eta_T)$")
    ax.set_xlabel("Total trajectory samples T")
    ax.set_ylabel(r"Objective $G(\eta_T)$")
    ax.set_title("Block-to-block route-and-hold objective history")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "objective_history.png", dpi=180)
    plt.close(fig)

    if records:
        indices = np.asarray([r.block for r in records])
        lengths = np.asarray([r.block_length for r in records])
        ratios = np.asarray([r.contraction_ratio for r in records], dtype=float)

        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        axes[0].plot(indices, lengths, linewidth=0.8)
        axes[0].set_ylabel("Block length")
        axes[0].grid(True, alpha=0.3)

        finite = np.isfinite(ratios)
        axes[1].plot(indices[finite], ratios[finite], linewidth=0.8)
        axes[1].axhline(contraction_rho, color="tab:red", linestyle="--", label="target rho")
        axes[1].set_xlabel("Block index")
        axes[1].set_ylabel("New/previous block error")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output_dir / "block_diagnostics.png", dpi=180)
        plt.close(fig)

    target_image = target.reshape(rows, cols)
    eta_image = final_eta.reshape(rows, cols)
    residual = eta_image - target_image
    limit = max(abs(float(residual.min())), abs(float(residual.max())), 1e-15)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    im0 = axes[0].imshow(target_image, origin="lower", cmap="viridis")
    axes[0].set_title("Target distribution")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(eta_image, origin="lower", cmap="viridis")
    axes[1].set_title("Final empirical distribution")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(residual, origin="lower", cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[2].set_title("Empirical minus target")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / "distribution_comparison.png", dpi=180)
    plt.close(fig)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Block-to-block route-and-hold contraction on a grid."
    )
    parser.add_argument("--blocks", type=int, default=number_of_blocks)
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=maximum_horizon,
        help="Use 0 for no cap.",
    )
    parser.add_argument("--rho", type=float, default=rho)
    parser.add_argument("--rho-endpoint", type=float, default=rho_endpoint)
    parser.add_argument("--seed", type=int, default=random_seed)
    parser.add_argument(
        "--self-loop-near-min-fraction",
        type=float,
        default=self_loop_near_min_fraction,
        help=(
            "Use an immediate stationary self-loop when current_error is at most "
            "this fraction of the global history gap and the nearest minimizer "
            "is sufficiently far away."
        ),
    )
    parser.add_argument(
        "--self-loop-min-distance",
        type=int,
        default=self_loop_minimum_distance,
        help="Minimum Manhattan distance to the nearest minimizer for aggressive self-loop release.",
    )
    parser.add_argument(
        "--long-block-warning-time",
        type=float,
        default=long_block_warning_time,
        help="Print detailed diagnostics for blocks longer than this many seconds.",
    )
    parser.add_argument(
        "--max-predicted-self-loop-time",
        type=float,
        default=maximum_predicted_self_loop_time,
        help=(
            "Reject a proposed route before execution when its predicted "
            "destination self-loop exceeds this duration."
        ),
    )
    parser.add_argument(
        "--route-improvement-tolerance",
        type=float,
        default=route_improvement_tolerance,
        help=(
            "A route is accepted only if its average frozen-gradient error is "
            "below the current-point error by at least this tolerance."
        ),
    )
    parser.add_argument("--initial-state", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("waypoint_point_reference_physical_astar_output"),
    )
    parser.add_argument("--verbose-every", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.initial_state is not None and not (0 <= args.initial_state < number_of_nodes):
        raise ValueError("Initial state is outside the graph.")

    horizon_cap = None if args.max_horizon == 0 else args.max_horizon
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        final_eta,
        trajectory,
        records,
        sample_history,
        objective_history,
        dynamic_trajectory,
        physical_time_history,
        trajectory_objective_history,
        final_trajectory_eta,
    ) = run_algorithm(
        blocks=args.blocks,
        max_horizon=horizon_cap,
        contraction_rho=args.rho,
        endpoint_rho=args.rho_endpoint,
        seed=args.seed,
        initial_state=args.initial_state,
        verbose_every=args.verbose_every,
        near_min_fraction=args.self_loop_near_min_fraction,
        minimum_goal_distance=args.self_loop_min_distance,
        long_block_time=args.long_block_warning_time,
        max_predicted_self_loop_time=args.max_predicted_self_loop_time,
        route_better_tolerance=args.route_improvement_tolerance,
    )

    save_records(records, args.output_dir / "block_history.csv")
    np.save(args.output_dir / "trajectory.npy", np.asarray(trajectory, dtype=np.int32))
    np.save(args.output_dir / "final_empirical_measure.npy", final_eta)
    np.save(args.output_dir / "target.npy", target)
    np.save(args.output_dir / "sample_history.npy", sample_history)
    np.save(args.output_dir / "objective_history.npy", objective_history)
    np.save(args.output_dir / "dynamic_trajectory_positions.npy", dynamic_trajectory)
    np.save(args.output_dir / "physical_time_history.npy", physical_time_history)
    np.save(args.output_dir / "trajectory_objective_history.npy", trajectory_objective_history)
    np.save(args.output_dir / "final_trajectory_empirical_measure.npy", final_trajectory_eta)
    save_plots(
        args.output_dir,
        sample_history,
        objective_history,
        records,
        final_eta,
        args.rho,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(physical_time_history, objective_history, label="Discrete G")
    ax.loglog(physical_time_history, trajectory_objective_history, label="Trajectory G")
    ax.set_xlabel("Elapsed physical time T")
    ax.set_ylabel(r"Objective $G(\eta)$")
    ax.set_title("Discrete and physical trajectory objectives")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "discrete_vs_trajectory_objective.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(target.reshape(rows, cols), origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
    if len(dynamic_trajectory) > 1:
        ax.plot(dynamic_trajectory[:, 0], dynamic_trajectory[:, 1], color="white", lw=0.45, alpha=0.75)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", title="Online dynamic execution")
    fig.colorbar(image, ax=ax, label="Target cell mass")
    fig.tight_layout()
    fig.savefig(args.output_dir / "online_dynamic_trajectory.png", dpi=200)
    plt.close(fig)

    strict_records = [
        r for r in records
        if not r.aggressive_self_loop
        and "zero_point_gap" not in r.mode
    ]
    summary = {
        "rows": rows,
        "cols": cols,
        "number_of_nodes": number_of_nodes,
        "blocks_completed": len(records),
        "total_samples": len(trajectory),
        "final_objective": objective(final_trajectory_eta),
        "final_discrete_objective": objective(final_eta),
        "final_trajectory_objective": objective(final_trajectory_eta),
        "physical_time": float(physical_time_history[-1]),
        "optimizer": "physical-time point-reference route-and-self-loop",
        "physical_execution": "actual-time A* route plus stationary destination self-loop",
        "contraction_reference": "Dirac occupation of current/last-position cell",
        "dynamic_feedback": True,
        "empirical_measure": "normalized exact physical residence time",
        "self_loop_near_min_fraction": args.self_loop_near_min_fraction,
        "self_loop_minimum_distance": args.self_loop_min_distance,
        "long_block_warning_time": args.long_block_warning_time,
        "maximum_predicted_self_loop_time": args.max_predicted_self_loop_time,
        "route_improvement_tolerance": args.route_improvement_tolerance,
        "routes_rejected_for_hold": int(sum(r.route_rejected_for_hold for r in records)),
        "aggressive_self_loop_releases": int(sum(r.aggressive_self_loop for r in records)),
        "long_blocks": int(sum(r.physical_block_time > args.long_block_warning_time for r in records)),
        "strict_contraction_blocks": len(strict_records),
        "contracted_strict_blocks": sum(r.contracted for r in strict_records),
        "contraction_failures": sum(not r.contracted for r in strict_records),
        "zero_warm_gap_holds": sum(r.mode == "hold_zero_warm_gap" for r in records),
        "rho": args.rho,
        "rho_endpoint": args.rho_endpoint,
        "maximum_horizon": horizon_cap,
        "initial_state": int(trajectory[0]),
        "final_state": int(trajectory[-1]),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Results written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
