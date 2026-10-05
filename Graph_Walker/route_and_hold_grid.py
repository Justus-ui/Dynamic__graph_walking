#!/usr/bin/env python3
"""
Route-and-hold block-to-history contraction on a rectangular grid.

Algorithm
---------
At each block:
1. Compute the empirical occupation measure eta and quadratic objective
       G(eta) = ||eta - target||_2^2.
2. Compute the first variation g = 2(eta - target), its minimum s, and
   the complete-history Frank-Wolfe gap H = <g, eta> - s.
3. Choose a sufficiently contracting destination v satisfying
       g(v) - s <= rho_endpoint * H,
   by minimizing a score combining Manhattan graph distance and normalized
   endpoint oracle error.
4. Run A* to that destination. The A* edge cost includes physical travel and
   the positive excess first-variation error above the destination value.
5. Execute the entire path, then use the destination self-loop until the
   exact empirical measure of the complete block satisfies
       <g, p_block> - s <= rho * H,
   or until maximum_horizon is reached.
6. Append the block, update the empirical history, and repeat.

The default parameters reproduce the 64 x 64 graph and Gaussian-mixture
 target supplied in the prompt. The script saves objective-history plots and
run data to the output directory.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ==================================================
# Parameters
# ==================================================
rows = 64
cols = 64
number_of_nodes = rows * cols

rho = 1 - 1e-7
number_of_blocks = 200_000
maximum_horizon = 500
random_seed = 13

# Destination must be strictly better than the final block threshold.
# This slack pays for transit error.
rho_endpoint = 0.45

# Destination score:
#   distance_weight * normalized graph distance
# + endpoint_weight * normalized endpoint oracle error.
distance_weight = 1.0
endpoint_weight = 4.0

# A* edge cost:
#   travel_edge_weight
# + path_error_weight * normalized positive excess error.
travel_edge_weight = 1.0
path_error_weight = 4.0

# Numerical settings.
objective_tolerance = 1e-15
contraction_tolerance = 1e-12
record_every_samples = 1

rng = np.random.default_rng(random_seed)


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


def objective(mu: np.ndarray) -> float:
    """Outer objective G(mu) = ||mu - target||_2^2."""
    return float(np.sum((mu - target) ** 2))


def first_variation(mu: np.ndarray) -> np.ndarray:
    """First variation g = 2(mu - target)."""
    return 2.0 * (mu - target)


# ==================================================
# Nonuniform target distribution
# ==================================================
centroids = np.zeros((number_of_nodes, 2), dtype=float)

for node in range(number_of_nodes):
    row, col = divmod(node, cols)
    centroids[node, 0] = (col + 0.5) / cols
    centroids[node, 1] = (row + 0.5) / rows


def gaussian_density(
    points: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    difference = points - mean
    inverse_covariance = np.linalg.inv(covariance)
    exponent = np.einsum(
        "ni,ij,nj->n",
        difference,
        inverse_covariance,
        difference,
    )
    return np.exp(-0.5 * exponent)


density_1 = gaussian_density(
    points=centroids,
    mean=np.array([0.25, 0.75]),
    covariance=np.array([[0.012, 0.0], [0.0, 0.018]]),
)

density_2 = gaussian_density(
    points=centroids,
    mean=np.array([0.75, 0.30]),
    covariance=np.array([[0.020, 0.008], [0.008, 0.015]]),
)

density_3 = gaussian_density(
    points=centroids,
    mean=np.array([0.70, 0.80]),
    covariance=np.array([[0.008, 0.0], [0.0, 0.008]]),
)

target = 0.50 * density_1 + 0.35 * density_2 + 0.15 * density_3
background_weight = 0.02
target = (1.0 - background_weight) * target + background_weight
target = target / np.sum(target)

assert np.all(target >= 0.0)
assert np.isclose(np.sum(target), 1.0)


# ==================================================
# Data structures
# ==================================================
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
    endpoint_error: float
    block_error: float
    contraction_ratio: float
    contracted: bool
    gamma: float
    astar_cost: float


# ==================================================
# Graph helpers
# ==================================================
def manhattan_distance(node_a: int, node_b: int) -> int:
    row_a, col_a = node_coordinates(node_a)
    row_b, col_b = node_coordinates(node_b)
    return abs(row_a - row_b) + abs(col_a - col_b)


def choose_suitable_destination(
    current: int,
    g: np.ndarray,
    history_gap: float,
    endpoint_rho: float,
    distance_w: float,
    endpoint_w: float,
) -> DestinationChoice:
    """
    Choose a destination that is sufficiently contracting and nearby.

    Feasibility condition:
        g(v) - min(g) <= endpoint_rho * H.

    Among feasible vertices minimize:
        distance_w * normalized Manhattan distance
      + endpoint_w * normalized endpoint error.

    Exact minimizers are always feasible when H > 0.
    """
    s = float(np.min(g))
    endpoint_errors = g - s

    if history_gap <= contraction_tolerance:
        return DestinationChoice(
            node=current,
            distance=0,
            endpoint_error=0.0,
            normalized_endpoint_error=0.0,
            score=0.0,
        )

    feasible = endpoint_errors <= endpoint_rho * history_gap
    candidate_nodes = np.flatnonzero(feasible)

    if candidate_nodes.size == 0:
        candidate_nodes = np.flatnonzero(
            np.isclose(g, s, atol=1e-15, rtol=0.0)
        )

    current_row, current_col = node_coordinates(current)
    candidate_rows = candidate_nodes // cols
    candidate_cols = candidate_nodes % cols
    distances = (
        np.abs(candidate_rows - current_row)
        + np.abs(candidate_cols - current_col)
    ).astype(float)

    max_distance = max(1.0, float(rows + cols - 2))
    normalized_distances = distances / max_distance
    normalized_errors = endpoint_errors[candidate_nodes] / history_gap

    scores = (
        distance_w * normalized_distances
        + endpoint_w * normalized_errors
    )

    best_position = int(np.argmin(scores))
    node = int(candidate_nodes[best_position])

    return DestinationChoice(
        node=node,
        distance=int(distances[best_position]),
        endpoint_error=float(endpoint_errors[node]),
        normalized_endpoint_error=float(normalized_errors[best_position]),
        score=float(scores[best_position]),
    )


def astar_variational_path(
    start: int,
    goal: int,
    g: np.ndarray,
    travel_weight: float,
    error_weight: float,
) -> Tuple[List[int], float]:
    """
    A* path from start to goal.

    The cost of entering node v is
        travel_weight
      + error_weight * [g(v) - g(goal)]_+ / scale.

    Thus A* prefers short routes whose intermediate vertices have low
    frozen first-variation error relative to the chosen destination.

    The heuristic uses only the remaining travel cost, so it is admissible
    because the additional variational penalty is nonnegative.
    """
    if start == goal:
        return [start], 0.0

    goal_value = float(g[goal])
    scale = max(float(np.max(g) - np.min(g)), 1e-15)

    def heuristic(node: int) -> float:
        return travel_weight * manhattan_distance(node, goal)

    best_cost = np.full(number_of_nodes, np.inf, dtype=float)
    predecessor = np.full(number_of_nodes, -1, dtype=np.int32)
    closed = np.zeros(number_of_nodes, dtype=bool)

    best_cost[start] = 0.0
    queue: List[Tuple[float, float, int]] = []
    heapq.heappush(queue, (heuristic(start), 0.0, start))

    while queue:
        _, current_cost, current = heapq.heappop(queue)

        if closed[current]:
            continue
        closed[current] = True

        if current == goal:
            path = [goal]
            cursor = goal

            while cursor != start:
                cursor = int(predecessor[cursor])
                if cursor < 0:
                    raise RuntimeError("A* predecessor chain is incomplete.")
                path.append(cursor)

            path.reverse()
            return path, float(current_cost)

        for nxt in neighbors[current]:
            if nxt == current:
                # Self-loops are used during holding, not routing.
                continue

            excess_error = max(0.0, float(g[nxt]) - goal_value)
            step_cost = (
                travel_weight
                + error_weight * excess_error / scale
            )
            candidate_cost = current_cost + step_cost

            if candidate_cost < best_cost[nxt]:
                best_cost[nxt] = candidate_cost
                predecessor[nxt] = current
                heapq.heappush(
                    queue,
                    (
                        candidate_cost + heuristic(nxt),
                        candidate_cost,
                        nxt,
                    ),
                )

    raise RuntimeError(f"No path found from {start} to {goal}.")


# ==================================================
# Exact route-and-hold block construction
# ==================================================
def build_contracting_block(
    path: Sequence[int],
    destination: int,
    g: np.ndarray,
    s: float,
    history_gap: float,
    contraction_rho: float,
    horizon_cap: Optional[int],
) -> Tuple[List[int], float, bool, int, int]:
    """
    Execute the route, then hold at the destination until the exact frozen
    block criterion is met:

        mean_{v in block} g(v) - s <= contraction_rho * history_gap.

    The returned block contains newly generated samples only. Therefore,
    path[0] (the already occupied current node) is not counted again.

    If already at the destination, one self-loop sample is generated.
    """
    route_samples = list(path[1:])

    if not route_samples:
        route_samples = [destination]
        travel_length = 0
    else:
        travel_length = len(route_samples)

    if horizon_cap is not None and len(route_samples) > horizon_cap:
        # The cap is an implementation safeguard. This truncated block need
        # not satisfy contraction; the caller records this explicitly.
        route_samples = route_samples[:horizon_cap]

    block = list(route_samples)
    total_g = float(np.sum(g[np.asarray(block, dtype=int)]))

    def current_error() -> float:
        return total_g / len(block) - s

    contracted = current_error() <= (
        contraction_rho * history_gap + contraction_tolerance
    )

    while not contracted:
        if horizon_cap is not None and len(block) >= horizon_cap:
            break

        block.append(destination)
        total_g += float(g[destination])
        contracted = current_error() <= (
            contraction_rho * history_gap + contraction_tolerance
        )

    block_error = current_error()
    holding_length = max(0, len(block) - len(route_samples))

    return (
        block,
        float(block_error),
        bool(contracted),
        int(travel_length),
        int(holding_length),
    )


# ==================================================
# Main algorithm
# ==================================================
def run_route_and_hold(
    blocks: int,
    max_horizon: Optional[int],
    contraction_rho: float,
    endpoint_rho: float,
    seed: int,
    initial_state: Optional[int] = None,
    verbose_every: int = 500,
) -> Tuple[np.ndarray, List[int], List[BlockRecord], np.ndarray, np.ndarray]:
    if not (0.0 <= endpoint_rho < contraction_rho < 1.0):
        raise ValueError("Require 0 <= rho_endpoint < rho < 1.")

    local_rng = np.random.default_rng(seed)

    if initial_state is None:
        initial_state = int(local_rng.integers(number_of_nodes))

    counts = np.zeros(number_of_nodes, dtype=np.int64)
    counts[initial_state] = 1

    current = initial_state
    total_samples = 1
    trajectory: List[int] = [initial_state]
    records: List[BlockRecord] = []

    sample_history = [total_samples]
    objective_history = [objective(counts / total_samples)]

    for block_index in range(blocks):
        eta = counts / total_samples
        objective_before = objective(eta)

        if objective_before <= objective_tolerance:
            print(
                f"Stopping at block {block_index}: "
                f"objective {objective_before:.3e}."
            )
            break

        g = first_variation(eta)
        s = float(np.min(g))
        history_gap = float(np.dot(g, eta) - s)

        if history_gap <= contraction_tolerance:
            print(
                f"Stopping at block {block_index}: "
                f"history gap {history_gap:.3e}."
            )
            break

        choice = choose_suitable_destination(
            current=current,
            g=g,
            history_gap=history_gap,
            endpoint_rho=endpoint_rho,
            distance_w=distance_weight,
            endpoint_w=endpoint_weight,
        )

        path, astar_cost = astar_variational_path(
            start=current,
            goal=choice.node,
            g=g,
            travel_weight=travel_edge_weight,
            error_weight=path_error_weight,
        )

        (
            block,
            block_error,
            contracted,
            travel_length,
            holding_length,
        ) = build_contracting_block(
            path=path,
            destination=choice.node,
            g=g,
            s=s,
            history_gap=history_gap,
            contraction_rho=contraction_rho,
            horizon_cap=max_horizon,
        )

        block_nodes = np.asarray(block, dtype=np.int64)
        np.add.at(counts, block_nodes, 1)
        trajectory.extend(block)

        block_length = len(block)
        total_samples += block_length
        current = int(block[-1])

        eta_after = counts / total_samples
        objective_after = objective(eta_after)
        gamma = block_length / total_samples
        contraction_ratio = block_error / history_gap
        destination_row, destination_col = node_coordinates(choice.node)

        records.append(
            BlockRecord(
                block=block_index,
                total_samples=total_samples,
                block_length=block_length,
                travel_length=travel_length,
                holding_length=holding_length,
                destination=choice.node,
                destination_row=destination_row,
                destination_col=destination_col,
                objective_before=objective_before,
                objective_after=objective_after,
                history_gap=history_gap,
                endpoint_error=choice.endpoint_error,
                block_error=block_error,
                contraction_ratio=contraction_ratio,
                contracted=contracted,
                gamma=gamma,
                astar_cost=astar_cost,
            )
        )

        sample_history.append(total_samples)
        objective_history.append(objective_after)

        if (
            verbose_every > 0
            and (block_index + 1) % verbose_every == 0
        ):
            contraction_failures = sum(
                not record.contracted for record in records
            )
            print(
                f"block={block_index + 1:7d}  "
                f"T={total_samples:10d}  "
                f"G={objective_after:.6e}  "
                f"N={block_length:4d}  "
                f"ratio={contraction_ratio:.4f}  "
                f"failures={contraction_failures}"
            )

    final_eta = counts / total_samples

    return (
        final_eta,
        trajectory,
        records,
        np.asarray(sample_history, dtype=np.int64),
        np.asarray(objective_history, dtype=float),
    )


# ==================================================
# Output and plotting
# ==================================================
def save_records(records: Sequence[BlockRecord], path: Path) -> None:
    if not records:
        return

    fieldnames = list(records[0].__dict__.keys())

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
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
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Objective against total physical samples.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(
        sample_history,
        objective_history,
        color="tab:blue",
        linewidth=1.5,
        label=r"$G(\eta_T)$",
    )

    if len(sample_history) >= 2:
        reference = objective_history[0] * (
            sample_history / sample_history[0]
        ) ** (-2.0 / 3.0)
        ax.loglog(
            sample_history,
            reference,
            linestyle="--",
            color="tab:orange",
            linewidth=1.2,
            label=r"reference $T^{-2/3}$",
        )

    ax.set_xlabel("Total trajectory samples T")
    ax.set_ylabel(r"Objective $G(\eta_T)$")
    ax.set_title("Route-and-hold objective history")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "objective_history.png", dpi=180)
    plt.close(fig)

    if records:
        block_indices = np.array(
            [record.block for record in records], dtype=int
        )
        block_lengths = np.array(
            [record.block_length for record in records], dtype=float
        )
        ratios = np.array(
            [record.contraction_ratio for record in records], dtype=float
        )

        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        axes[0].plot(block_indices, block_lengths, linewidth=0.8)
        axes[0].set_ylabel("Block length")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(block_indices, ratios, linewidth=0.8)
        axes[1].axhline(
            contraction_rho,
            color="tab:red",
            linestyle="--",
            label=r"target $\rho$",
        )
        axes[1].set_xlabel("Block index")
        axes[1].set_ylabel("Block/history gap ratio")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(
            output_dir / "block_diagnostics.png",
            dpi=180,
        )
        plt.close(fig)

    target_image = target.reshape(rows, cols)
    eta_image = final_eta.reshape(rows, cols)
    residual_image = eta_image - target_image

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    image_0 = axes[0].imshow(target_image, origin="lower", cmap="viridis")
    axes[0].set_title("Target distribution")
    fig.colorbar(image_0, ax=axes[0], fraction=0.046)

    image_1 = axes[1].imshow(eta_image, origin="lower", cmap="viridis")
    axes[1].set_title("Final empirical distribution")
    fig.colorbar(image_1, ax=axes[1], fraction=0.046)

    limit = max(abs(float(residual_image.min())), abs(float(residual_image.max())))
    image_2 = axes[2].imshow(
        residual_image,
        origin="lower",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axes[2].set_title("Empirical minus target")
    fig.colorbar(image_2, ax=axes[2], fraction=0.046)

    for axis in axes:
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")

    fig.tight_layout()
    fig.savefig(output_dir / "distribution_comparison.png", dpi=180)
    plt.close(fig)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route-and-hold block-to-history contraction on a grid."
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=number_of_blocks,
        help="Maximum number of blocks.",
    )
    parser.add_argument(
        "--max-horizon",
        type=int,
        default=maximum_horizon,
        help=(
            "Maximum block length. Use 0 for no cap. A finite cap can "
            "cause contraction failures if it truncates the holding phase."
        ),
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=rho,
        help="Required final block-to-history contraction factor.",
    )
    parser.add_argument(
        "--rho-endpoint",
        type=float,
        default=rho_endpoint,
        help="Maximum endpoint error factor; must be strictly below rho.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random_seed,
        help="Random seed used only for choosing the initial state.",
    )
    parser.add_argument(
        "--initial-state",
        type=int,
        default=None,
        help="Optional initial node index.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("route_and_hold_output"),
        help="Directory for plots and CSV output.",
    )
    parser.add_argument(
        "--verbose-every",
        type=int,
        default=500,
        help="Print progress every this many blocks; use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.initial_state is not None:
        if not (0 <= args.initial_state < number_of_nodes):
            raise ValueError("Initial state is outside the graph.")

    horizon_cap = None if args.max_horizon == 0 else args.max_horizon
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (
        final_eta,
        trajectory,
        records,
        sample_history,
        objective_history,
    ) = run_route_and_hold(
        blocks=args.blocks,
        max_horizon=horizon_cap,
        contraction_rho=args.rho,
        endpoint_rho=args.rho_endpoint,
        seed=args.seed,
        initial_state=args.initial_state,
        verbose_every=args.verbose_every,
    )

    save_records(records, args.output_dir / "block_history.csv")
    np.save(args.output_dir / "trajectory.npy", np.asarray(trajectory, dtype=np.int32))
    np.save(args.output_dir / "final_empirical_measure.npy", final_eta)
    np.save(args.output_dir / "target.npy", target)
    np.save(args.output_dir / "sample_history.npy", sample_history)
    np.save(args.output_dir / "objective_history.npy", objective_history)

    save_plots(
        output_dir=args.output_dir,
        sample_history=sample_history,
        objective_history=objective_history,
        records=records,
        final_eta=final_eta,
        contraction_rho=args.rho,
    )

    contracted_blocks = sum(record.contracted for record in records)
    failed_blocks = len(records) - contracted_blocks

    summary = {
        "rows": rows,
        "cols": cols,
        "number_of_nodes": number_of_nodes,
        "blocks_completed": len(records),
        "total_samples": len(trajectory),
        "final_objective": objective(final_eta),
        "contracted_blocks": contracted_blocks,
        "contraction_failures": failed_blocks,
        "rho": args.rho,
        "rho_endpoint": args.rho_endpoint,
        "maximum_horizon": horizon_cap,
        "initial_state": int(trajectory[0]),
        "final_state": int(trajectory[-1]),
    }

    with (args.output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Results written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
