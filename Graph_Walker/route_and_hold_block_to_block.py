#!/usr/bin/env python3
"""
Block-to-block route-and-hold contraction on a 64 x 64 grid.

At block k, freeze the current first variation g_k = 2(eta_k - target).
The reference error is the previous block measure p_k evaluated under g_k:

    h_warm_k = <g_k, p_k> - min(g_k).

The new block is constructed to satisfy

    <g_k, p_{k+1}> - min(g_k) <= rho * h_warm_k.

A destination is selected among vertices satisfying a stricter endpoint
condition. A* minimizes physical travel plus accumulated excess oracle error.
After reaching the destination, the system uses its self-loop until the exact
block-to-block contraction test succeeds.

Degenerate warm-gap case:
If h_warm_k is numerically zero, strict multiplicative contraction is not
meaningful. The script performs one self-loop "hold" sample. This changes the
complete empirical measure and therefore the next first variation; it is the
release-and-hold mechanism from the sketch.
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
rows = 64
cols = 64
number_of_nodes = rows * cols

rho = 1 - 1e-7
number_of_blocks = 100_000
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

objective_tolerance = 1e-15
warm_gap_tolerance = 1e-12
contraction_tolerance = 1e-12



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
) -> Tuple[List[int], float]:
    """A* with travel plus accumulated positive excess over goal error."""
    if start == goal:
        return [start], 0.0

    goal_value = float(g[goal])
    scale = max(float(np.max(g) - np.min(g)), 1e-15)

    def heuristic(node: int) -> float:
        return travel_edge_weight * manhattan_distance(node, goal)

    best = np.full(number_of_nodes, np.inf)
    predecessor = np.full(number_of_nodes, -1, dtype=np.int32)
    closed = np.zeros(number_of_nodes, dtype=bool)
    best[start] = 0.0

    queue = [(heuristic(start), 0.0, start)]
    while queue:
        _, cost, current = heapq.heappop(queue)
        if closed[current]:
            continue
        closed[current] = True

        if current == goal:
            path = [goal]
            cursor = goal
            while cursor != start:
                cursor = int(predecessor[cursor])
                if cursor < 0:
                    raise RuntimeError("Incomplete A* predecessor chain.")
                path.append(cursor)
            path.reverse()
            return path, float(cost)

        for nxt in neighbors[current]:
            if nxt == current:
                continue
            excess = max(0.0, float(g[nxt]) - goal_value)
            step_cost = travel_edge_weight + path_error_weight * excess / scale
            candidate = cost + step_cost
            if candidate < best[nxt]:
                best[nxt] = candidate
                predecessor[nxt] = current
                heapq.heappush(
                    queue,
                    (candidate + heuristic(nxt), candidate, nxt),
                )

    raise RuntimeError(f"No path from {start} to {goal}.")


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
):
    if not (0.0 <= endpoint_rho < contraction_rho < 1.0):
        raise ValueError("Require 0 <= rho_endpoint < rho < 1.")

    rng = np.random.default_rng(seed)
    if initial_state is None:
        initial_state = int(rng.integers(number_of_nodes))

    counts = np.zeros(number_of_nodes, dtype=np.int64)
    counts[initial_state] = 1
    total_samples = 1
    current = initial_state
    trajectory = [initial_state]

    # Initial reference block is the initial one-sample occupation measure.
    previous_block = [initial_state]
    previous_block_measure = block_measure(previous_block)

    records: List[BlockRecord] = []
    sample_history = [total_samples]
    objective_history = [objective(counts / total_samples)]

    for block_index in range(blocks):
        eta = counts / total_samples
        objective_before = objective(eta)
        if objective_before <= objective_tolerance:
            break

        g = first_variation(eta)
        s = float(np.min(g))
        history_gap = float(np.dot(g, eta) - s)
        warm_gap = float(np.dot(g, previous_block_measure) - s)
        warm_gap = max(0.0, warm_gap)

        if history_gap <= warm_gap_tolerance:
            break

        if warm_gap <= warm_gap_tolerance:
            # Hold step: strict contraction of zero is unavailable. The new
            # sample changes eta and hence changes g at the next iteration.
            destination = current
            path = [current]
            astar_cost = 0.0
            block = [current]
            new_block_error = float(g[current] - s)
            contracted = new_block_error <= contraction_tolerance
            travel_length = 0
            holding_length = 1
            mode = "hold_zero_warm_gap"
        else:
            choice = choose_destination(
                current=current,
                g=g,
                warm_gap=warm_gap,
                endpoint_rho=endpoint_rho,
            )
            destination = choice.node
            path, astar_cost = astar_variational_path(current, destination, g)
            (
                block,
                new_block_error,
                contracted,
                travel_length,
                holding_length,
            ) = build_block_to_block_contraction(
                path=path,
                destination=destination,
                g=g,
                s=s,
                warm_gap=warm_gap,
                contraction_rho=contraction_rho,
                horizon_cap=max_horizon,
            )
            mode = "contract" if contracted else "horizon_truncated"

        nodes = np.asarray(block, dtype=np.int64)
        np.add.at(counts, nodes, 1)
        trajectory.extend(block)
        total_samples += len(block)
        current = int(block[-1])

        new_block_measure = block_measure(block)
        eta_after = counts / total_samples
        objective_after = objective(eta_after)
        gamma = len(block) / total_samples
        contraction_ratio = (
            new_block_error / warm_gap if warm_gap > warm_gap_tolerance else np.nan
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
            )
        )

        previous_block = block
        previous_block_measure = new_block_measure
        sample_history.append(total_samples)
        objective_history.append(objective_after)

        if verbose_every > 0 and (block_index + 1) % verbose_every == 0:
            failures = sum(not r.contracted for r in records if r.mode != "hold_zero_warm_gap")
            holds = sum(r.mode == "hold_zero_warm_gap" for r in records)
            print(
                f"block={block_index + 1:7d} T={total_samples:10d} "
                f"G={objective_after:.6e} N={len(block):4d} "
                f"ratio={contraction_ratio!s:>10} holds={holds} failures={failures}"
            )

    return (
        counts / total_samples,
        trajectory,
        records,
        np.asarray(sample_history, dtype=np.int64),
        np.asarray(objective_history, dtype=float),
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
    parser.add_argument("--initial-state", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("block_to_block_output"),
    )
    parser.add_argument("--verbose-every", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.initial_state is not None and not (0 <= args.initial_state < number_of_nodes):
        raise ValueError("Initial state is outside the graph.")

    horizon_cap = None if args.max_horizon == 0 else args.max_horizon
    args.output_dir.mkdir(parents=True, exist_ok=True)

    final_eta, trajectory, records, sample_history, objective_history = run_algorithm(
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
        args.output_dir,
        sample_history,
        objective_history,
        records,
        final_eta,
        args.rho,
    )

    strict_records = [r for r in records if r.mode != "hold_zero_warm_gap"]
    summary = {
        "rows": rows,
        "cols": cols,
        "number_of_nodes": number_of_nodes,
        "blocks_completed": len(records),
        "total_samples": len(trajectory),
        "final_objective": objective(final_eta),
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
