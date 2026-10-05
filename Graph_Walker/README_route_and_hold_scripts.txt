ROUTE-AND-HOLD GRID SCRIPTS
===========================

OVERVIEW
--------
This folder contains two discrete route-and-hold occupation-matching
experiments on a rectangular grid:

    route_and_hold_block_to_block.py
    route_and_hold_grid.py

Both scripts seek to approximate a nonuniform target distribution with an
empirical occupation measure eta. They minimize the quadratic objective

    G(eta) = ||eta - target||_2^2

and use the first variation

    g = 2(eta - target).

At every iteration, a destination is selected, A* computes a route of adjacent
grid cells, and destination self-loop samples are appended until a specified
contraction condition is met or the maximum horizon is reached.

These scripts are fully discrete. One generated cell index corresponds to one
empirical sample. They do not simulate continuous motion, physical travel time,
or within-cell patrol dynamics.

Required packages:

    numpy
    matplotlib

Run either script with --help to view its command-line arguments.


SHARED MODEL AND ALGORITHM COMPONENTS
-------------------------------------
Grid and target:
    Both scripts use a 64 x 64 rectangular grid with four-neighbor movement and
    self-loops. The target is a normalized mixture of three Gaussian-shaped
    densities plus a small uniform background component.

Destination selection:
    Candidate destinations must have sufficiently small frozen-gradient error.
    Among feasible candidates, the scripts minimize a weighted score combining
    normalized Manhattan distance and normalized endpoint error.

A* routing:
    Standard grid A* finds a path to the selected destination. Its edge cost
    combines a unit travel cost and a nonnegative penalty for entering cells
    whose frozen-gradient value exceeds the destination value.

Route-and-hold construction:
    The already occupied path start is not counted again. Newly entered route
    cells are appended to the block. If the route contribution does not satisfy
    the relevant contraction condition, the destination is appended repeatedly
    as a self-loop sample.

Maximum horizon:
    --max-horizon limits the total number of newly generated samples in a block.
    Use 0 to disable the cap. A finite cap can truncate the holding phase and
    produce a recorded contraction failure.


1. route_and_hold_block_to_block.py
-----------------------------------
Purpose:
    Implements discrete block-to-block route-and-hold contraction.

Empirical measure:
    The global empirical measure is the normalized count of all generated cell
    samples.

Contraction reference:
    The previous block measure is re-evaluated under the current frozen
    gradient. If p_k denotes the previous block, the warm error is

        h_warm,k = <g_k, p_k> - min(g_k).

    The new block p_{k+1} is constructed to satisfy

        <g_k, p_{k+1}> - min(g_k)
            <= rho * h_warm,k.

    The previous block, rather than the complete trajectory history, therefore
    supplies the contraction threshold.

Zero warm-gap handling:
    If the previous block has numerically zero error under the new gradient,
    strict multiplicative contraction is unavailable. The script generates one
    self-loop sample at the current cell. This release-and-hold step changes the
    global empirical measure and allows the gradient to be recomputed.

Recorded diagnostics:
    Each block record includes the complete-history gap, previous-block error,
    new-block error, contraction ratio, route length, holding length, selected
    destination, A* cost, and objective before and after the block.

Typical command:

    python route_and_hold_block_to_block.py \
        --blocks 100000 \
        --max-horizon 500 \
        --rho 0.9999999 \
        --rho-endpoint 0.45 \
        --verbose-every 500 \
        --output-dir results_block_to_block

Main outputs:
    block_history.csv
    summary.json
    trajectory.npy
    target.npy
    final_empirical_measure.npy
    sample_history.npy
    objective_history.npy
    objective_history.png
    block_diagnostics.png
    distribution_comparison.png

Use this script when:
    You want to study successive block contraction, warm-start behavior, or the
    relation between consecutive route-and-hold blocks.


2. route_and_hold_grid.py
-------------------------
Purpose:
    Implements discrete block-to-history route-and-hold contraction.

Empirical measure:
    The global empirical measure is the normalized count of all generated cell
    samples, as in the block-to-block script.

Contraction reference:
    The complete accumulated history supplies the contraction threshold. The
    history Frank-Wolfe gap is

        H_k = <g_k, eta_k> - min(g_k).

    The new block p_k is constructed to satisfy

        <g_k, p_k> - min(g_k)
            <= rho * H_k.

    The block is therefore compared directly with the current complete
    empirical history rather than with the previous block.

Destination condition:
    A feasible destination v satisfies

        g_k(v) - min(g_k)
            <= rho_endpoint * H_k.

    The stricter endpoint threshold provides slack for route samples whose
    gradient values may be higher than the destination value.

Recorded diagnostics:
    Each block record includes the history gap, endpoint error, block error,
    contraction ratio, route and holding lengths, destination, A* cost, mixing
    coefficient gamma, and objective before and after the block.

Reference-rate plot:
    The objective-history plot also includes a T^(-2/3) reference curve, which
    provides a visual rate comparison. The curve is only a plotting reference
    and does not alter the algorithm.

Typical command:

    python route_and_hold_grid.py \
        --blocks 200000 \
        --max-horizon 500 \
        --rho 0.9999999 \
        --rho-endpoint 0.45 \
        --verbose-every 500 \
        --output-dir results_block_to_history

Main outputs:
    block_history.csv
    summary.json
    trajectory.npy
    target.npy
    final_empirical_measure.npy
    sample_history.npy
    objective_history.npy
    objective_history.png
    block_diagnostics.png
    distribution_comparison.png

Use this script when:
    You want each new route-and-hold block to contract relative to the complete
    accumulated trajectory history.


KEY DIFFERENCE
--------------
The scripts share the same target, grid, destination logic, A* model, and basic
route-and-hold construction. Their essential difference is the contraction
reference.

Block-to-block:

    new block versus previous block

    <g_k, p_{k+1}> - min(g_k)
        <= rho * (<g_k, p_k> - min(g_k)).

Block-to-history:

    new block versus complete empirical history

    <g_k, p_k> - min(g_k)
        <= rho * (<g_k, eta_k> - min(g_k)).

This distinction changes the destination feasibility threshold, required number
of destination self-loops, contraction ratio, and interpretation of each block.


IMPORTANT PARAMETERS
--------------------
--blocks
    Maximum number of route-and-hold blocks.

--max-horizon
    Maximum number of samples generated in a block. Use 0 for no cap.

--rho
    Requested contraction factor. It must satisfy rho_endpoint < rho < 1.

--rho-endpoint
    Stricter endpoint-error factor used during destination selection.

--seed
    Random seed used to choose the initial grid node when no initial state is
    supplied.

--initial-state
    Optional explicit initial node index.

--verbose-every
    Number of blocks between console progress messages. Use 0 to disable.

--output-dir
    Directory receiving arrays, CSV data, JSON summary, and plots.


OUTPUT INTERPRETATION
---------------------
trajectory.npy
    Complete discrete sequence of occupied grid-node indices.

final_empirical_measure.npy
    Final normalized visit-count distribution over all grid nodes.

target.npy
    Target occupation distribution used by the experiment.

sample_history.npy
    Total number of generated samples after every completed block.

objective_history.npy
    Quadratic objective value after every completed block.

block_history.csv
    Per-block optimization and contraction diagnostics.

summary.json
    Compact run summary containing final objective, contraction counts,
    parameter values, and initial/final states.

objective_history.png
    Log-log objective curve versus total sample count.

block_diagnostics.png
    Block length and contraction-ratio diagnostics.

distribution_comparison.png
    Target, final empirical distribution, and residual heatmaps.


RECOMMENDED COMPARISON
----------------------
To compare the two algorithms fairly, use identical values for:

    --blocks
    --max-horizon
    --rho
    --rho-endpoint
    --seed
    --initial-state

Useful comparison plots and statistics include:

    objective versus total sample count
    objective versus block index
    block length
    route length versus holding length
    contraction ratio
    number of horizon-truncated or failed blocks
    final empirical residual

Because both scripts are discrete and share the same A* implementation, their
results provide a focused comparison of block-to-block and block-to-history
contraction without introducing continuous-time execution differences.
