WAYPOINT OCCUPATION-MATCHING SCRIPTS
====================================

OVERVIEW
--------
This folder contains three Python experiments for matching a target occupation
distribution on a rectangular grid. All scripts use a single-integrator motion
model, x_dot = u with ||u|| <= u_max, and evaluate the quadratic objective

    G(eta) = ||eta - target||^2.

The target is a normalized mixture of Gaussian-shaped densities over the grid.
A planned cell route is converted into a continuous trajectory through entry
waypoints. The physical time spent before and after each cell boundary is
assigned to the corresponding source and destination cells.

The main differences between the scripts are:

1. Whether optimization feedback uses discrete visit counts or physical
   residence time.
2. Whether contraction is measured against the previous block or the complete
   accumulated history.
3. Whether A* uses a standard cell-level cost or a continuous-position,
   residence-time-aware cost.

Required packages:

    numpy
    matplotlib

Each script provides command-line arguments. Run a script with --help to see
all available options.


1. waypoint_block_to_block_discrete.py
--------------------------------------
Purpose:
    Discrete block-to-block route-and-hold optimizer with separate continuous
    trajectory execution.

Optimization state:
    The empirical measure used by the optimizer is based on discrete cell
    samples. Every route cell and every self-loop sample has equal weight.
    Physical travel time does not feed back into destination selection or the
    discrete gradient.

Contraction reference:
    The previous discrete block measure. At block k, the new block is built to
    contract the previous block error under the newly frozen gradient.

Path planning:
    Standard grid A*. Its cost combines graph travel and positive excess
    gradient error relative to the destination.

Physical execution:
    The discrete route is executed continuously. Travel time is split between
    source and destination cells. Destination holding is represented by the
    physical execution policy in the script. The resulting physical occupation
    and trajectory objective are recorded separately from the discrete
    optimizer objective.

Important interpretation:
    This script is the nominal discrete baseline. It is useful for comparing a
    coherent discrete optimizer against the physical trajectory produced by its
    cell sequence. The physical trajectory is evaluated, but it does not alter
    subsequent optimization decisions.

Typical command:

    python waypoint_block_to_block_discrete.py \
        --blocks 40000 \
        --max-horizon 500 \
        --verbose-every 500 \
        --output-dir results_block_to_block_discrete

Main outputs:
    block_history.csv
    summary.json
    objective_history.npy
    trajectory_objective_history.npy
    physical_time_history.npy
    final_empirical_measure.npy
    final_trajectory_empirical_measure.npy
    dynamic_trajectory_positions.npy
    objective and trajectory plots


2. waypoint_block_to_history.py
-------------------------------
Purpose:
    Physical block-to-history occupation matching with standard grid A*.

Optimization state:
    The empirical measure eta is the normalized exact physical residence time
    accumulated in every grid cell. Consequently, the gradient and reported
    objective directly correspond to the continuously executed trajectory.

Contraction reference:
    The complete accumulated physical history. The new route-and-patrol block
    is constructed so that its frozen-gradient error contracts relative to the
    current history error.

Path planning:
    Standard cell-level A*. The path cost combines a unit travel term with the
    positive excess gradient value of the candidate next cell relative to the
    destination. A* does not propagate continuous entry positions while
    searching.

Physical execution:
    The selected cell route is realized through projected entry waypoints. The
    required destination holding time is calculated from the complete physical
    transit residence vector. Holding is executed as uniform waypoint patrol
    inside the destination cell for exactly the requested duration.

Fallback behavior:
    If the route exceeds max_horizon, the required patrol is not finite, or an
    enabled max_block_time is exceeded, the script performs an in-cell fallback
    patrol and recomputes the gradient on the following block.

Important parameters:
    --max-horizon
        Maximum number of route transitions. Use 0 to disable.

    --max-block-time
        Maximum total route-plus-patrol time. Use 0 to disable.

    --self-loop-time
        Duration used for initialization, zero-route minimum patrol, and
        fallback patrols.

Typical command:

    python waypoint_block_to_history.py \
        --blocks 40000 \
        --max-horizon 500 \
        --max-block-time 0 \
        --self-loop-time 0.01 \
        --verbose-every 500 \
        --output-dir results_block_to_history

Main outputs:
    block_history.csv
    summary.json
    objective_history.npy
    time_history.npy
    final_empirical_measure.npy
    trajectory_positions.npy
    objective_history.png
    trajectory_over_target.png


3. waypoint_block_to_history_dynamic_astar.py
---------------------------------------------
Purpose:
    Physical block-to-history occupation matching with a dynamic,
    residence-time-aware A* planner.

Optimization state and contraction:
    These are the same physical block-to-history quantities as in
    waypoint_block_to_history.py. The empirical measure, gradient, contraction
    reference, and objective use exact physical residence time.

Path planning:
    This is the main difference from the standard block-to-history script.
    Each A* state is represented by

        (current cell, incoming cell).

    Distinguishing the incoming cell preserves information about the side from
    which the trajectory enters a cell. A predicted continuous entry waypoint
    is propagated with every best A* label.

    For a transition u -> v, the edge cost uses the exact predicted residence
    split:

        t_u * max(0, g[u] - g[goal])
        + t_v * max(0, g[v] - g[goal]).

    A lightly weighted, dimensionally scaled Manhattan heuristic guides the
    search. This planner therefore accounts for the current continuous position
    and expected physical travel occupation while selecting the cell path.

Physical execution and fallback:
    Route realization, required patrol calculation, uniform destination patrol,
    max_horizon handling, and optional max_block_time fallback follow the same
    structure as in waypoint_block_to_history.py.

Typical command:

    python waypoint_block_to_history_dynamic_astar.py \
        --blocks 30000 \
        --max-horizon 500 \
        --max-block-time 0 \
        --self-loop-time 0.01 \
        --verbose-every 500 \
        --output-dir results_block_to_history_dynamic_astar

Main outputs:
    block_history.csv
    summary.json
    objective_history.npy
    time_history.npy
    final_empirical_measure.npy
    trajectory_positions.npy
    objective_history.png
    trajectory_over_target.png


RECOMMENDED COMPARISONS
-----------------------
For a fair numerical comparison, use the same random seed, speed, horizon,
number of blocks, and target/grid configuration whenever possible.

The most informative primary plot is:

    trajectory objective G(eta) versus elapsed physical time.

Additional useful diagnostics are:

    objective versus block index
    travel time and patrol time per block
    contraction ratio per block
    route length
    number of contracted and fallback blocks
    final empirical occupation minus target

The two block-to-history scripts can be compared directly to isolate the effect
of the A* model. Comparing either of them with the discrete block-to-block script
also changes the feedback measure and contraction reference, so differences
cannot be attributed to A* alone.


NOTES ON TERMINOLOGY
--------------------
Discrete empirical measure:
    Equal weight per generated cell sample, independent of physical duration.

Physical empirical measure:
    Weight proportional to the exact time spent in each cell.

Block-to-block contraction:
    The new block is compared with the previous block under the current frozen
    gradient.

Block-to-history contraction:
    The new block is compared with the complete accumulated empirical history.

Uniform patrol:
    Holding is implemented by moving between random interior waypoints of the
    same cell. Since the cell is convex, all patrol segments remain inside it,
    and the entire patrol duration is assigned to that cell.
