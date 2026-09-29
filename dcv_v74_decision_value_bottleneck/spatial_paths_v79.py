"""Spatially continuous candidate paths for the V7.9 controlled task.

Each graph edge is represented by exactly one image patch.  Five hub patches
are arranged from west to east.  Between each adjacent pair of hubs there are
three edge patches (upper, middle and lower).  Choosing one edge in each of
the four stages gives 3**4 = 81 candidate paths.
"""

from itertools import product

import numpy as np


N_STAGES = 4
CHOICES_PER_STAGE = 3
N_SPATIAL_EDGES = N_STAGES * CHOICES_PER_STAGE
N_SPATIAL_PATHS = CHOICES_PER_STAGE**N_STAGES


def patch_index(row, column, grid):
    return int(row * grid + column)


def patch_coordinate(index, grid):
    return divmod(int(index), int(grid))


def spatial_parallel_corridor(rng, grid):
    """Return a fixed observable 9x9 corridor and its 81 continuous paths.

    The topology is centered when ``grid`` is larger than nine.  A graph edge
    is a macro-edge between two hub nodes, and its visual evidence lives in the
    single intermediate patch stored in ``edge_patch``.
    """
    grid = int(grid)
    if grid < 9:
        raise ValueError(
            "V7.9 spatial paths require GRID>=9 so four three-way stages fit"
        )

    center_row = grid // 2
    column_offset = (grid - 9) // 2
    hub_columns = [column_offset + 2 * stage for stage in range(N_STAGES + 1)]
    hub_patches = np.asarray(
        [patch_index(center_row, column, grid) for column in hub_columns],
        dtype=np.int64,
    )

    edge_patch = []
    edges = []
    edge_stage = []
    edge_choice = []
    for stage in range(N_STAGES):
        branch_column = hub_columns[stage] + 1
        for choice, row_offset in enumerate((-1, 0, 1)):
            edge_patch.append(
                patch_index(center_row + row_offset, branch_column, grid)
            )
            # Three parallel macro-edges connect the same adjacent hubs.
            edges.append((stage, stage + 1))
            edge_stage.append(stage)
            edge_choice.append(choice)

    edge_patch = np.asarray(edge_patch, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64)
    edge_stage = np.asarray(edge_stage, dtype=np.int64)
    edge_choice = np.asarray(edge_choice, dtype=np.int64)

    path_edge_indices = []
    path_patch_sequence = []
    for choices in product(range(CHOICES_PER_STAGE), repeat=N_STAGES):
        path_edges = np.asarray(
            [
                stage * CHOICES_PER_STAGE + choice
                for stage, choice in enumerate(choices)
            ],
            dtype=np.int64,
        )
        sequence = [int(hub_patches[0])]
        for stage, edge_index in enumerate(path_edges):
            sequence.extend(
                [int(edge_patch[edge_index]), int(hub_patches[stage + 1])]
            )
        path_edge_indices.append(path_edges)
        path_patch_sequence.append(sequence)

    path_edge_indices = np.stack(path_edge_indices)
    path_patch_sequence = np.asarray(path_patch_sequence, dtype=np.int64)
    path_mask = np.zeros((N_SPATIAL_PATHS, N_SPATIAL_EDGES), dtype=np.float32)
    for path_index, path_edges in enumerate(path_edge_indices):
        path_mask[path_index, path_edges] = 1.0

    # Middle branches are very slightly shorter; semantic risk is still the
    # dominant source of path changes.
    choice_bias = np.asarray([0.02, -0.02, 0.02], dtype=np.float32)
    base_cost = 1.0 + np.tile(choice_bias, N_STAGES)
    base_cost += rng.uniform(-0.01, 0.01, size=N_SPATIAL_EDGES).astype(np.float32)

    topology = {
        "edges": edges,
        "base_cost": base_cost.astype(np.float32),
        "path_mask": path_mask,
        "edge_patch": edge_patch,
        "hub_patches": hub_patches,
        "edge_stage": edge_stage,
        "edge_choice": edge_choice,
        "path_edge_indices": path_edge_indices,
        "path_patch_sequence": path_patch_sequence,
    }
    validate_spatial_topology(topology, grid)
    return topology


def validate_spatial_topology(topology, grid):
    """Fail fast if edge/patch alignment or spatial continuity is broken."""
    edge_patch = np.asarray(topology["edge_patch"])
    path_mask = np.asarray(topology["path_mask"])
    path_edges = np.asarray(topology["path_edge_indices"])
    sequences = np.asarray(topology["path_patch_sequence"])

    if edge_patch.shape != (N_SPATIAL_EDGES,):
        raise ValueError(f"expected {N_SPATIAL_EDGES} edge patches")
    if len(np.unique(edge_patch)) != N_SPATIAL_EDGES:
        raise ValueError("every graph edge must map to a distinct image patch")
    if path_mask.shape != (N_SPATIAL_PATHS, N_SPATIAL_EDGES):
        raise ValueError("path_mask must have shape [81, 12]")
    if not np.all(path_mask.sum(axis=1) == N_STAGES):
        raise ValueError(
            "every path must select one edge in each of four stages"
        )
    if sequences.shape != (N_SPATIAL_PATHS, 2 * N_STAGES + 1):
        raise ValueError(
            "each physical path must contain five hubs and four edge patches"
        )

    for path_index, sequence in enumerate(sequences):
        expected_edge_patches = edge_patch[path_edges[path_index]]
        if not np.array_equal(sequence[1::2], expected_edge_patches):
            raise ValueError("path sequence and selected graph edges disagree")
        coordinates = [patch_coordinate(index, grid) for index in sequence]
        for left, right in zip(coordinates[:-1], coordinates[1:]):
            chebyshev_distance = max(
                abs(left[0] - right[0]), abs(left[1] - right[1])
            )
            if chebyshev_distance != 1:
                raise ValueError(
                    f"path {path_index} is not spatially continuous: {left} -> {right}"
                )
