from dataclasses import replace

import numpy as np

from config import CFG
from generate_dataset_v79_standalone import build_one
from spatial_paths_v79 import (
    N_SPATIAL_EDGES,
    N_SPATIAL_PATHS,
    patch_coordinate,
    spatial_parallel_corridor,
)


def v79_config():
    return replace(CFG, grid=9)


def test_all_81_paths_are_spatially_continuous_and_edge_aligned():
    topology = spatial_parallel_corridor(np.random.default_rng(1), grid=9)
    assert topology["edge_patch"].shape == (N_SPATIAL_EDGES,)
    assert topology["path_mask"].shape == (N_SPATIAL_PATHS, N_SPATIAL_EDGES)
    assert np.all(topology["path_mask"].sum(axis=1) == 4)

    for path_index, sequence in enumerate(topology["path_patch_sequence"]):
        selected_edges = topology["path_edge_indices"][path_index]
        np.testing.assert_array_equal(
            sequence[1::2], topology["edge_patch"][selected_edges]
        )
        coordinates = [patch_coordinate(index, 9) for index in sequence]
        assert all(
            max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
            for a, b in zip(coordinates[:-1], coordinates[1:])
        )


def test_generated_sample_keeps_optimal_path_patch_correspondence():
    sample = build_one(np.random.default_rng(2), v79_config())
    optimal_index = int(sample["optimal_path_idx"])
    selected_edges = np.flatnonzero(sample["path_mask"][optimal_index])
    np.testing.assert_array_equal(
        sample["optimal_edge_patches"], sample["edge_patch"][selected_edges]
    )
    np.testing.assert_array_equal(
        sample["optimal_path_patches"], sample["path_patch_sequence"][optimal_index]
    )
    assert sample["image"].shape == (288, 288, 3)
