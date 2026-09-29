import numpy as np

from generate_camera_dataset_v81 import (
    IMAGE_SIDE,
    N_TRAJECTORIES,
    SELECTOR_GRID,
    TRAJECTORY_SAMPLES,
    build_one,
)


def test_first_person_sample_shapes_and_labels():
    sample = build_one(np.random.default_rng(23))
    regions = SELECTOR_GRID**2
    assert sample["rgb"].shape == (IMAGE_SIDE, IMAGE_SIDE, 3)
    assert np.array_equal(sample["rgb"], sample["image"])
    assert sample["depth_gt"].shape == (IMAGE_SIDE, IMAGE_SIDE)
    assert sample["semantic_gt"].shape == (IMAGE_SIDE, IMAGE_SIDE)
    assert sample["trajectory_points_robot"].shape == (
        N_TRAJECTORIES,
        TRAJECTORY_SAMPLES,
        2,
    )
    assert sample["trajectory_costs"].shape == (N_TRAJECTORIES,)
    assert sample["candidate_path_patch_relation"].shape == (
        N_TRAJECTORIES,
        regions,
    )
    assert sample["trajectory_region_features"].shape == (
        N_TRAJECTORIES,
        regions,
        4,
    )
    assert str(sample["student_input_mode"]) == "first_person_rgb_only"


def test_motion_primitives_are_continuous_and_optimum_is_feasible():
    sample = build_one(np.random.default_rng(41))
    cell_delta = np.abs(np.diff(sample["trajectory_cells"], axis=1))
    assert int(cell_delta.max()) <= 2
    features = sample["trajectory_features"]
    assert np.count_nonzero(features[:, 3] == 0.0) >= 2
    assert int(sample["optimal_path_idx"]) == int(
        np.argmin(sample["trajectory_costs"])
    )


def test_camera_to_bev_mapping_has_visible_regions():
    sample = build_one(np.random.default_rng(7))
    valid = sample["patch_to_bev_valid"]
    assert np.count_nonzero(valid > 0.0) > SELECTOR_GRID
    relation = sample["candidate_path_patch_relation"]
    assert np.isfinite(relation).all()
    assert np.count_nonzero(relation) > 0
