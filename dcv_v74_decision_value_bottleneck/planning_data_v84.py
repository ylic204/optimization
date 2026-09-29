"""Unified data construction for nuPlan and Habitat PointNav.

Raw exporters only need to provide an RGB image, an ego-frame reference path,
a traversability raster and a dynamic-obstacle raster.  This module constructs
candidate trajectories, signed-distance fields and solver supervision.
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from mapped_planning_schema_v82 import (
    N_CANDIDATES,
    TRAJECTORY_STEPS,
    make_frenet_candidates,
    trajectory_geometry_features,
    wrap_angle,
)
from optimization_spec_v83 import METRIC_NAMES


SCHEMA_VERSION = 84
SDF_SIZE = 128


def resize_binary_grid(grid, output_size=SDF_SIZE):
    """Nearest-neighbor resize without importing a simulator-specific package."""
    from PIL import Image

    image = Image.fromarray(np.asarray(grid, dtype=np.uint8) * 255)
    image = image.resize((output_size, output_size), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def signed_distance_field(occupied, bounds):
    """Positive distance in free space, negative distance inside obstacles."""
    occupied = np.asarray(occupied, dtype=bool)
    xmin, xmax, ymin, ymax = np.asarray(bounds, dtype=np.float32)
    resolution_x = (xmax - xmin) / occupied.shape[1]
    resolution_y = (ymax - ymin) / occupied.shape[0]
    sampling = (resolution_y, resolution_x)
    outside = distance_transform_edt(~occupied, sampling=sampling)
    inside = distance_transform_edt(occupied, sampling=sampling)
    return (outside - inside).astype(np.float32)


def sample_grid_nearest(grid, xy, bounds):
    """Sample an ego-frame raster at x-forward/y-left trajectory points."""
    xmin, xmax, ymin, ymax = np.asarray(bounds, dtype=np.float32)
    x = np.asarray(xy)[..., 0]
    y = np.asarray(xy)[..., 1]
    col = np.rint((x - xmin) / (xmax - xmin) * (grid.shape[1] - 1)).astype(int)
    row = np.rint((ymax - y) / (ymax - ymin) * (grid.shape[0] - 1)).astype(int)
    inside = (row >= 0) & (row < grid.shape[0]) & (col >= 0) & (col < grid.shape[1])
    row = np.clip(row, 0, grid.shape[0] - 1)
    col = np.clip(col, 0, grid.shape[1] - 1)
    values = grid[row, col]
    return values, inside


def candidate_metric_targets(
    candidates,
    reference_path,
    traversable,
    dynamic_obstacle,
    sdf,
    bounds,
    goal_xy,
):
    """Compute the eight normalized lower-is-better planning metrics."""
    xy = candidates[..., :2]
    traversable_value, inside = sample_grid_nearest(traversable, xy, bounds)
    obstacle_value, _ = sample_grid_nearest(dynamic_obstacle, xy, bounds)
    clearance, _ = sample_grid_nearest(sdf, xy, bounds)

    collision = obstacle_value.mean(-1)
    non_traversable = ((~traversable_value) | (~inside)).mean(-1)
    safety_risk = np.exp(-np.clip(clearance, 0.0, 5.0)).mean(-1)

    reference_xy = np.asarray(reference_path, dtype=np.float32)[:, :2]
    distance_to_route = np.linalg.norm(
        xy[:, :, None, :] - reference_xy[None, None, :, :], axis=-1
    ).min(-1)
    route_deviation = np.clip(distance_to_route.mean(-1) / 5.0, 0.0, 1.0)

    goal_xy = np.asarray(goal_xy, dtype=np.float32)
    initial_goal_distance = max(float(np.linalg.norm(goal_xy)), 1e-3)
    final_goal_distance = np.linalg.norm(xy[:, -1] - goal_xy[None], axis=-1)
    progress = np.clip(
        (initial_goal_distance - final_goal_distance) / initial_goal_distance,
        0.0,
        1.0,
    )
    lack_of_progress = 1.0 - progress
    goal_error = np.clip(final_goal_distance / initial_goal_distance, 0.0, 1.0)

    segment = np.linalg.norm(np.diff(xy, axis=1), axis=-1)
    path_length = np.clip(segment.sum(-1) / 50.0, 0.0, 1.0)
    dyaw = np.abs(wrap_angle(np.diff(candidates[..., 2], axis=1)))
    curvature = dyaw / np.clip(segment, 1e-3, None)
    discomfort = np.clip(curvature.mean(-1) / 0.5, 0.0, 1.0)

    values = {
        "collision": collision,
        "non_traversable": non_traversable,
        "safety_risk": safety_risk,
        "route_deviation": route_deviation,
        "lack_of_progress": lack_of_progress,
        "path_length": path_length,
        "discomfort": discomfort,
        "goal_error": goal_error,
    }
    return np.stack([values[name] for name in METRIC_NAMES], axis=-1).astype(
        np.float32
    )


def process_raw_record(raw, lateral_span):
    """Convert one source-specific raw dictionary into the V8.4 schema."""
    bounds = np.asarray(raw["map_bounds"], dtype=np.float32)
    traversable = resize_binary_grid(raw["traversable"])
    dynamic_obstacle = resize_binary_grid(raw["dynamic_obstacle"])
    occupied = (~traversable) | dynamic_obstacle
    sdf = signed_distance_field(occupied, bounds)

    reference = np.asarray(raw["reference_path_ego"], dtype=np.float32)
    candidates = make_frenet_candidates(
        reference,
        n_candidates=N_CANDIDATES,
        steps=TRAJECTORY_STEPS,
        lateral_span=lateral_span,
    )
    metrics = candidate_metric_targets(
        candidates,
        reference,
        traversable,
        dynamic_obstacle,
        sdf,
        bounds,
        raw["goal_xy"],
    )
    # Validity describes whether a candidate is represented by this local map.
    # Collision and non-traversability stay as solver costs/constraints rather
    # than being leaked into this structural mask.
    _, inside = sample_grid_nearest(
        traversable, candidates[..., :2], bounds
    )
    candidate_valid = inside.mean(-1) >= 0.75

    dx, dy = np.asarray(raw["goal_xy"], dtype=np.float32)
    goal_state = np.asarray(
        [dx, dy, np.hypot(dx, dy), np.arctan2(dy, dx)], dtype=np.float32
    )
    return {
        "schema_version": np.int64(SCHEMA_VERSION),
        "sample_id": np.str_(raw["sample_id"].item()),
        "task_text": np.str_(raw["task_text"].item()),
        "source_id": np.int64(raw["source_id"].item()),
        "image": np.asarray(raw["image"], dtype=np.uint8),
        "candidate_trajectories": candidates,
        "candidate_features": trajectory_geometry_features(candidates),
        "candidate_metrics": metrics,
        "candidate_valid": candidate_valid,
        "goal_state": goal_state,
        "ego_state": np.asarray(raw["ego_state"], dtype=np.float32),
        "sdf": sdf,
        "map_bounds": bounds,
        "reference_path_ego": reference,
    }


def convert_directory(raw_root, output_root, source):
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    lateral_span = 2.5 if source == "nuplan" else 0.8
    for raw_path in sorted(raw_root.rglob("*.npz")):
        with np.load(raw_path, allow_pickle=False) as data:
            raw = {key: data[key] for key in data.files}
        sample = process_raw_record(raw, lateral_span)
        output_path = output_root / raw_path.relative_to(raw_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **sample)
