"""Unified data construction for nuPlan and Habitat PointNav.

Raw exporters provide RGB, an ego-frame reference path, map rasters and,
preferably, a future expert/geodesic trajectory.  This module constructs a
variable candidate bank, signed-distance fields and solver supervision.
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from mapped_planning_schema_v82 import (
    TRAJECTORY_STEPS,
    _resample_polyline,
    trajectory_geometry_features,
    wrap_angle,
)
from optimization_spec_v83 import METRIC_NAMES


SCHEMA_VERSION = 84
SCHEMA_REVISION = 2
SDF_SIZE = 128
MIN_CANDIDATES = 8
MAX_CANDIDATES = 32


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


def adaptive_candidate_count(
    reference_path,
    traversable,
    min_candidates=MIN_CANDIDATES,
    max_candidates=MAX_CANDIDATES,
):
    """Allocate more proposals to curved or spatially constrained scenes.

    The returned count is a per-sample planning budget, not a learned label.
    It only uses route geometry and the known static traversability map.
    Dynamic obstacles are deliberately excluded to avoid leaking supervision.
    """
    center = _resample_polyline(reference_path, TRAJECTORY_STEPS)
    segment = np.linalg.norm(np.diff(center[:, :2], axis=0), axis=-1).clip(1e-3)
    curvature = np.abs(wrap_angle(np.diff(center[:, 2]))) / segment
    curvature_score = np.clip(float(curvature.mean()) / 0.18, 0.0, 1.0)
    obstacle_density = 1.0 - float(np.asarray(traversable, dtype=bool).mean())
    obstacle_score = np.clip(obstacle_density / 0.35, 0.0, 1.0)
    complexity = 0.55 * curvature_score + 0.45 * obstacle_score
    count = int(
        round(min_candidates + complexity * (max_candidates - min_candidates))
    )
    return int(np.clip(count, min_candidates, max_candidates))


def _progress_variant(center, power):
    """Resample a geometric path with a different longitudinal profile."""
    source = np.linspace(0.0, 1.0, len(center), dtype=np.float32)
    target = np.linspace(0.0, 1.0, len(center), dtype=np.float32) ** float(power)
    x = np.interp(target, source, center[:, 0])
    y = np.interp(target, source, center[:, 1])
    yaw = np.interp(target, source, np.unwrap(center[:, 2]))
    return np.stack([x, y, wrap_angle(yaw)], axis=-1).astype(np.float32)


def _frenet_variant(center, lateral_offset, terminal_heading, progress_power):
    """Build one continuous candidate with endpoint and heading diversity."""
    route = _progress_variant(center, progress_power)
    progress = np.linspace(0.0, 1.0, len(route), dtype=np.float32)
    length = np.linalg.norm(np.diff(route[:, :2], axis=0), axis=-1).sum()

    # Cubic lateral profile: d(0)=d'(0)=0, d(1)=offset and
    # d'(1)=tan(terminal_heading)*path_length.
    terminal_slope = np.tan(float(terminal_heading)) * max(float(length), 1e-3)
    a = terminal_slope - 2.0 * float(lateral_offset)
    b = 3.0 * float(lateral_offset) - terminal_slope
    lateral = a * progress**3 + b * progress**2
    x = route[:, 0] - np.sin(route[:, 2]) * lateral
    y = route[:, 1] + np.cos(route[:, 2]) * lateral
    yaw = np.arctan2(np.gradient(y), np.gradient(x))
    trajectory = np.stack([x, y, wrap_angle(yaw)], axis=-1).astype(np.float32)
    trajectory[0] = center[0]
    return trajectory


def _farthest_point_subset(candidates, count, center_index):
    """Select a deterministic, geometrically diverse candidate subset."""
    flat = candidates[:, :, :2].reshape(len(candidates), -1)
    selected = [int(center_index)]
    minimum_distance = np.linalg.norm(flat - flat[center_index], axis=-1)
    while len(selected) < min(int(count), len(candidates)):
        minimum_distance[selected] = -1.0
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = np.linalg.norm(flat - flat[next_index], axis=-1)
        minimum_distance = np.minimum(minimum_distance, distance)
    return candidates[np.asarray(selected, dtype=np.int64)]


def make_adaptive_candidates(
    reference_path,
    traversable,
    bounds,
    lateral_span,
    source_id,
    min_candidates=MIN_CANDIDATES,
    max_candidates=MAX_CANDIDATES,
):
    """Generate a variable-size local trajectory set from known map geometry.

    Proposals combine lateral endpoint, terminal-heading and longitudinal
    profile variations.  A static-map coverage test removes unusable proposals,
    then farthest-point sampling keeps a diverse per-scene subset.  The caller
    pads the result for batching; padding is never treated as a real candidate.
    """
    center = _resample_polyline(reference_path, TRAJECTORY_STEPS)
    heading_span = np.deg2rad(12.0 if int(source_id) == 0 else 30.0)
    offsets = np.linspace(-lateral_span, lateral_span, 13, dtype=np.float32)
    headings = np.asarray([-heading_span, 0.0, heading_span], dtype=np.float32)
    progress_powers = np.asarray([0.75, 1.0, 1.35], dtype=np.float32)

    configurations = [(0.0, 0.0, 1.0)]
    configurations.extend(
        (float(offset), float(heading), float(power))
        for offset in offsets
        for heading in headings
        for power in progress_powers
        if not (
            abs(float(offset)) < 1e-6
            and abs(float(heading)) < 1e-6
            and abs(float(power) - 1.0) < 1e-6
        )
    )
    pool = np.stack([
        _frenet_variant(center, *configuration)
        for configuration in configurations
    ])

    static_free, inside = sample_grid_nearest(
        np.asarray(traversable, dtype=bool), pool[..., :2], bounds
    )
    coverage = inside.mean(-1)
    static_fraction = (static_free & inside).mean(-1)
    keep = (coverage >= 0.75) & (static_fraction >= 0.25)
    keep[0] = coverage[0] >= 0.75
    viable_indices = np.flatnonzero(keep)
    if len(viable_indices) < min_candidates:
        quality = coverage + 0.25 * static_fraction
        supplement = np.argsort(-quality)
        viable_indices = np.asarray(
            list(dict.fromkeys([
                *viable_indices.tolist(),
                *supplement[:min_candidates].tolist(),
            ])),
            dtype=np.int64,
        )

    viable = pool[viable_indices]
    center_matches = np.flatnonzero(viable_indices == 0)
    center_index = int(center_matches[0]) if len(center_matches) else 0
    desired = adaptive_candidate_count(
        reference_path, traversable, min_candidates, max_candidates
    )
    return _farthest_point_subset(viable, desired, center_index)


def pad_candidates(candidates, metrics, max_candidates=MAX_CANDIDATES):
    """Pad a variable candidate set and return a structural validity mask."""
    count = min(len(candidates), int(max_candidates))
    padded_trajectory = np.zeros(
        (max_candidates, TRAJECTORY_STEPS, 3), dtype=np.float32
    )
    padded_metrics = np.zeros(
        (max_candidates, len(METRIC_NAMES)), dtype=np.float32
    )
    valid = np.zeros(max_candidates, dtype=bool)
    padded_trajectory[:count] = candidates[:count]
    padded_metrics[:count] = metrics[:count]
    valid[:count] = True
    return padded_trajectory, padded_metrics, valid, count


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


def process_raw_record(
    raw,
    lateral_span,
    min_candidates=MIN_CANDIDATES,
    max_candidates=MAX_CANDIDATES,
):
    """Convert one source-specific raw dictionary into the V8.4 schema."""
    bounds = np.asarray(raw["map_bounds"], dtype=np.float32)
    traversable = resize_binary_grid(raw["traversable"])
    dynamic_obstacle = resize_binary_grid(raw["dynamic_obstacle"])
    occupied = (~traversable) | dynamic_obstacle
    sdf = signed_distance_field(occupied, bounds)

    reference = np.asarray(raw["reference_path_ego"], dtype=np.float32)
    candidates = make_adaptive_candidates(
        reference,
        traversable,
        bounds,
        lateral_span=lateral_span,
        source_id=int(raw["source_id"].item()),
        min_candidates=min_candidates,
        max_candidates=max_candidates,
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
    candidates, metrics, candidate_valid, candidate_count = pad_candidates(
        candidates, metrics, max_candidates
    )

    expert_source = raw.get("expert_trajectory_ego", reference)
    expert_trajectory = _resample_polyline(
        np.asarray(expert_source, dtype=np.float32), TRAJECTORY_STEPS
    )[:, :2]

    dx, dy = np.asarray(raw["goal_xy"], dtype=np.float32)
    goal_state = np.asarray(
        [dx, dy, np.hypot(dx, dy), np.arctan2(dy, dx)], dtype=np.float32
    )
    return {
        "schema_version": np.int64(SCHEMA_VERSION),
        "schema_revision": np.int64(SCHEMA_REVISION),
        "sample_id": np.str_(raw["sample_id"].item()),
        "task_text": np.str_(raw["task_text"].item()),
        "source_id": np.int64(raw["source_id"].item()),
        "image": np.asarray(raw["image"], dtype=np.uint8),
        "candidate_trajectories": candidates,
        "candidate_features": trajectory_geometry_features(candidates),
        "candidate_metrics": metrics,
        "candidate_valid": candidate_valid,
        "candidate_count": np.int64(candidate_count),
        "expert_trajectory": expert_trajectory.astype(np.float32),
        "expert_is_fallback": np.bool_("expert_trajectory_ego" not in raw),
        "goal_state": goal_state,
        "ego_state": np.asarray(raw["ego_state"], dtype=np.float32),
        "sdf": sdf,
        "map_bounds": bounds,
        "reference_path_ego": reference,
    }


def convert_directory(
    raw_root,
    output_root,
    source,
    min_candidates=MIN_CANDIDATES,
    max_candidates=MAX_CANDIDATES,
):
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    lateral_span = 2.5 if source == "nuplan" else 0.8
    for raw_path in sorted(raw_root.rglob("*.npz")):
        with np.load(raw_path, allow_pickle=False) as data:
            raw = {key: data[key] for key in data.files}
        sample = process_raw_record(
            raw,
            lateral_span,
            min_candidates=min_candidates,
            max_candidates=max_candidates,
        )
        output_path = output_root / raw_path.relative_to(raw_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **sample)
