"""Persist simulator-exported BEV records without camera-frame inputs."""

from pathlib import Path

import numpy as np

from bev_renderer_v85 import SEMANTIC_NAMES


INPUT_MODE = "ego_centric_bev_v85"


def save_bev_raw_record(
    output_path,
    bev_rgb,
    bev_semantic,
    region_world_bounds,
    region_semantic_counts,
    reference_path_ego,
    traversable,
    dynamic_obstacle,
    map_bounds,
    goal_xy,
    ego_state,
    task_text,
    sample_id,
    expert_trajectory_ego,
    bev_config_json,
    scene_type="unknown",
    log_name="",
    scenario_token="",
    timestamp_us=0,
    iteration=0,
):
    """Save a nuPlan BEV sample; no first-person image is accepted."""
    bev_rgb = np.asarray(bev_rgb, dtype=np.uint8)
    bev_semantic = np.asarray(bev_semantic, dtype=np.uint8)
    if bev_rgb.ndim != 3 or bev_rgb.shape[-1] != 3:
        raise ValueError("bev_rgb must have shape [H,W,3]")
    if bev_semantic.shape != (len(SEMANTIC_NAMES), *bev_rgb.shape[:2]):
        raise ValueError("bev_semantic shape does not match the BEV RGB")
    if np.asarray(traversable).shape != bev_rgb.shape[:2]:
        raise ValueError("traversable must match the BEV spatial shape")
    if np.asarray(dynamic_obstacle).shape != bev_rgb.shape[:2]:
        raise ValueError("dynamic_obstacle must match the BEV spatial shape")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        raw_schema_version=np.int64(85),
        input_mode=np.str_(INPUT_MODE),
        bev_rgb=bev_rgb,
        bev_semantic=bev_semantic,
        bev_semantic_names=np.asarray(SEMANTIC_NAMES, dtype="U32"),
        bev_config_json=np.str_(bev_config_json),
        region_world_bounds=np.asarray(region_world_bounds, dtype=np.float32),
        region_semantic_counts=np.asarray(
            region_semantic_counts, dtype=np.float32
        ),
        reference_path_ego=np.asarray(reference_path_ego, dtype=np.float32),
        traversable=np.asarray(traversable, dtype=bool),
        dynamic_obstacle=np.asarray(dynamic_obstacle, dtype=bool),
        map_bounds=np.asarray(map_bounds, dtype=np.float32),
        goal_xy=np.asarray(goal_xy, dtype=np.float32),
        ego_state=np.asarray(ego_state, dtype=np.float32),
        task_text=np.str_(task_text),
        source_id=np.int64(0),
        sample_id=np.str_(sample_id),
        expert_trajectory_ego=np.asarray(
            expert_trajectory_ego, dtype=np.float32
        ),
        scene_type=np.str_(scene_type),
        log_name=np.str_(log_name),
        scenario_token=np.str_(scenario_token),
        timestamp_us=np.int64(timestamp_us),
        iteration=np.int64(iteration),
    )
