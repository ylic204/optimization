"""Persist simulator-exported BEV records without camera-frame inputs."""

from pathlib import Path

import numpy as np

from bev_renderer_v85 import SEMANTIC_NAMES, STATIC_SEMANTIC_NAMES


INPUT_MODE = "ego_centric_bev_v85"
STATIC_INPUT_MODE = "ego_centric_static_bev_v85"


def save_static_bev_raw_record(output_path, rendered, config, goal_xy, task,
                               scene_id, metadata, evaluation, reference, logged_gt=None):
    """Static car task: numeric references are evaluator-only, not human GT.

    Only RGB, task specification and fixed geometry are model inputs. All
    masks, distances and reachability diagnostics are evaluator-only fields.
    """
    import json
    from static_road_planner_v85 import json_ready, task_cost

    rgb = np.asarray(rendered["bev_rgb"], dtype=np.uint8)
    semantic = np.asarray(rendered["bev_semantic"], dtype=np.uint8)
    if rgb.shape != (config.image_size, config.image_size, 3):
        raise ValueError("invalid static BEV RGB shape")
    if semantic.shape != (len(STATIC_SEMANTIC_NAMES), *rgb.shape[:2]):
        raise ValueError("invalid static semantic shape")
    for name in ("route", "reference", "history", "traffic_red", "traffic_yellow", "traffic_green"):
        if semantic[SEMANTIC_NAMES.index(name)].any():
            raise ValueError(f"static BEV must not contain {name}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gt_cost = task_cost(logged_gt["static_metrics"], task) if logged_gt is not None else np.inf
    if logged_gt is not None:
        if (not np.allclose(logged_gt["poses"][-1, :2], goal_xy, atol=1e-6)
                or len(logged_gt["times"]) != len(logged_gt["poses"])
                or (np.diff(logged_gt["times"]) <= 0).any()):
            raise ValueError("logged GT endpoint/timestamps must match the exported task")
    np.savez_compressed(
        output_path,
        raw_schema_version=np.int64(85), schema_revision=np.int64(6),
        input_mode=np.str_(STATIC_INPUT_MODE),
        planning_mode=np.str_("static_multitask"),
        metric_raster_layout=np.str_("bev_x_rows_y_cols"),
        bev_rgb=rgb, bev_semantic=semantic,
        bev_semantic_names=np.asarray(STATIC_SEMANTIC_NAMES, dtype="U32"),
        bev_config_json=np.str_(config.to_json()),
        region_world_bounds=rendered["region_world_bounds"],
        region_semantic_counts=rendered["region_semantic_counts"],
        map_bounds=config.bounds, start_xy=np.zeros(2, dtype=np.float32),
        goal_xy=np.asarray(goal_xy, dtype=np.float32),
        task_id=np.str_(task["id"]), task_text=np.str_(task["text"]),
        task_spec_json=np.str_(json.dumps(task, sort_keys=True)),
        scene_id=np.str_(scene_id), sample_id=np.str_(f"{scene_id}/{task['id']}"),
        trajectory_gt_available=np.bool_(logged_gt is not None),
        logged_gt_trajectory_ego=np.asarray(logged_gt["poses"] if logged_gt is not None else np.empty((0,3)), dtype=np.float64),
        logged_gt_time_s=np.asarray(logged_gt["times"] if logged_gt is not None else [], dtype=np.float64),
        logged_gt_source=np.str_(logged_gt["source"] if logged_gt is not None else "unavailable"),
        logged_gt_end_interpolated=np.bool_(logged_gt["end_interpolated"] if logged_gt is not None else False),
        logged_gt_boundary_inset_m=np.float64(logged_gt["boundary_inset_m"] if logged_gt is not None else np.nan),
        logged_gt_static_task_feasible=np.bool_(np.isfinite(gt_cost)),
        logged_gt_static_task_cost=np.float64(gt_cost),
        logged_gt_is_task_optimum=np.bool_(False),
        reference_solution_available=np.bool_(reference["status"] == "optimal_on_graph"),
        reference_status=np.str_(reference["status"]),
        reference_trajectory_ego=np.asarray(reference["poses"], dtype=np.float64),
        reference_edge_ids=np.asarray(reference["edge_ids"], dtype=np.int64),
        reference_cost=np.float64(reference["cost"] if reference["cost"] is not None else np.inf),
        reference_metrics_json=np.str_(json.dumps(
            json_ready(reference["metrics"]),
            sort_keys=True, allow_nan=False)),
        **{key: np.asarray(value) for key, value in metadata.items()},
        **{key: np.asarray(value) for key, value in evaluation.items()},
    )


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
    expert_time_s=None,
    planning_time_s=None,
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
    timing = {}
    if expert_time_s is not None and planning_time_s is not None:
        timing = dict(expert_time_s=np.asarray(expert_time_s, dtype=np.float64),
                      planning_time_s=np.asarray(planning_time_s, dtype=np.float64))
    elif expert_time_s is not None or planning_time_s is not None:
        raise ValueError("expert and planning timestamps must be supplied together")
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
        **timing,
    )
