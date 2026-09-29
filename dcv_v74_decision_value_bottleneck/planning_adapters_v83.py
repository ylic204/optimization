"""Small adapters called inside nuPlan and Habitat data-export environments."""

import numpy as np

from mapped_optimization_schema_v83 import make_sample, save_sample


NUPLAN_TEXT = (
    "Drive safely to the route goal. Avoid collisions and non-drivable areas, "
    "follow the mapped route, make progress, and keep the ride comfortable."
)

POINTNAV_TEXT = (
    "Reach the PointGoal through navigable space. Avoid collisions, minimize "
    "remaining geodesic error, and prefer a short path."
)


def relative_goal(dx, dy):
    return np.asarray(
        [dx, dy, np.hypot(dx, dy), np.arctan2(dy, dx)], dtype=np.float32
    )


def export_nuplan(
    output_path,
    front_rgb,
    reference_route_ego,
    normalized_metrics,
    candidate_valid,
    goal_xy_ego,
    ego_state,
    sample_id,
):
    """Write a nuPlan frame after the devkit evaluates eleven candidates."""
    sample = make_sample(
        image=front_rgb,
        task_text=NUPLAN_TEXT,
        source_id=0,
        reference_path_ego=reference_route_ego,
        metric_dict=normalized_metrics,
        goal_state=relative_goal(*goal_xy_ego),
        ego_state=ego_state,
        candidate_valid=candidate_valid,
        sample_id=sample_id,
        lateral_span=2.5,
    )
    save_sample(output_path, sample)


def export_pointnav(
    output_path,
    rgb,
    reference_path_ego,
    normalized_metrics,
    candidate_valid,
    pointgoal_xy,
    ego_state,
    sample_id,
):
    """Write a Habitat frame after navmesh candidate evaluation."""
    sample = make_sample(
        image=rgb,
        task_text=POINTNAV_TEXT,
        source_id=1,
        reference_path_ego=reference_path_ego,
        metric_dict=normalized_metrics,
        goal_state=relative_goal(*pointgoal_xy),
        ego_state=ego_state,
        candidate_valid=candidate_valid,
        sample_id=sample_id,
        lateral_span=0.8,
    )
    save_sample(output_path, sample)
