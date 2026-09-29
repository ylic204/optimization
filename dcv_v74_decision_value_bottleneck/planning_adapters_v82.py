"""Dataset-side adapters into the common V8.2 planning sample.

These helpers deliberately do not import nuPlan or Habitat so each simulator
can keep its own pinned environment.  Simulator-specific scripts pass arrays
and metric results into these functions and write portable NPZ files.
"""

import numpy as np

from mapped_planning_schema_v82 import (
    SOURCE_NUPLAN,
    SOURCE_POINTNAV,
    build_sample,
    make_frenet_candidates,
    save_sample,
)


NUPLAN_TASK_TEXT = (
    "Choose a safe, drivable, comfortable trajectory that follows the mapped "
    "route and makes progress toward the navigation goal."
)

POINTNAV_TASK_TEXT = (
    "Choose a collision-free path that follows the known navigable map and "
    "reaches the PointGoal with minimum remaining geodesic distance."
)


def goal_state_from_xy(dx, dy):
    distance = float(np.hypot(dx, dy))
    bearing = float(np.arctan2(dy, dx))
    return np.asarray([dx, dy, distance, bearing], dtype=np.float32)


def weighted_task_cost(components, weights):
    """Combine named simulator metrics into one lower-is-better task cost."""
    unknown = sorted(set(weights) - set(components))
    if unknown:
        raise KeyError(f"cost weights have no component arrays: {unknown}")
    total = None
    for name, weight in weights.items():
        value = np.asarray(components[name], dtype=np.float32)
        total = value * float(weight) if total is None else total + value * float(weight)
    if total is None:
        raise ValueError("at least one cost component is required")
    return total.astype(np.float32)


def export_nuplan_sample(
    output_path,
    front_rgb,
    reference_route_ego,
    cost_components,
    ego_state,
    goal_xy_ego,
    sample_id,
    valid=None,
    lateral_span=2.5,
    cost_weights=None,
):
    """Export one nuPlan log frame after candidate metric evaluation.

    Recommended components are collision, off_drivable_area, wrong_way, TTC,
    route_deviation, lack_of_progress, speed_error and discomfort.  Values
    should be penalties where lower is better.
    """
    if cost_weights is None:
        cost_weights = {
            "collision": 20.0,
            "off_drivable_area": 10.0,
            "wrong_way": 5.0,
            "ttc": 3.0,
            "route_deviation": 2.0,
            "lack_of_progress": 1.0,
            "discomfort": 0.2,
        }
    candidates = make_frenet_candidates(
        reference_route_ego, lateral_span=lateral_span
    )
    costs = weighted_task_cost(cost_components, cost_weights)
    sample = build_sample(
        image=front_rgb,
        task_text=NUPLAN_TASK_TEXT,
        source_id=SOURCE_NUPLAN,
        candidate_trajectories=candidates,
        candidate_costs=costs,
        candidate_valid=valid,
        goal_state=goal_state_from_xy(*goal_xy_ego),
        ego_state=ego_state,
        sample_id=sample_id,
    )
    return save_sample(output_path, sample)


def export_pointnav_sample(
    output_path,
    rgb,
    reference_path_ego,
    cost_components,
    ego_state,
    pointgoal_xy,
    sample_id,
    valid=None,
    lateral_span=0.8,
    cost_weights=None,
):
    """Export one Habitat PointNav observation after navmesh rollouts.

    The ``pointgoal_xy`` should be the current local goal updated by VO when
    GPS+Compass is disabled.  Candidate costs come from Habitat/navmesh
    rollouts and remain training labels only.
    """
    if cost_weights is None:
        cost_weights = {
            "collision": 10.0,
            "remaining_geodesic": 1.0,
            "path_length": 0.1,
            "goal_miss": 5.0,
        }
    candidates = make_frenet_candidates(
        reference_path_ego, lateral_span=lateral_span
    )
    costs = weighted_task_cost(cost_components, cost_weights)
    sample = build_sample(
        image=rgb,
        task_text=POINTNAV_TASK_TEXT,
        source_id=SOURCE_POINTNAV,
        candidate_trajectories=candidates,
        candidate_costs=costs,
        candidate_valid=valid,
        goal_state=goal_state_from_xy(*pointgoal_xy),
        ego_state=ego_state,
        sample_id=sample_id,
    )
    return save_sample(output_path, sample)
