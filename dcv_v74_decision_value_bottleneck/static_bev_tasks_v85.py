"""Task-independent geometry and task definitions for static nuPlan snapshots.

Reachability selects a task goal, never a reference path. Numeric scene masks
are evaluator data: do not forward this module's outputs to the frozen VLM.
"""

from collections import deque
import hashlib

import numpy as np
from scipy.ndimage import distance_transform_edt

from bev_renderer_v85 import (
    SEMANTIC_INDEX, ego_to_pixel, pixel_to_ego, oriented_box_corners,
)


CONSTRUCTION_TYPES = {"traffic_cone", "barrier", "czone_sign"}
VISUAL_LEGEND = (
    "BEV: orange S is the start, gold star G is the goal; blue is traversable "
    "road, white background is non-traversable; blue boxes are stationary "
    "vehicles, brown boxes pedestrians, pink boxes bicycles, black boxes "
    "static obstacles, peach areas construction-derived hazard zones. "
    "All objects are frozen at their current positions. Up is +x (forward), "
    "left is +y. Gray lane centerlines are map context, not a planned path."
)


def hazard_polygons(agents, buffer_m):
    """Operational annotation around construction objects, not nuPlan risk GT."""
    if not np.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("hazard buffer must be finite and nonnegative")
    return [
        oriented_box_corners(a["center"], a["heading"],
                             max(a["length"], .8) + 2 * buffer_m,
                             max(a["width"], .6) + 2 * buffer_m)
        for a in agents if a["type"] in CONSTRUCTION_TYPES
    ]


def reachable_cells(free, start_rc):
    """4-connected reachability: no diagonal crossing through obstacle corners."""
    reached = np.zeros_like(free, dtype=bool)
    row, col = start_rc
    if not (0 <= row < free.shape[0] and 0 <= col < free.shape[1]) or not free[row, col]:
        return reached
    reached[row, col] = True
    queue = deque([(row, col)])
    while queue:
        row, col = queue.popleft()
        for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if (0 <= r < free.shape[0] and 0 <= c < free.shape[1]
                    and free[r, c] and not reached[r, c]):
                reached[r, c] = True
                queue.append((r, c))
    return reached


def static_evaluation_layers(rendered, config, clearance_m):
    if not np.isfinite(clearance_m) or clearance_m <= 0:
        raise ValueError("agent radius must be finite and positive")
    semantic = rendered["bev_semantic"].astype(bool)
    road = rendered["traversable"].astype(bool)
    occupied = rendered["dynamic_obstacle"].astype(bool)
    step = (config.x_max - config.x_min) / (config.image_size - 1)
    # Padding treats the viewport boundary as blocked, not an escape route.
    raw_free = road & ~occupied
    distance = distance_transform_edt(np.pad(raw_free, 1))[1:-1, 1:-1] * step
    free = raw_free & (distance > clearance_m)
    hazard = semantic[-1]
    if hazard.any():
        hazard_clearance = distance_transform_edt(~hazard) * step
        hazard_exclusion = hazard_clearance <= clearance_m
    else:
        hazard_exclusion = np.zeros_like(hazard)
    pedestrian = semantic[SEMANTIC_INDEX["pedestrian"]]
    pedestrian_distance = (distance_transform_edt(~pedestrian) * step if pedestrian.any()
                           else np.full(road.shape, np.inf))
    col, row = np.rint(ego_to_pixel([[0, 0]], config)[0]).astype(int)
    reachable = reachable_cells(free, (row, col))
    safe_reachable = reachable_cells(free & ~hazard_exclusion, (row, col))
    return dict(
        traversable=road, static_obstacle=occupied, planning_free=free,
        hazard_mask=hazard, hazard_exclusion=hazard_exclusion,
        pedestrian_mask=pedestrian,
        pedestrian_distance_m=pedestrian_distance.astype(np.float32),
        reachable=reachable, hazard_free_reachable=safe_reachable,
        agent_radius_m=np.float32(clearance_m),
        has_pedestrians=np.bool_(pedestrian.any()), has_hazards=np.bool_((hazard & road).any()),
    )


def select_goal(evaluation, config, scene_id, args):
    """Select a visible reachable goal from the current raster, independent of logs."""
    reachable = evaluation["reachable"]
    if "avoid_hazards" in args.tasks:
        reachable = reachable & evaluation["hazard_free_reachable"]
    margin = args.goal_margin_m
    if args.goal_xy is not None:
        xy = np.asarray(args.goal_xy, dtype=np.float32)
        if (not np.isfinite(xy).all() or not
                (config.x_min + margin <= xy[0] <= config.x_max - margin and
                 config.y_min + margin <= xy[1] <= config.y_max - margin)):
            return None
        col, row = np.rint(ego_to_pixel([xy], config)[0]).astype(int)
        if not reachable[row, col] or np.linalg.norm(xy) < args.goal_min_distance:
            return None
        return xy
    rows, cols = np.nonzero(reachable)
    if not len(rows):
        return None
    xy = pixel_to_ego(np.column_stack([cols, rows]), config)
    distance = np.linalg.norm(xy, axis=1)
    valid = ((distance >= args.goal_min_distance) & (distance <= args.goal_max_distance)
             & (xy[:, 0] >= config.x_min + margin) & (xy[:, 0] <= config.x_max - margin)
             & (xy[:, 1] >= config.y_min + margin) & (xy[:, 1] <= config.y_max - margin))
    candidates = xy[valid]
    if not len(candidates):
        return None
    digest = hashlib.sha256(f"{args.goal_seed}:{scene_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return candidates[int(rng.integers(len(candidates)))].astype(np.float32)


def task_variants(goal, args, evaluation):
    goal_text = f"G=({goal[0]:.2f}, {goal[1]:.2f}) meters"
    common = (f"Plan a static path from S=(0,0) to {goal_text}. Stay on traversable "
              f"road and avoid every object with agent radius {args.agent_radius_m:g} m. ")
    # Constant nominal speed makes minimum travel time equivalent to minimum length.
    base = dict(goal_tolerance_m=.75, nominal_speed_mps=args.nominal_speed,
                agent_radius_m=args.agent_radius_m, length_weight=1.,
                pedestrian_weight=0., pedestrian_scale_m=args.pedestrian_scale_m,
                forbid_hazards=False)
    variants = [
        dict(base, id="shortest", objective="travel_time",
             text=common + f"Minimize travel time at constant nominal speed {args.nominal_speed:g} m/s.",
             task_applicable=True),
        dict(base, id="pedestrian_clearance", objective="length_plus_pedestrian_exposure",
             pedestrian_weight=args.pedestrian_weight,
             text=common + "Prefer paths farther from pedestrians. Minimize the path integral "
                  f"of 1 + {args.pedestrian_weight:g} * exp(-d/{args.pedestrian_scale_m:g}), "
                  "where d is distance in meters to the nearest pedestrian footprint.",
             task_applicable=bool(evaluation["has_pedestrians"])),
        dict(base, id="avoid_hazards", objective="length", forbid_hazards=True,
             text=common + "Do not enter the peach hazard areas; maintain the agent-radius "
                  "clearance from them. Among feasible paths minimize path length.",
             task_applicable=bool(evaluation["has_hazards"])),
    ]
    for task in variants:
        task["text"] += " " + VISUAL_LEGEND
    return [task for task in variants if task["id"] in args.tasks]
