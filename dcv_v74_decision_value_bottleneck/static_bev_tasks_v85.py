"""Public task specifications and evaluator-only raster diagnostics."""
import numpy as np
from scipy.ndimage import distance_transform_edt
from bev_renderer_v85 import SEMANTIC_INDEX, oriented_box_corners

CONSTRUCTION_TYPES = {"traffic_cone", "barrier", "czone_sign"}
VISUAL_LEGEND = (
    "BEV: orange vehicle with a white arrow is the start; the gold star is the goal. "
    "Pale blue is road; off-road background is non-traversable. Blue boxes are frozen "
    "vehicles, brown boxes pedestrians, pink boxes bicycles, black boxes obstacles, "
    "peach areas construction-derived hazard zones. Gray arrows show the direction "
    "of all lanes, not a selected route. Up is +x (forward), left is +y. "
    "No reference solution or actor history is shown."
)


def hazard_polygons(agents, buffer_m):
    """Operational annotation, not nuPlan risk GT."""
    if not np.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("hazard buffer must be finite and nonnegative")
    return [oriented_box_corners(a["center"], a["heading"],
                                max(a["length"], .8)+2*buffer_m,
                                max(a["width"], .6)+2*buffer_m)
            for a in agents if a["type"] in CONSTRUCTION_TYPES]


def static_evaluation_layers(rendered, config):
    """Raster labels for audit, not a vehicle configuration-space/free-space mask."""
    semantic = rendered["bev_semantic"].astype(bool)
    road = rendered["traversable"].astype(bool)
    pedestrian = semantic[SEMANTIC_INDEX["pedestrian"]]
    step = (config.x_max-config.x_min)/(config.image_size-1)
    distance = (distance_transform_edt(~pedestrian)*step if pedestrian.any()
                else np.full(road.shape, np.inf))
    hazard = semantic[-1]
    return dict(traversable=road, static_obstacle=rendered["dynamic_obstacle"].astype(bool),
                hazard_mask=hazard, pedestrian_mask=pedestrian,
                pedestrian_distance_m=distance.astype(np.float32),
                has_pedestrians=np.bool_(pedestrian.any()),
                has_hazards=np.bool_((hazard & road).any()))


def task_variants(goal_pose, args, evaluation, vehicle):
    goal = np.asarray(goal_pose)
    common = (f"Plan a forward-only static vehicle path from rear-axle pose (0,0,0) to "
              f"({goal[0]:.6f},{goal[1]:.6f},{goal[2]:.6f}) (meters, meters, radians). "
              f"Vehicle length {vehicle['length']:.6f} m, width {vehicle['width']:.6f} m, "
              f"rear-axle-to-center {vehicle['rear_axle_to_center']:.6f} m. "
              f"Keep the entire body plus {args.safety_margin_m:g} m safety margin on road "
              f"and clear of all objects. Minimum turning radius is {args.min_turn_radius_m:g} m. "
              "Follow lane directions and explicit outgoing lane connections. "
              + ("Curved lane changes are allowed only between directly adjacent lanes. "
                 if args.lane_change_policy == "adjacent" else "No lateral lane changes. ")
              + "No reversing or sideways translation. ")
    base = dict(goal_tolerance_m=.75, goal_heading_tolerance_rad=.15,
                nominal_speed_mps=args.nominal_speed, vehicle=dict(vehicle),
                min_turn_radius_m=args.min_turn_radius_m, safety_margin_m=args.safety_margin_m,
                forward_only=True, lane_change_policy=args.lane_change_policy,
                length_weight=1., pedestrian_weight=0., pedestrian_scale_m=args.pedestrian_scale_m,
                forbid_hazards=False)
    variants = [
        dict(base, id="shortest", objective="travel_time",
             text=common + f"Minimize length / {args.nominal_speed:g} at constant nominal speed; this is a geometric path task.",
             task_applicable=True),
        dict(base, id="pedestrian_clearance", objective="length_plus_pedestrian_exposure",
             pedestrian_weight=args.pedestrian_weight,
             text=common + f"Minimize the path integral of 1 + {args.pedestrian_weight:g} * "
                  f"exp(-d/{args.pedestrian_scale_m:g}), where d is clearance in meters from "
                  "the safety-inflated vehicle body to the nearest pedestrian footprint.",
             task_applicable=bool(evaluation["has_pedestrians"])),
        dict(base, id="avoid_hazards", objective="length", forbid_hazards=True,
             text=common + "Keep the entire safety-inflated body outside peach hazard zones. Minimize length.",
             task_applicable=bool(evaluation["has_hazards"])),
    ]
    for task in variants:
        task["text"] += " " + VISUAL_LEGEND
    return [t for t in variants if t["id"] in args.tasks]
