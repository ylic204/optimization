"""Evaluate a static [N,3] rear-axle (x,y,heading) path using full scene geometry.

This evaluator has no VLM dependency. Do not forward its labels to the selector.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from bev_renderer_v85 import wrap_angle
from raw_record_bev_v85 import STATIC_INPUT_MODE
from static_road_planner_v85 import RoadScene, json_ready, task_cost


MODEL_INPUT_KEYS = (
    "bev_rgb", "task_text", "task_spec_json", "map_bounds", "bev_config_json",
    "region_world_bounds", "start_pose", "goal_pose", "vehicle_geometry_json",
)


def model_inputs(record):
    """Explicit allowlist: no masks, graph, future states or numeric references."""
    if str(record["input_mode"].item()) != STATIC_INPUT_MODE or int(record["schema_revision"]) != 5:
        raise ValueError("expected static BEV schema85 revision5; re-export older records")
    return {key: np.asarray(record[key]).copy() for key in MODEL_INPUT_KEYS}


def evaluate_record(record_path, trajectory):
    record_path = Path(record_path)
    with np.load(record_path, allow_pickle=False) as record:
        model_inputs(record)  # validate contract
        task = json.loads(str(record["task_spec_json"].item()))
        goal = record["goal_pose"].copy()
        reference_cost = float(record["reference_cost"])
        reference_status = str(record["reference_status"].item())
    scene = RoadScene.from_dict(json.loads((record_path.parent/"audit/scene_geometry.json").read_text()))
    trajectory = np.asarray(trajectory, dtype=float)
    metrics = scene.assess(trajectory)
    valid = (trajectory.ndim == 2 and trajectory.shape[1] == 3 and len(trajectory) >= 2
             and np.isfinite(trajectory).all())
    start_ok = bool(valid and np.linalg.norm(trajectory[0,:2]) < 1e-4
                    and abs(float(wrap_angle(trajectory[0,2]))) < 1e-4)
    goal_ok = bool(valid and np.linalg.norm(trajectory[-1,:2]-goal[:2]) <= task["goal_tolerance_m"]
                   and abs(float(wrap_angle(trajectory[-1,2]-goal[2]))) <= task["goal_heading_tolerance_rad"])
    cost = task_cost(metrics, task)
    success = bool(start_ok and goal_ok and np.isfinite(cost))
    reason = (metrics["reason"] if not metrics["feasible"] else
              "wrong_start" if not start_ok else "goal_not_reached" if not goal_ok else
              "hazard_violation" if not np.isfinite(cost) else "ok")
    return json_ready(dict(success=success, reason=reason, start_ok=start_ok, goal_ok=goal_ok,
                           metrics=metrics, cost=cost if success else None,
                           reference_status=reference_status, reference_cost=reference_cost,
                           cost_gap_to_graph_reference=(cost-reference_cost)
                           if success and np.isfinite(reference_cost) else None,
                           reference_scope="finite graph; a continuous path may have lower cost"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    parser.add_argument("--trajectory", help=".npy array [N,3] in ego-frame meters/radians")
    args = parser.parse_args()
    if args.trajectory:
        trajectory = np.load(args.trajectory, allow_pickle=False)
    else:
        with np.load(args.record, allow_pickle=False) as record:
            trajectory = record["reference_trajectory_ego"]
    print(json.dumps(evaluate_record(args.record, trajectory), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
