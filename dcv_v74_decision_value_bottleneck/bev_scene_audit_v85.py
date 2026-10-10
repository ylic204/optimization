"""Evaluator-only object, scene and reference diagnostics; never filter samples.

Distances identify spatial proximity, not causal decision importance. The model
input allowlist deliberately excludes all files produced by this module.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union

from bev_renderer_v85 import AGENT_LAYERS, agent_layer, oriented_box_corners, wrap_angle
from static_bev_tasks_v85 import CONSTRUCTION_TYPES
from static_road_planner_v85 import json_ready, write_json

CATEGORIES = ("vehicle", "pedestrian", "bicycle", "static_obstacle")


def _counts(objects):
    counts = Counter(a["category"] for a in objects)
    return {key: counts[key] for key in CATEGORIES}


def _reference_sweep(scene, poses):
    """Same interpolation and conservative envelopes as the static evaluator."""
    poses = np.asarray(poses)
    dense = [poses[0]]
    for a, b in zip(poses[:-1], poses[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b[:2]-a[:2])/scene.config.collision_step_m)))
        yaw = float(wrap_angle(b[2]-a[2]))
        for t in np.arange(1, n+1)/n:
            dense.append(np.r_[a[:2]+t*(b[:2]-a[:2]), a[2]+t*yaw])
    bodies = [scene._body(pose) for pose in dense]
    radius = math.hypot(scene.vehicle["length"]/2+abs(scene.vehicle["rear_axle_to_center"])
                        +scene.config.safety_margin_m, scene.vehicle["width"]/2+scene.config.safety_margin_m)
    pieces = [Polygon(bodies[0])]
    for i in range(1, len(bodies)):
        angle = abs(float(wrap_angle(dense[i][2]-dense[i-1][2])))
        pieces.append(MultiPoint(np.concatenate([bodies[i-1], bodies[i]])).convex_hull.buffer(
            radius*(1-math.cos(angle/2))+1e-6))
    return unary_union(pieces)


def audit_scene(scene, rendered, references, tasks, scene_id, near_distance_m=3., multi_vehicle_min=5):
    raster = rendered["object_raster_audit"]
    if len(raster) != len(scene.agents):
        raise RuntimeError("Tracked-object conversion/render count mismatch")
    sweeps = {task["id"]: _reference_sweep(scene, ref["poses"])
              for task, ref in zip(tasks, references) if ref["status"] == "optimal_on_graph"}
    objects = []
    for index, (agent, pixels) in enumerate(zip(scene.agents, raster)):
        if pixels["object_index"] != index or agent.get("source_index", index) != index:
            raise RuntimeError("Tracked-object ordering mismatch")
        actual = Polygon(oriented_box_corners(agent["center"], agent["heading"], agent["length"], agent["width"]))
        footprint = Polygon(oriented_box_corners(agent["center"], agent["heading"],
                                                max(agent["length"], .8), max(agent["width"], .6)))
        in_view = footprint.intersects(scene.viewport)
        road_distance = float(footprint.distance(scene.road))
        distances = {task["id"]: float(footprint.distance(sweeps[task["id"]]))
                     if task["id"] in sweeps else None for task in tasks}
        warnings = []
        if in_view and pixels["fill_pixels_before_overdraw"] == 0:
            warnings.append("in_view_without_fill_pixels")
        if pixels["fill_pixels_before_overdraw"] > 0 and pixels["visible_owned_pixels"] == 0:
            warnings.append("fully_overdrawn")
        if agent["type"] not in AGENT_LAYERS:
            warnings.append("unknown_type_rendered_as_static_obstacle")
        objects.append(dict(index=index, track_token=agent.get("track_token"),
            source_type=agent.get("source_type", agent["type"]), type=agent["type"],
            category=agent_layer(agent["type"]), center_ego=agent["center"],
            raw_length_m=agent["length"], raw_width_m=agent["width"],
            raw_footprint_in_view=actual.intersects(scene.viewport), footprint_in_view=in_view,
            center_in_view=scene.viewport.covers(Point(agent["center"])),
            distance_to_road_m=road_distance, near_road=bool(in_view and road_distance <= near_distance_m),
            distance_to_feasible_reference_sweep_m=distances,
            near_feasible_reference={k: bool(in_view and v <= near_distance_m) if v is not None else None
                                     for k, v in distances.items()},
            **pixels, warnings=warnings))
    in_view = [o for o in objects if o["footprint_in_view"]]
    warnings = Counter(w for o in objects for w in o["warnings"])
    audit = dict(audit_revision=1, scene_id=scene_id,
        source="current_frame_tracked_objects_excludes_ego", raw_count=len(objects),
        converted_count=len(scene.agents), submitted_to_renderer_count=len(raster),
        raw_type_counts=dict(Counter(o["type"] for o in objects)), raw_category_counts=_counts(objects),
        in_view_count=len(in_view), outside_view_count=len(objects)-len(in_view),
        in_view_counts=_counts(in_view),
        raw_footprint_in_view_count=sum(o["raw_footprint_in_view"] for o in objects),
        rasterized_counts=_counts([o for o in objects if o["fill_pixels_before_overdraw"] > 0]),
        visible_counts=_counts([o for o in objects if o["visible_owned_pixels"] > 0]),
        warning_counts=dict(warnings), objects=objects,
        note="One-to-one source index retained; invalid geometry aborts export, never silently dropped. "
             "In-view counts use minimum-size footprints (0.8 x 0.6 m), also used by the static planner. "
             "Raw-footprint visibility is recorded separately. Visible pixels include outline/heading and "
             "account for later objects, ego and goal overdraw. Region IDs are audit-only, not causal labels.")
    n = _counts(in_view)
    tags = ["sparse_traffic" if n["vehicle"] <= 1 else
            "multi_vehicle" if n["vehicle"] >= multi_vehicle_min else "moderate_traffic"]
    if n["pedestrian"]:
        tags.append("pedestrian_present")
    if n["bicycle"]:
        tags.append("bicycle_present")
    if n["static_obstacle"]:
        tags.append("static_obstacle_present")
    if any(o["type"] in CONSTRUCTION_TYPES for o in in_view) or scene.hazard.intersects(scene.viewport):
        tags.append("construction_present")
    features = dict(audit_revision=1, scene_id=scene_id, tags=tags, in_view_counts=n,
        near_road_counts=_counts([o for o in in_view if o["near_road"]]),
        near_feasible_reference_counts={t["id"]: _counts([o for o in in_view if o["near_feasible_reference"][t["id"]]])
                                        if t["id"] in sweeps else None for t in tasks},
        lane_count=len(scene.lanes), road_area_m2=float(scene.road.area),
        reference_status={t["id"]: r["status"] for t, r in zip(tasks, references)},
        protocol=dict(near_distance_m=near_distance_m, multi_vehicle_min=multi_vehicle_min,
                      sparse_vehicle_max=1, excludes_ego=True,
                      grouping="multi-label; all exported scenes retained; no task-difference filtering"),
        note="Near-road is geometric, not guaranteed traversability. Near-reference uses swept vehicle "
             "footprints of feasible task references; unavailable references are null, not zero. "
             "Proximity and object presence do not prove decision relevance.")
    return audit, features


def _progress_points(poses, samples=101):
    points = np.asarray(poses)[:, :2]
    station = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    grid = np.linspace(0., station[-1], samples)
    return np.c_[np.interp(grid, station, points[:, 0]), np.interp(grid, station, points[:, 1])]


def compare_references(references, tasks, cross_cost, difference_threshold_m=.5):
    pairs = []
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            a, b = references[i], references[j]
            pair = dict(tasks=[tasks[i]["id"], tasks[j]["id"]],
                        status="unavailable", different_geometry=None, cost_conflict=None)
            if a["status"] == b["status"] == "optimal_on_graph":
                distance = np.linalg.norm(_progress_points(a["poses"])-_progress_points(b["poses"]), axis=1)
                # Compare costs only within an objective column; never across units.
                gaps = [cross_cost[j][i]-a["cost"], cross_cost[i][j]-b["cost"]]
                conflict = any(not np.isfinite(gap) or gap > 1e-5*max(1., abs(ref["cost"]))
                               for gap, ref in zip(gaps, (a, b)))
                pair.update(status="compared", mean_progress_distance_m=float(distance.mean()),
                            max_progress_distance_m=float(distance.max()),
                            different_geometry=bool(distance.max() > difference_threshold_m),
                            cost_conflict=bool(conflict), cross_objective_excess_cost=gaps,
                            cross_objective_feasible=[bool(np.isfinite(cross_cost[j][i])),
                                                      bool(np.isfinite(cross_cost[i][j]))])
            pairs.append(pair)
    compared = [p for p in pairs if p["status"] == "compared"]
    complete = len(compared) == len(pairs) and bool(pairs)
    return dict(pairs=pairs, comparison_complete=complete,
        any_geometry_difference=any(p["different_geometry"] for p in compared) if compared else None,
        any_cost_conflict=any(p["cost_conflict"] for p in compared) if compared else None,
        all_tasks_no_cost_conflict=not any(p["cost_conflict"] for p in compared) if complete else None,
        protocol=dict(progress_samples=101, difference_threshold_m=difference_threshold_m,
                      cost_relative_tolerance=1e-5),
        note="Arc-length progress comparison of geometric reference paths, not time-aligned ADE/FDE. "
             "Different geometry can be a cost tie. Unavailable is not no-conflict. No scenes filtered.")


def write_dataset_index(root):
    """Deterministically index every exported scene once, without splitting task variants."""
    root = Path(root)
    tags, warnings, source_types, feasibility = Counter(), Counter(), Counter(), Counter()
    total_objects = Counter()
    comparison_counts = Counter()
    rows = []
    protocols = set()
    for path in sorted(root.glob("*/*/*/audit/scene_features.json")):
        feature = json.loads(path.read_text())
        objects = json.loads((path.parent/"objects.json").read_text())
        refs = json.loads((path.parent/"reference_summary.json").read_text())
        folder = path.parent.parent
        rows.append(dict(scene_id=feature["scene_id"], frame_directory=str(folder.relative_to(root)),
            log_name=folder.parent.parent.name, tags=feature["tags"], in_view_counts=feature["in_view_counts"],
            near_road_counts=feature["near_road_counts"],
            near_feasible_reference_counts=feature["near_feasible_reference_counts"],
            reference_status=feature["reference_status"], reference_cost=refs["reference_cost"],
            task_order=refs["task_order"], comparison=feature["reference_comparison"],
            object_warning_counts=objects["warning_counts"]))
        comparison = feature["reference_comparison"]
        comparison_counts["complete" if comparison["comparison_complete"] else "partial_or_unavailable"] += 1
        if comparison["any_cost_conflict"] is True:
            comparison_counts["observed_cost_conflict"] += 1
        if comparison["all_tasks_no_cost_conflict"] is True:
            comparison_counts["all_tasks_no_cost_conflict"] += 1
        if comparison["any_geometry_difference"] is True:
            comparison_counts["observed_geometry_difference"] += 1
        tags.update(feature["tags"])
        warnings.update(objects["warning_counts"])
        source_types.update(objects["raw_type_counts"])
        for key in ("raw_count", "converted_count", "submitted_to_renderer_count", "in_view_count", "outside_view_count"):
            total_objects[key] += objects[key]
        feasibility.update(task+":"+status for task, status in feature["reference_status"].items())
        protocols.add(json.dumps(feature["protocol"], sort_keys=True))
    with (root/"scene_index.jsonl").open("w") as out:
        for row in rows:
            out.write(json.dumps(json_ready(row), sort_keys=True, allow_nan=False)+"\n")
    report = dict(indexed_scenes=len(rows), scene_tag_counts=dict(tags), warning_counts=dict(warnings),
                  object_totals=dict(total_objects), source_type_counts=dict(source_types),
                  task_reference_status_counts=dict(feasibility),
                  reference_comparison_counts=dict(comparison_counts),
                  grouping_protocols=[json.loads(p) for p in sorted(protocols)],
                  mixed_grouping_protocols=len(protocols)>1,
                  note="Every exported frame indexed once, including sparse and no-conflict scenes. "
                       "Counts exclude ego. Tags overlap. Scene index is evaluation-only. Split by log.")
    write_json(root/"dataset_audit_summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild evaluation-only scene index from revision7 exports")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    print(json.dumps(write_dataset_index(args.output_root), indent=2))
