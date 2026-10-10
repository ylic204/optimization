"""Export camera-free ego-centric BEV records directly from nuPlan DBs.

Default: wide static snapshots with a goal on the truncated logged future GT.
GT is saved for evaluation, never rendered in the model input. No actor histories
are requested. Legacy route/GT export is opt-in via
--planning-mode route_tracking and uses a distinct record contract.
"""

import argparse
import math
import json
from pathlib import Path

import numpy as np

from bev_renderer_v85 import BevConfig, render_bev, world_to_ego, wrap_angle
from raw_record_bev_v85 import save_bev_raw_record, save_static_bev_raw_record
from static_bev_tasks_v85 import (
    hazard_polygons, static_evaluation_layers, task_variants,
)


DEFAULT_TASK = (
    "Drive safely to the route goal. Avoid collisions and non-drivable areas, "
    "follow traffic controls, and make efficient progress."
)


def _geometry_polygons(geometry):
    if geometry is None:
        return []
    geometries = list(getattr(geometry, "geoms", [geometry]))
    result = []
    for item in geometries:
        exterior = getattr(item, "exterior", None)
        if exterior is not None:
            result.append(np.asarray(exterior.coords, dtype=np.float64)[:, :2])
    return result


def _object_polygons(objects, ego_pose):
    result = []
    for item in objects:
        for polygon in _geometry_polygons(getattr(item, "polygon", None)):
            result.append(world_to_ego(polygon, ego_pose))
    return result


def _baseline(item):
    path = getattr(item, "baseline_path", None)
    line = getattr(path, "linestring", None)
    if line is None:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(line.coords, dtype=np.float64)[:, :2]


def _route_object(map_api, object_id, semantic_layer):
    for layer in (
        semantic_layer.ROADBLOCK,
        semantic_layer.ROADBLOCK_CONNECTOR,
    ):
        try:
            result = map_api.get_map_object(str(object_id), layer)
        except (KeyError, RuntimeError, ValueError):
            result = None
        if result is not None:
            return result
    return None


def _route_edges(map_api, route_ids, semantic_layer):
    roadblocks = [
        _route_object(map_api, object_id, semantic_layer)
        for object_id in route_ids
    ]
    return [
        (roadblock, list(getattr(roadblock, "interior_edges", [])))
        for roadblock in roadblocks
        if roadblock is not None
    ]


def _reference_path_world(route_edges, ego_xy):
    """Choose a connected route-lane sequence without using future ego poses."""
    candidates = []
    for block_index, (_, edges) in enumerate(route_edges):
        for edge in edges:
            points = _baseline(edge)
            if len(points) >= 2:
                distance = np.linalg.norm(points - ego_xy[None], axis=-1).min()
                candidates.append((float(distance), block_index, edge, points))
    if not candidates:
        return np.zeros((0, 2), dtype=np.float32)

    _, start_block, _, first_points = min(candidates, key=lambda value: value[0])
    nearest = int(np.linalg.norm(first_points - ego_xy[None], axis=-1).argmin())
    # nuPlan lane baselines follow the legal driving direction.  Never choose
    # the longer side of the line here: that can silently reverse the route
    # when ego is already near the end of the current lane.
    points = first_points[nearest:]
    if not len(points):
        points = first_points[-1:]
    pieces = [points]
    endpoint = points[-1]

    for _, edges in route_edges[start_block + 1 :]:
        options = []
        for edge in edges:
            value = _baseline(edge)
            if len(value) < 2:
                continue
            direct = np.linalg.norm(value[0] - endpoint)
            reverse = np.linalg.norm(value[-1] - endpoint)
            if reverse < direct:
                value = value[::-1]
                direct = reverse
            options.append((float(direct), value))
        if not options:
            continue
        _, chosen = min(options, key=lambda value: value[0])
        if np.linalg.norm(chosen[0] - endpoint) < 1e-3:
            chosen = chosen[1:]
        if not len(chosen):
            continue
        pieces.append(chosen)
        endpoint = chosen[-1]
    return np.concatenate(pieces, axis=0).astype(np.float32)


def _reference_path_ego(route_edges, ego_pose, config):
    world = _reference_path_world(route_edges, np.asarray(ego_pose[:2]))
    if len(world) < 2:
        return np.zeros((0, 3), dtype=np.float32)
    points = world_to_ego(world, ego_pose)
    distance = np.linalg.norm(points, axis=-1)
    start = int(distance.argmin())
    points = points[start:]
    if len(points) < 2:
        return np.zeros((0, 3), dtype=np.float32)
    if np.linalg.norm(points[0]) > 0.5:
        points = np.concatenate([np.zeros((1, 2), dtype=np.float32), points])
    inside = (
        (points[:, 0] >= config.x_min - 5.0)
        & (points[:, 0] <= config.x_max + 15.0)
        & (points[:, 1] >= config.y_min - 5.0)
        & (points[:, 1] <= config.y_max + 5.0)
    )
    if inside.any():
        last = min(len(points), int(np.flatnonzero(inside)[-1]) + 2)
        points = points[:last]
    delta = np.gradient(points, axis=0)
    yaw = np.arctan2(delta[:, 1], delta[:, 0])
    return np.concatenate([points, yaw[:, None]], axis=-1).astype(np.float32)


def _tracked_list(detections):
    collection = getattr(detections, "tracked_objects", detections)
    return list(getattr(collection, "tracked_objects", collection))


def _object_token(value):
    return str(
        getattr(value, "track_token", getattr(value, "token", id(value)))
    )


def _agent_primitive(value, ego_pose):
    center = value.center
    object_type = getattr(value, "tracked_object_type", None)
    type_name = getattr(object_type, "fullname", None) or getattr(
        object_type, "name", "generic_object"
    )
    geometry = [center.x, center.y, center.heading, value.box.length, value.box.width]
    if not np.isfinite(geometry).all() or value.box.length <= 0 or value.box.width <= 0:
        raise ValueError("Invalid tracked-object geometry; refusing to silently drop an object")
    token = getattr(value, "track_token", None) or getattr(value, "token", None)
    return dict(center=world_to_ego([[center.x, center.y]], ego_pose)[0],
                heading=float(wrap_angle(center.heading - ego_pose[2])),
                length=float(value.box.length), width=float(value.box.width),
                track_token=str(token) if token is not None else None,
                source_type=str(type_name), type=str(type_name).lower())


def _current_agents(scenario, iteration, ego_pose):
    values = _tracked_list(scenario.get_tracked_objects_at_iteration(iteration))
    return [dict(_agent_primitive(value, ego_pose), source_index=i) for i, value in enumerate(values)]


def _agents_and_histories(scenario, iteration, ego_pose, args):
    current_objects = _tracked_list(
        scenario.get_tracked_objects_at_iteration(iteration)
    )
    current_tokens = {_object_token(value) for value in current_objects}
    history = {token: [] for token in current_tokens}
    try:
        observations = list(
            scenario.get_past_tracked_objects(
                iteration,
                time_horizon=args.history_seconds,
                num_samples=args.history_samples,
            )
        )
    except (AssertionError, RuntimeError, ValueError):
        observations = []
    for observation in observations:
        for value in _tracked_list(observation):
            token = _object_token(value)
            if token in history:
                center = value.center
                history[token].append([center.x, center.y])

    agents = []
    for value in current_objects:
        center = value.center
        token = _object_token(value)
        history[token].append([center.x, center.y])
        local_center = world_to_ego([[center.x, center.y]], ego_pose)[0]
        object_type = getattr(value, "tracked_object_type", None)
        type_name = getattr(object_type, "fullname", None) or getattr(
            object_type, "name", "generic_object"
        )
        agents.append(
            {
                "center": local_center,
                "heading": float(wrap_angle(center.heading - ego_pose[2])),
                "length": float(value.box.length),
                "width": float(value.box.width),
                "type": str(type_name).lower(),
            }
        )
    history_lines = [
        world_to_ego(points, ego_pose)
        for points in history.values()
        if len(points) >= 2
    ]
    return agents, history_lines


def _traffic_lines(proximal, traffic_status, ego_pose, semantic_layer):
    status_names = {0: "green", 1: "yellow", 2: "red", 3: "unknown"}
    by_connector = {
        str(value.lane_connector_id): status_names.get(
            int(value.status), "unknown"
        )
        for value in traffic_status
    }
    result = {name: [] for name in ("red", "yellow", "green", "unknown")}
    connectors = proximal.get(semantic_layer.LANE_CONNECTOR, [])
    for connector in connectors:
        status = by_connector.get(str(connector.id))
        if status is None:
            continue
        stop_lines = list(getattr(connector, "stop_lines", []))
        if stop_lines:
            for stop_line in stop_lines:
                for polygon in _geometry_polygons(stop_line.polygon):
                    result[status].append(world_to_ego(polygon, ego_pose))
        else:
            baseline = _baseline(connector)
            if len(baseline) >= 2:
                result[status].append(world_to_ego(baseline[:2], ego_pose))
    return result


def _ego_state_vector(ego):
    dynamic = ego.dynamic_car_state
    velocity = dynamic.rear_axle_velocity_2d
    acceleration = dynamic.rear_axle_acceleration_2d
    return np.asarray(
        [
            velocity.x,
            velocity.y,
            acceleration.x,
            acceleration.y,
            dynamic.angular_velocity,
            ego.tire_steering_angle,
            getattr(dynamic, "angular_acceleration", 0.0),
            0.0,
        ],
        dtype=np.float32,
    )


def _future_expert(scenario, iteration, ego_pose, args):
    try:
        states = list(
            scenario.get_ego_future_trajectory(
                iteration,
                time_horizon=args.future_horizon,
                num_samples=args.future_samples,
            )
        )
    except (AssertionError, RuntimeError, ValueError):
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0)
    if len(states) < 2:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0)
    anchor = scenario.get_time_point(iteration).time_us
    states = [state for state in states if state.time_point.time_us > anchor]
    times = np.asarray([(state.time_point.time_us - anchor) * 1e-6 for state in states])
    if len(times) != args.future_samples or (np.diff(times) <= 0).any():
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0)
    xy = world_to_ego(
        [[state.rear_axle.x, state.rear_axle.y] for state in states],
        ego_pose,
    )
    yaw = np.asarray(
        [wrap_angle(state.rear_axle.heading - ego_pose[2]) for state in states],
        dtype=np.float32,
    )
    return np.concatenate([xy, yaw[:, None]], axis=-1), times


def _proximal_map_objects(map_api, point, radius, semantic_layer):
    """Query nuPlan's object layers; DRIVABLE_AREA has no object getter.

    The driving surface is rendered from LANE and LANE_CONNECTOR polygons.
    DRIVABLE_AREA is available via the separate raster API, not this query.
    """
    layers = [
        semantic_layer.LANE,
        semantic_layer.LANE_CONNECTOR,
        semantic_layer.CROSSWALK,
        semantic_layer.STOP_LINE,
        semantic_layer.ROADBLOCK,
        semantic_layer.ROADBLOCK_CONNECTOR,
    ]
    return map_api.get_proximal_map_objects(point, radius, layers)


def _export_route_tracking(scenario, iteration, output_root, config, args):
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer

    ego = scenario.get_ego_state_at_iteration(iteration)
    ego_pose = np.asarray(
        [ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading],
        dtype=np.float64,
    )
    query_radius = math.hypot(
        max(abs(config.x_min), abs(config.x_max)),
        max(abs(config.y_min), abs(config.y_max)),
    ) + 10.0
    proximal = _proximal_map_objects(
        scenario.map_api,
        Point2D(ego_pose[0], ego_pose[1]),
        query_radius,
        SemanticMapLayer,
    )
    route_ids = scenario.get_route_roadblock_ids()
    route_edges = _route_edges(scenario.map_api, route_ids, SemanticMapLayer)
    reference = _reference_path_ego(route_edges, ego_pose, config)
    if len(reference) < 2:
        return False
    expert, expert_times = _future_expert(scenario, iteration, ego_pose, args)
    planning_times = np.arange(1, args.future_samples + 1) * args.future_horizon / args.future_samples
    if len(expert) < 2 or planning_times[-1] > expert_times[-1] + 1e-7:
        return False

    drivable_objects = (
        proximal.get(SemanticMapLayer.LANE, [])
        + proximal.get(SemanticMapLayer.LANE_CONNECTOR, [])
    )
    route_id_set = {str(value) for value in route_ids}
    local_route_objects = [
        value
        for layer in (
            SemanticMapLayer.ROADBLOCK,
            SemanticMapLayer.ROADBLOCK_CONNECTOR,
        )
        for value in proximal.get(layer, [])
        if str(value.id) in route_id_set
    ]
    lanes = proximal.get(SemanticMapLayer.LANE, []) + proximal.get(
        SemanticMapLayer.LANE_CONNECTOR, []
    )
    map_polygons = {
        "drivable": _object_polygons(drivable_objects, ego_pose),
        "route": _object_polygons(local_route_objects, ego_pose),
        "crosswalk": _object_polygons(
            proximal.get(SemanticMapLayer.CROSSWALK, []), ego_pose
        ),
    }
    map_lines = {
        "lane_center": [
            world_to_ego(value, ego_pose)
            for value in (_baseline(lane) for lane in lanes)
            if len(value) >= 2
        ]
    }
    traffic = list(scenario.get_traffic_light_status_at_iteration(iteration))
    traffic_lines = _traffic_lines(
        proximal, traffic, ego_pose, SemanticMapLayer
    )
    agents, histories = _agents_and_histories(
        scenario, iteration, ego_pose, args
    )
    goal_xy = reference[-1, :2]
    vehicle = scenario.ego_vehicle_parameters
    rendered = render_bev(
        map_polygons,
        map_lines,
        agents,
        histories,
        traffic_lines,
        reference,
        goal_xy,
        config=config,
        ego_size=(vehicle.length, vehicle.width),
    )
    if not rendered["traversable"].any():
        return False

    timestamp = scenario.get_time_point(iteration).time_us
    sample_id = f"{scenario.log_name}/{scenario.token}/{iteration:04d}"
    output_path = (
        Path(output_root)
        / scenario.log_name
        / scenario.token
        / f"{iteration:04d}.npz"
    )
    save_bev_raw_record(
        output_path=output_path,
        reference_path_ego=reference,
        map_bounds=config.bounds,
        goal_xy=goal_xy,
        ego_state=_ego_state_vector(ego),
        task_text=args.task_text,
        sample_id=sample_id,
        expert_trajectory_ego=expert,
        expert_time_s=expert_times,
        planning_time_s=planning_times,
        bev_config_json=config.to_json(),
        scene_type=scenario.scenario_type,
        log_name=scenario.log_name,
        scenario_token=scenario.token,
        timestamp_us=timestamp,
        iteration=iteration,
        **rendered,
    )
    return True


def _save_preview(rendered, goal, config, output_path):
    """Exactly the model RGB: no text, axes, legend, padding or reference."""
    from PIL import Image
    Image.fromarray(rendered["bev_rgb"]).save(output_path)


def _skip_static(args, reason):
    counts = getattr(args, "skip_reasons", {})
    counts[reason] = counts.get(reason, 0) + 1
    args.skip_reasons = counts
    return False


def _road_lanes(proximal, ego_pose, semantic_layer):
    """Keep all local directed lanes, not the logged route or future intent."""
    result = []
    for kind, layer in (("lane", semantic_layer.LANE), ("connector", semantic_layer.LANE_CONNECTOR)):
        next_kind = "connector" if kind == "lane" else "lane"
        for lane in proximal.get(layer, []):
            baseline = _baseline(lane)
            polygons = _geometry_polygons(lane.polygon)
            if (len(baseline) < 2 or len(polygons) != 1 or not np.isfinite(baseline).all()
                    or np.linalg.norm(np.diff(baseline, axis=0), axis=1).sum() < .1):
                continue
            result.append(dict(id=f"{kind}:{lane.id}", kind=kind,
                               polygon=world_to_ego(polygons[0], ego_pose).tolist(),
                               holes=[world_to_ego(np.asarray(h.coords, dtype=np.float64)[:, :2], ego_pose).tolist()
                                      for h in getattr(lane.polygon, "interiors", ())],
                               baseline=world_to_ego(baseline, ego_pose).tolist(),
                               outgoing=[f"{next_kind}:{edge.id}" for edge in lane.outgoing_edges],
                               adjacent=[f"lane:{edge.id}" for edge in
                                         (lane.adjacent_edges if kind == "lane" else ()) if edge is not None]))
    return result


def _export_static_multitask(scenario, iteration, output_root, config, args):
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from static_road_planner_v85 import RoadScene, RoadGraph, RoadPlanningConfig, write_json, task_cost

    ego = scenario.get_ego_state_at_iteration(iteration)
    ego_pose = np.asarray([ego.rear_axle.x, ego.rear_axle.y, ego.rear_axle.heading])
    logged_gt = None
    gt_mode = args.goal_source == "logged_gt_boundary"
    vehicle = scenario.ego_vehicle_parameters
    center = ego.center
    ego_center = world_to_ego([[center.x, center.y]], ego_pose)[0]
    geometry = dict(length=float(vehicle.length), width=float(vehicle.width),
                    rear_axle_to_center=float(ego_center[0]))
    if gt_mode:
        from logged_gt_goal_v85 import load_logged_future, boundary_inset, truncate_at_boundary
        try:
            poses, times = load_logged_future(scenario, iteration, ego_pose,
                                             args.gt_horizon_s, args.gt_sample_interval_s)
            inset = boundary_inset(geometry, args.safety_margin_m, args.goal_margin_m)
            logged_gt = truncate_at_boundary(poses, times, config.bounds, inset)
        except (AssertionError, RuntimeError, ValueError) as error:
            return _skip_static(args, "logged_gt_unavailable_or_invalid:"+type(error).__name__)
        if logged_gt is None:
            return _skip_static(args, "logged_gt_does_not_reach_boundary")
    radius = math.hypot(max(abs(config.x_min), abs(config.x_max)),
                        max(abs(config.y_min), abs(config.y_max))) + 10.
    proximal = _proximal_map_objects(scenario.map_api, Point2D(*ego_pose[:2]), radius, SemanticMapLayer)
    lanes = _road_lanes(proximal, ego_pose, SemanticMapLayer)
    if not lanes:
        return _skip_static(args, "no_directed_lane_geometry")
    polygons = dict(drivable=[lane["polygon"] for lane in lanes],
                    road_holes=[hole for lane in lanes for hole in lane["holes"]],
                    crosswalk=_object_polygons(proximal.get(SemanticMapLayer.CROSSWALK, []), ego_pose))
    polygons["lane_boundary"] = [lane["polygon"] for lane in lanes if lane["kind"] == "lane"]
    lines = {"lane_center": [lane["baseline"] for lane in lanes if lane["kind"] == "lane"],
             "connector_center": [lane["baseline"] for lane in lanes if lane["kind"] == "connector"]}
    agents = _current_agents(scenario, iteration, ego_pose)
    hazards = hazard_polygons(agents, args.hazard_buffer_m)
    planner_config = RoadPlanningConfig(station_step_m=args.station_step_m,
                                        lateral_offsets_m=tuple(args.lateral_offsets_m),
                                        collision_step_m=args.collision_step_m,
                                        min_turn_radius_m=args.min_turn_radius_m,
                                        safety_margin_m=args.safety_margin_m,
                                        pedestrian_scale_m=args.pedestrian_scale_m,
                                        lane_change_policy=args.lane_change_policy)
    scene = RoadScene(lanes, agents, hazards, config.bounds, geometry, planner_config)
    lines["road_outline"] = [np.asarray(p.exterior.coords) for p in
                             getattr(scene.road, "geoms", [scene.road]) if hasattr(p, "exterior")]
    if args.goal_xy is not None:
        x, y = args.goal_xy
        if not (config.x_min+args.goal_margin_m <= x <= config.x_max-args.goal_margin_m
                and config.y_min+args.goal_margin_m <= y <= config.y_max-args.goal_margin_m
                and x >= args.goal_min_forward_m
                and args.goal_min_distance <= np.linalg.norm(args.goal_xy) <= args.goal_max_distance):
            return _skip_static(args, "invalid_explicit_goal")
    scene_id = f"{scenario.log_name}/{scenario.token}/{iteration:04d}"
    graph = None
    target = -1
    if gt_mode:
        goal_pose = logged_gt["poses"][-1].copy()
        from logged_gt_goal_v85 import geometric_gt
        logged_gt["static_metrics"] = scene.assess(geometric_gt(logged_gt["poses"]))
        if not args.skip_numeric_reference:
            graph = RoadGraph(scene).build()
            target = graph.add_pose_goal(goal_pose)
    else:
        graph = RoadGraph(scene).build(args.goal_xy)
        base_task = dict(objective="length", pedestrian_weight=0., forbid_hazards=False)
        base_solution = graph.solve(base_task)
        target = graph.choose_goal(base_solution[0], args, scene_id)
        if target is None:
            return _skip_static(args, "no_forward_reachable_road_goal")
        goal_pose = graph.poses[target]
    goal = goal_pose[:2]
    rendered = render_bev(polygons, lines, agents, [], {}, None, goal, config=config,
                          ego_size=(vehicle.length, vehicle.width), static_scene=True,
                          hazard_polygons=hazards, ego_center_xy=ego_center)
    evaluation = static_evaluation_layers(rendered, config)
    tasks = task_variants(goal_pose, args, evaluation, geometry)
    references = ([graph.reference(target, graph.solve(task), task) for task in tasks] if graph is not None else
                  [dict(status="not_computed", cost=None, poses=np.empty((0,3)), edge_ids=[], metrics={}) for _ in tasks])
    # Cross-task evaluation records task conflict; it does not select/filter goals.
    cross_cost = [[task_cost(ref["metrics"], task) if ref["metrics"] else np.inf
                   for task in tasks] for ref in references]
    metadata = dict(scene_type=np.str_(scenario.scenario_type),
                    log_name=np.str_(scenario.log_name), scenario_token=np.str_(scenario.token),
                    timestamp_us=np.int64(scenario.get_time_point(iteration).time_us),
                    iteration=np.int64(iteration), goal_seed=np.int64(args.goal_seed),
                    goal_source=np.str_("logged_gt_boundary" if gt_mode else
                                        "explicit_road_goal" if args.goal_xy is not None else "forward_downstream_lane_station"),
                    goal_lane_id=np.str_(graph.node_lane[target] if graph is not None else ""), goal_node_id=np.int64(target),
                    goal_pose=np.asarray(goal_pose), start_pose=np.zeros(3),
                    vehicle_geometry_json=np.str_(json.dumps(geometry, sort_keys=True)),
                    planner_config_json=np.str_(json.dumps(scene.to_dict()["config"], sort_keys=True)),
                    hazard_source=np.str_("construction_object_buffer_not_dataset_risk_gt"),
                    hazard_buffer_m=np.float32(args.hazard_buffer_m),
                    reference_scope=np.str_("finite_directed_road_graph_sampled_cost"))
    folder = Path(output_root) / scenario.log_name / scenario.token / f"{iteration:04d}"
    if folder.exists():
        raise FileExistsError(f"Refusing to mix or overwrite exports: {folder}; choose a new --output-root")
    folder.mkdir(parents=True)
    audit = folder / "audit"
    audit.mkdir()
    from bev_scene_audit_v85 import audit_scene, compare_references
    object_audit, features = audit_scene(scene, rendered, references, tasks, scene_id,
                                        near_distance_m=args.audit_near_distance_m,
                                        multi_vehicle_min=args.audit_multi_vehicle_min)
    comparison = compare_references(references, tasks, cross_cost)
    write_json(audit / "objects.json", object_audit)
    write_json(audit / "scene_features.json", dict(features, reference_comparison=comparison))
    write_json(audit / "scene_geometry.json", scene.to_dict())
    if graph is not None:
        graph.save(audit / "planning_graph.npz")
    if logged_gt is not None:
        write_json(audit / "logged_gt_summary.json", {k:v for k,v in logged_gt.items() if k not in ("poses", "times")})
        if args.save_preview:
            from PIL import Image, ImageDraw
            from bev_renderer_v85 import ego_to_pixel
            image = Image.fromarray(rendered["bev_rgb"])
            ImageDraw.Draw(image).line([tuple(p) for p in ego_to_pixel(logged_gt["poses"][:,:2], config)],
                                       fill=(34, 145, 75), width=max(2, config.image_size//240))
            image.save(audit / "logged_gt.png")
    for task, reference in zip(tasks, references):
        save_static_bev_raw_record(folder / f"{task['id']}.npz", rendered, config,
                                   goal, task, scene_id, metadata, evaluation, reference, logged_gt=logged_gt)
        if args.save_preview and reference["status"] == "optimal_on_graph":
            from PIL import Image, ImageDraw
            from bev_renderer_v85 import ego_to_pixel
            image = Image.fromarray(rendered["bev_rgb"])
            ImageDraw.Draw(image).line([tuple(p) for p in ego_to_pixel(reference["poses"][:, :2], config)],
                                       fill=(132, 43, 169), width=max(2, config.image_size//180))
            image.save(audit / f"reference_{task['id']}.png")
    write_json(audit / "reference_summary.json", dict(
        scene_id=scene_id, goal_pose=goal_pose, reference_scope=str(metadata["reference_scope"]),
        graph_nodes=len(graph.poses) if graph is not None else 0, graph_edges=len(graph.edges) if graph is not None else 0, task_order=[t["id"] for t in tasks],
        reference_status=[r["status"] for r in references],
        reference_cost=[r["cost"] for r in references], cross_task_cost=cross_cost,
        cross_task_feasible=[[bool(np.isfinite(v)) for v in row] if ref["status"] == "optimal_on_graph"
                             else [None]*len(tasks) for ref, row in zip(references, cross_cost)],
        comparison=comparison,
        note="Rows: reference paths. Columns: task objectives. Null means infeasible/unavailable. "
             "Costs have different task units; compare within a column. No task-contrast filtering."))
    if args.save_preview:
        _save_preview(rendered, goal, config, folder / "bev_preview.png")
    return True


def export_iteration(scenario, iteration, output_root, config, args):
    if args.planning_mode == "static_multitask":
        return _export_static_multitask(scenario, iteration, output_root, config, args)
    if args.planning_mode == "route_tracking":
        return _export_route_tracking(scenario, iteration, output_root, config, args)
    raise ValueError(f"unknown planning mode: {args.planning_mode}")


def build_scenarios(args):
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
        NuPlanScenarioBuilder,
    )
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    builder = NuPlanScenarioBuilder(
        data_root=args.data_root,
        map_root=args.map_root,
        sensor_root=args.sensor_root or args.data_root,
        db_files=args.db_files,
        map_version=args.map_version,
        include_cameras=False,
        max_workers=1,
        verbose=True,
    )
    scenario_filter = ScenarioFilter(
        scenario_types=args.scenario_types,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=args.limit_scenarios,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=args.planning_mode == "route_tracking",
        shuffle=False,
        ego_start_speed_threshold=None,
        ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
        token_set_path=None,
        fraction_in_token_set_threshold=None,
        ego_route_radius=5.0,
    )
    return builder.get_scenarios(scenario_filter, Sequential())


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--map-root", required=True)
    parser.add_argument("--sensor-root", default="")
    parser.add_argument("--db-files", nargs="+", required=True)
    parser.add_argument("--map-version", default="nuplan-maps-v1.0")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--planning-mode", choices=("static_multitask", "route_tracking"),
                        default="static_multitask")
    parser.add_argument("--limit-scenarios", type=int)
    parser.add_argument("--scenario-types", nargs="+")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--future-horizon", type=float, default=8.0)
    parser.add_argument("--future-samples", type=int, default=16)
    parser.add_argument("--history-seconds", type=float, default=None,
                        help="route_tracking only; defaults to 2s there")
    parser.add_argument("--history-samples", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--region-grid", type=int, default=9)
    parser.add_argument("--x-min", type=float, default=None)
    parser.add_argument("--x-max", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--task-text", default=None, help="route_tracking only")
    parser.add_argument("--tasks", nargs="+", choices=("shortest", "pedestrian_clearance", "avoid_hazards"),
                        default=["shortest", "pedestrian_clearance", "avoid_hazards"])
    parser.add_argument("--goal-source", choices=("logged_gt_boundary", "road_graph"), default=None,
                        help="default: logged GT first exit; explicit --goal-xy implies road_graph")
    parser.add_argument("--gt-horizon-s", type=float, default=20.)
    parser.add_argument("--gt-sample-interval-s", type=float, default=.1)
    parser.add_argument("--skip-numeric-reference", action="store_true",
                        help="GT boundary mode: export GT/images without building the numeric reference graph")
    parser.add_argument("--audit-near-distance-m", type=float, default=3.,
                        help="audit only: maximum footprint distance to road/reference sweep")
    parser.add_argument("--audit-multi-vehicle-min", type=int, default=5,
                        help="audit only: visible non-ego vehicle count defining multi_vehicle")
    parser.add_argument("--goal-xy", type=float, nargs=2, metavar=("X", "Y"),
                        help="optional goal in local meters; invalid/unreachable scenes are skipped")
    parser.add_argument("--goal-min-distance", type=float, default=18.)
    parser.add_argument("--goal-max-distance", type=float, default=28.)
    parser.add_argument("--goal-margin-m", type=float, default=3.)
    parser.add_argument("--goal-seed", type=int, default=7)
    parser.add_argument("--goal-min-forward-m", type=float, default=8.)
    parser.add_argument("--station-step-m", type=float, default=4.)
    parser.add_argument("--lateral-offsets-m", type=float, nargs="+", default=[-.5, 0., .5])
    parser.add_argument("--collision-step-m", type=float, default=.25)
    parser.add_argument("--min-turn-radius-m", type=float, default=5.5)
    parser.add_argument("--safety-margin-m", type=float, default=.15)
    parser.add_argument("--lane-change-policy", choices=("adjacent", "outgoing_only"), default="adjacent",
                        help="adjacent permits curvature/footprint-checked changes between explicit map neighbors")
    parser.add_argument("--hazard-buffer-m", type=float, default=3.)
    parser.add_argument("--pedestrian-weight", type=float, default=4.)
    parser.add_argument("--pedestrian-scale-m", type=float, default=4.)
    parser.add_argument("--nominal-speed", type=float, default=1.)
    parser.add_argument("--no-preview", dest="save_preview", action="store_false", default=True)
    args = parser.parse_args(argv)
    static = args.planning_mode == "static_multitask"
    if static and (args.history_seconds not in (None, 0.) or args.history_samples not in (None, 0)):
        parser.error("static_multitask has no histories; remove --history-seconds/--history-samples")
    if static and args.task_text is not None:
        parser.error("static_multitask uses defined task objectives; select --tasks instead of --task-text")
    args.history_seconds = 0. if static else (2. if args.history_seconds is None else args.history_seconds)
    args.history_samples = 0 if static else (5 if args.history_samples is None else args.history_samples)
    args.goal_source = ("road_graph" if args.goal_xy is not None else "logged_gt_boundary") if args.goal_source is None else args.goal_source
    if static and args.goal_source == "logged_gt_boundary" and args.goal_xy is not None:
        parser.error("GT boundary goal cannot be overridden by --goal-xy; use --goal-source road_graph")
    if args.skip_numeric_reference and (not static or args.goal_source != "logged_gt_boundary"):
        parser.error("--skip-numeric-reference requires static logged_gt_boundary mode")
    args.image_size = (1440 if static else 288) if args.image_size is None else args.image_size
    args.y_min = (-48. if static else -40.) if args.y_min is None else args.y_min
    args.y_max = (48. if static else 40.) if args.y_max is None else args.y_max
    args.x_min = (-24. if static else -16.) if args.x_min is None else args.x_min
    args.x_max = (72. if static else 64.) if args.x_max is None else args.x_max
    args.task_text = DEFAULT_TASK if args.task_text is None else args.task_text
    args.tasks = list(dict.fromkeys(args.tasks))
    if static:
        if not np.isfinite(args.audit_near_distance_m) or args.audit_near_distance_m < 0 or args.audit_multi_vehicle_min < 2:
            parser.error("audit distance must be finite/nonnegative; multi-vehicle threshold must be >=2")
        positive = (args.goal_min_distance, args.goal_max_distance, args.goal_margin_m,
                    args.goal_min_forward_m, args.station_step_m, args.collision_step_m,
                    args.min_turn_radius_m, args.pedestrian_scale_m, args.nominal_speed,
                    args.gt_horizon_s, args.gt_sample_interval_s)
        if not all(np.isfinite(v) and v > 0 for v in positive):
            parser.error("goal distances, margins, planning steps, radius, scale and speed must be finite and positive")
        if args.gt_horizon_s < args.gt_sample_interval_s:
            parser.error("GT horizon must be at least one sampling interval")
        if args.goal_xy is not None and not np.isfinite(args.goal_xy).all():
            parser.error("goal must be finite")
        if (not np.isfinite(args.safety_margin_m) or args.safety_margin_m < 0
                or not np.isfinite(args.lateral_offsets_m).all() or 0. not in args.lateral_offsets_m
                or len(set(args.lateral_offsets_m)) != len(args.lateral_offsets_m)
                or args.collision_step_m > args.station_step_m):
            parser.error("invalid safety margin, offsets or collision step")
        if args.goal_max_distance < args.goal_min_distance:
            parser.error("goal-max-distance must be >= goal-min-distance")
        if (not np.isfinite(args.hazard_buffer_m) or args.hazard_buffer_m < 0 or
                not np.isfinite(args.pedestrian_weight) or args.pedestrian_weight < 0):
            parser.error("hazard buffer and pedestrian weight must be finite and nonnegative")
    return args


def main():
    args = parse_args()
    from tqdm import tqdm

    if args.frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    config = BevConfig(
        image_size=args.image_size,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        region_grid=args.region_grid,
        history_seconds=args.history_seconds,
        history_samples=args.history_samples,
    )
    if args.planning_mode == "static_multitask" and not (
            config.x_min < 0 < config.x_max and config.y_min < 0 < config.y_max):
        raise ValueError("static BEV bounds must contain the start S=(0,0)")
    if args.planning_mode == "route_tracking" and (args.future_horizon <= 0.0 or args.future_samples != 16):
        raise ValueError("V8.5 processing requires a positive horizon and exactly 16 future samples")
    if args.planning_mode == "route_tracking" and (args.history_seconds <= 0.0 or args.history_samples < 2):
        raise ValueError("history rendering requires a positive horizon and >=2 samples")
    scenarios = build_scenarios(args)
    written = 0
    skipped = 0
    for scenario in tqdm(scenarios, desc="nuPlan BEV scenarios"):
        for iteration in range(
            0, scenario.get_number_of_iterations(), args.frame_stride
        ):
            if export_iteration(scenario, iteration, args.output_root, config, args):
                written += 1
            else:
                skipped += 1
    print(f"{args.planning_mode}: wrote {written} scenes; skipped {skipped} invalid/unreachable scenes")
    if args.planning_mode == "static_multitask":
        print(f"{written * len(set(args.tasks))} task records; same RGB/goal across tasks in each scene")
        from static_road_planner_v85 import write_json
        root = Path(args.output_root)
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "export_summary.json", dict(written_scenes=written, skipped_scenes=skipped,
                   skip_reasons=getattr(args, "skip_reasons", {}), args=vars(args),
                   note="Logged GT (when available) and static numeric references are distinct; not closed-loop nuPlan evaluation."))
        from bev_scene_audit_v85 import write_dataset_index
        report = write_dataset_index(root)
        print("Scene groups:", report["scene_tag_counts"])
        print("Object audit warnings:", report["warning_counts"])
        print("Skip reasons:", getattr(args, "skip_reasons", {}))


if __name__ == "__main__":
    main()
