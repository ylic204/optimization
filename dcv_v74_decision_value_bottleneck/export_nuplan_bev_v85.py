"""Export camera-free ego-centric BEV records directly from nuPlan DBs.

Run this script in the pinned nuPlan-devkit environment.  It uses current
map/tracked-object/traffic-light state plus two seconds of past tracks.  Future
ego poses are saved only as supervision and are never rendered into the BEV.
"""

import argparse
import math
from pathlib import Path

import numpy as np

from bev_renderer_v85 import BevConfig, render_bev, world_to_ego, wrap_angle
from raw_record_bev_v85 import save_bev_raw_record


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
            result.append(np.asarray(exterior.coords, dtype=np.float32)[:, :2])
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
    return np.asarray(line.coords, dtype=np.float32)[:, :2]


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
        return np.zeros((0, 3), dtype=np.float32)
    if len(states) < 2:
        return np.zeros((0, 3), dtype=np.float32)
    xy = world_to_ego(
        [[state.rear_axle.x, state.rear_axle.y] for state in states],
        ego_pose,
    )
    yaw = np.asarray(
        [wrap_angle(state.rear_axle.heading - ego_pose[2]) for state in states],
        dtype=np.float32,
    )
    return np.concatenate([xy, yaw[:, None]], axis=-1)


def export_iteration(scenario, iteration, output_root, config, args):
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
    layers = [
        SemanticMapLayer.DRIVABLE_AREA,
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
        SemanticMapLayer.CROSSWALK,
        SemanticMapLayer.STOP_LINE,
        SemanticMapLayer.ROADBLOCK,
        SemanticMapLayer.ROADBLOCK_CONNECTOR,
    ]
    proximal = scenario.map_api.get_proximal_map_objects(
        Point2D(ego_pose[0], ego_pose[1]), query_radius, layers
    )
    route_ids = scenario.get_route_roadblock_ids()
    route_edges = _route_edges(scenario.map_api, route_ids, SemanticMapLayer)
    reference = _reference_path_ego(route_edges, ego_pose, config)
    if len(reference) < 2:
        return False
    expert = _future_expert(scenario, iteration, ego_pose, args)
    if len(expert) < 2:
        return False

    drivable_objects = proximal.get(SemanticMapLayer.DRIVABLE_AREA, [])
    if not drivable_objects:
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
        bev_config_json=config.to_json(),
        scene_type=scenario.scenario_type,
        log_name=scenario.log_name,
        scenario_token=scenario.token,
        timestamp_us=timestamp,
        iteration=iteration,
        **rendered,
    )
    return True


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
        remove_invalid_goals=True,
        shuffle=False,
        ego_start_speed_threshold=None,
        ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
        token_set_path=None,
        fraction_in_token_set_threshold=None,
        ego_route_radius=5.0,
    )
    return builder.get_scenarios(scenario_filter, Sequential())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--map-root", required=True)
    parser.add_argument("--sensor-root", default="")
    parser.add_argument("--db-files", nargs="+", required=True)
    parser.add_argument("--map-version", default="nuplan-maps-v1.0")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit-scenarios", type=int)
    parser.add_argument("--scenario-types", nargs="+")
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--future-horizon", type=float, default=8.0)
    parser.add_argument("--future-samples", type=int, default=16)
    parser.add_argument("--history-seconds", type=float, default=2.0)
    parser.add_argument("--history-samples", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--region-grid", type=int, default=9)
    parser.add_argument("--x-min", type=float, default=-16.0)
    parser.add_argument("--x-max", type=float, default=64.0)
    parser.add_argument("--y-min", type=float, default=-40.0)
    parser.add_argument("--y-max", type=float, default=40.0)
    parser.add_argument("--task-text", default=DEFAULT_TASK)
    return parser.parse_args()


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
    if args.future_horizon <= 0.0 or args.future_samples < 2:
        raise ValueError("future supervision requires a positive horizon and >=2 samples")
    if args.history_seconds <= 0.0 or args.history_samples < 2:
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
    print(f"wrote {written} BEV records; skipped {skipped} incomplete records")


if __name__ == "__main__":
    main()
