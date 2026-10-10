"""Directed, car-footprint static reference planning (no nuPlan dependency).

Dijkstra is exact only on the exported finite graph and its sampled edge costs.
This is a low-speed geometric benchmark, not a closed-loop driving controller.
Geometry, graph and references are evaluator-only, never selector/VLM inputs.
"""

from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
import math

import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep

from bev_renderer_v85 import oriented_box_corners, wrap_angle


@dataclass(frozen=True)
class RoadPlanningConfig:
    station_step_m: float = 4.
    lateral_offsets_m: tuple = (-.5, 0., .5)
    max_station_skip: int = 2
    collision_step_m: float = .25
    min_turn_radius_m: float = 5.5
    safety_margin_m: float = .15
    max_heading_error_rad: float = math.pi / 3
    pedestrian_scale_m: float = 4.
    lane_change_policy: str = "adjacent"

    def __post_init__(self):
        values = (self.station_step_m, self.collision_step_m, self.min_turn_radius_m,
                  self.max_heading_error_rad, self.pedestrian_scale_m)
        if not all(np.isfinite(x) and x > 0 for x in values):
            raise ValueError("road planning distances and angles must be finite and positive")
        if not np.isfinite(self.safety_margin_m) or self.safety_margin_m < 0:
            raise ValueError("safety margin must be finite and nonnegative")
        if self.max_station_skip < 1 or self.collision_step_m > self.station_step_m:
            raise ValueError("invalid graph or collision resolution")
        if 0. not in self.lateral_offsets_m or not np.isfinite(self.lateral_offsets_m).all():
            raise ValueError("lateral offsets must be finite and include zero")
        if self.lane_change_policy not in ("adjacent", "outgoing_only"):
            raise ValueError("unsupported lane-change policy")


def _polygon(points, holes=None):
    shape = Polygon(points, holes=holes)
    return shape if shape.is_valid else shape.buffer(0)


def _line_state(line, station):
    station = float(np.clip(station, 0, line.length))
    xy = np.asarray(line.interpolate(station).coords[0])
    a = np.asarray(line.interpolate(max(0., station - .25)).coords[0])
    b = np.asarray(line.interpolate(min(line.length, station + .25)).coords[0])
    return np.r_[xy, math.atan2(*(b - a)[::-1])]


def hermite_motion(start, end, step_m, radius_m):
    """Forward tangent-continuous primitive; sampled analytic curvature bound."""
    start, end = np.asarray(start), np.asarray(end)
    distance = np.linalg.norm(end[:2] - start[:2])
    if distance < 1e-3:
        return None
    v0 = distance * np.array([math.cos(start[2]), math.sin(start[2])])
    v1 = distance * np.array([math.cos(end[2]), math.sin(end[2])])
    t = np.linspace(0, 1, max(9, int(math.ceil(2 * distance / step_m)) + 1))[:, None]
    xy = ((2*t**3-3*t**2+1)*start[:2] + (t**3-2*t**2+t)*v0
          + (-2*t**3+3*t**2)*end[:2] + (t**3-t**2)*v1)
    d = ((6*t**2-6*t)*start[:2] + (3*t**2-4*t+1)*v0
         + (-6*t**2+6*t)*end[:2] + (3*t**2-2*t)*v1)
    dd = ((12*t-6)*start[:2] + (6*t-4)*v0
          + (-12*t+6)*end[:2] + (6*t-2)*v1)
    speed = np.linalg.norm(d, axis=1)
    if speed.min() < 1e-5:
        return None
    curvature = np.abs(d[:, 0]*dd[:, 1] - d[:, 1]*dd[:, 0]) / speed**3
    if curvature.max() > 1 / radius_m + 1e-7:
        return None
    pose = np.c_[xy, np.arctan2(d[:, 1], d[:, 0])]
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    # Bound the sampling gap, including pathological looping primitives.
    if ds.max() > step_m or ds.sum() > 1.6 * distance:
        return None
    return pose


class RoadScene:
    def __init__(self, lanes, agents, hazards, bounds, vehicle, config=None):
        self.config = config or RoadPlanningConfig()
        self.bounds = np.asarray(bounds, dtype=float)
        self.vehicle = dict(vehicle)
        if not all(np.isfinite(vehicle[k]) for k in ("length", "width", "rear_axle_to_center")):
            raise ValueError("invalid vehicle geometry")
        if vehicle["length"] <= 0 or vehicle["width"] <= 0:
            raise ValueError("vehicle dimensions must be positive")
        self.lanes = {lane["id"]: lane for lane in lanes}
        self.polygons = {k: _polygon(v["polygon"], v.get("holes")) for k, v in self.lanes.items()}
        self.lane_checks = {k: prep(p.buffer(1e-7)) for k, p in self.polygons.items()}
        self.lines = {k: LineString(v["baseline"]) for k, v in self.lanes.items()}
        self.viewport = box(bounds[0], bounds[2], bounds[1], bounds[3])
        self.road = unary_union(list(self.polygons.values())).intersection(self.viewport)
        self.road_check = prep(self.road.buffer(1e-7))
        self.agents, self.hazards = agents, hazards
        footprints = [(_polygon(oriented_box_corners(
            a["center"], a["heading"], max(a["length"], .8), max(a["width"], .6))), a["type"])
            for a in agents]
        self.obstacles = unary_union([p for p, _ in footprints])
        self.obstacle_check = prep(self.obstacles)
        self.pedestrians = unary_union([p for p, kind in footprints if kind == "pedestrian"])
        self.hazard = unary_union([_polygon(p) for p in hazards])
        self.hazard_check = prep(self.hazard)

    def to_dict(self):
        return dict(lanes=list(self.lanes.values()), agents=self.agents, hazards=self.hazards,
                    bounds=self.bounds.tolist(), vehicle=self.vehicle, config=asdict(self.config))

    @classmethod
    def from_dict(cls, data):
        return cls(data["lanes"], data["agents"], data["hazards"], data["bounds"],
                   data["vehicle"], RoadPlanningConfig(**data["config"]))

    def lane_matches(self, pose, lane_ids=None):
        point = Point(pose[:2])
        result = []
        for key in self.lanes if lane_ids is None else lane_ids:
            if not self.lane_checks[key].covers(point):
                continue
            line = self.lines[key]
            yaw = _line_state(line, line.project(point))[2]
            if abs(float(wrap_angle(pose[2] - yaw))) <= self.config.max_heading_error_rad:
                result.append(key)
        return result

    def adjacent(self, key):
        if self.config.lane_change_policy != "adjacent" or self.lanes[key]["kind"] != "lane":
            return []
        return [other for other in self.lanes[key].get("adjacent", [])
                if other in self.lanes and self.lanes[other]["kind"] == "lane"
                and key in self.lanes[other].get("adjacent", [])
                and self.polygons[key].distance(self.polygons[other]) < .05]

    def _body(self, pose):
        offset = self.vehicle["rear_axle_to_center"]
        center = pose[:2] + offset * np.array([math.cos(pose[2]), math.sin(pose[2])])
        margin = self.config.safety_margin_m
        return oriented_box_corners(center, pose[2], self.vehicle["length"] + 2*margin,
                                    self.vehicle["width"] + 2*margin)

    def assess(self, poses, lane_ids=None):
        """Independent swept-body/direction check and sampled metric costs.

        Pose segments use forward linear xy/shortest-angle heading interpolation.
        Convex swept envelopes get a rotational sagitta buffer. Collision sampling
        is numerical, with its resolution saved in the record, not a safety proof.
        """
        poses = np.asarray(poses, dtype=float)
        failure = dict(feasible=False, reason="invalid_trajectory")
        if poses.ndim != 2 or poses.shape[1] != 3 or len(poses) < 2 or not np.isfinite(poses).all():
            return failure
        ds = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
        dh = np.asarray(wrap_angle(np.diff(poses[:, 2])))
        if (ds <= 1e-9).any() or (np.abs(dh) / ds > 1/self.config.min_turn_radius_m + 1e-5).any():
            return dict(failure, reason="curvature_or_zero_length")
        mid_yaw = poses[:-1, 2] + dh/2
        direction = np.c_[np.cos(mid_yaw), np.sin(mid_yaw)]
        displacement = np.diff(poses[:, :2], axis=0)
        cosine = (direction * displacement).sum(1) / ds
        if (cosine < math.cos(.08)).any():
            return dict(failure, reason="reverse_or_lateral_motion")
        # Densify arbitrary external trajectories before collision and topology checks.
        dense = [poses[0]]
        for i, distance in enumerate(ds):
            n = max(1, int(math.ceil(distance / self.config.collision_step_m)))
            for t in np.arange(1, n+1)/n:
                dense.append(np.r_[poses[i, :2] + t*displacement[i], poses[i, 2]+t*dh[i]])
        dense = np.asarray(dense)
        active = set(self.lane_matches(dense[0], lane_ids))
        if not active:
            return dict(failure, reason="lane_direction")
        bodies = []
        distances = []
        hazard_hit = False
        reach = math.hypot(self.vehicle["length"]/2 + abs(self.vehicle["rear_axle_to_center"])
                          + self.config.safety_margin_m, self.vehicle["width"]/2+self.config.safety_margin_m)
        for index, pose in enumerate(dense):
            possible = active | {out for k in active for out in self.lanes[k]["outgoing"] + self.adjacent(k)
                                 if out in self.lanes}
            if lane_ids is not None:
                possible &= set(lane_ids)
            active = set(self.lane_matches(pose, possible))
            if not active:
                return dict(failure, reason="disconnected_or_wrong_way_lane")
            body = self._body(pose)
            if index:
                angle = abs(float(wrap_angle(pose[2] - dense[index-1, 2])))
                sweep = MultiPoint(np.concatenate([bodies[-1], body])).convex_hull
                sweep = sweep.buffer(reach * (1-math.cos(angle/2)) + 1e-6)
            else:
                sweep = Polygon(body)
            if not self.road_check.covers(sweep):
                return dict(failure, reason="body_outside_road_or_viewport")
            if self.obstacle_check.intersects(sweep):
                return dict(failure, reason="body_collision")
            hazard_hit |= self.hazard_check.intersects(sweep)
            distances.append(Polygon(body).distance(self.pedestrians) if not self.pedestrians.is_empty else np.inf)
            bodies.append(body)
        lengths = np.linalg.norm(np.diff(dense[:, :2], axis=0), axis=1)
        exposure = np.exp(-np.asarray(distances) / self.config.pedestrian_scale_m)
        integral = float(np.sum(lengths * (exposure[:-1]+exposure[1:]) / 2))
        return dict(feasible=True, reason="ok", length_m=float(lengths.sum()),
                    pedestrian_integral_m=integral, hazard_hit=bool(hazard_hit),
                    min_pedestrian_clearance_m=float(min(distances)),
                    max_curvature_inv_m=float(np.max(np.abs(dh)/ds)))


def task_cost(metrics, task):
    if not metrics["feasible"] or (task["forbid_hazards"] and metrics["hazard_hit"]):
        return math.inf
    length = metrics["length_m"]
    if task["objective"] == "travel_time":
        return length / task["nominal_speed_mps"]
    return length + task["pedestrian_weight"] * metrics["pedestrian_integral_m"]


class RoadGraph:
    def __init__(self, scene):
        self.scene = scene
        self.poses = [np.zeros(3)]
        self.node_lane = [""]
        self.station = [0.]
        self.offset = [0.]
        self.edges = []
        self.adjacency = [[]]
        self.groups = {}

    def add_node(self, pose, lane, station, offset):
        self.poses.append(np.asarray(pose, dtype=float))
        self.node_lane.append(lane)
        self.station.append(station)
        self.offset.append(offset)
        self.adjacency.append([])
        return len(self.poses)-1

    def add_edge(self, source, target, lane_ids):
        if any(self.edges[i]["target"] == target for i in self.adjacency[source]):
            return
        cfg = self.scene.config
        poses = hermite_motion(self.poses[source], self.poses[target], cfg.collision_step_m, cfg.min_turn_radius_m)
        if poses is None:
            return
        metrics = self.scene.assess(poses, lane_ids)
        if not metrics["feasible"]:
            return
        self.adjacency[source].append(len(self.edges))
        self.edges.append(dict(source=source, target=target, poses=poses, metrics=metrics))

    def build(self, explicit_goal=None):
        cfg = self.scene.config
        xmin, xmax, ymin, ymax = self.scene.bounds
        for key in sorted(self.scene.lanes):
            line = self.scene.lines[key]
            stations = np.unique(np.r_[np.arange(0, line.length, cfg.station_step_m), line.length])
            groups = []
            for station in stations:
                state = _line_state(line, station)
                group = []
                for offset in cfg.lateral_offsets_m:
                    pose = state.copy()
                    pose[:2] += offset * np.array([-math.sin(pose[2]), math.cos(pose[2])])
                    if xmin < pose[0] < xmax and ymin < pose[1] < ymax:
                        group.append(self.add_node(pose, key, station, offset))
                groups.append(group)
            self.groups[key] = groups
            for j, group in enumerate(groups):
                for following in groups[j+1:j+1+cfg.max_station_skip]:
                    for source in group:
                        for target in following:
                            self.add_edge(source, target, [key])
        # Only explicit map outgoing relations create lane-to-lane transitions.
        # No inferred proximity links. Lane changes separately require explicit
        # reciprocal adjacency, shared road support and forward curved motion.
        for key, groups in self.groups.items():
            for outgoing in self.scene.lanes[key]["outgoing"]:
                if outgoing not in self.groups:
                    continue
                for group in groups[-cfg.max_station_skip:]:
                    for following in self.groups[outgoing][:cfg.max_station_skip+1]:
                        for source in group:
                            for target in following:
                                if np.linalg.norm(self.poses[target][:2]-self.poses[source][:2]) <= cfg.station_step_m * (cfg.max_station_skip+1):
                                    self.add_edge(source, target, [key, outgoing])
        for source in range(1, len(self.poses)):
            if self.offset[source] != 0.:
                continue
            key = self.node_lane[source]
            for adjacent in self.scene.adjacent(key):
                line = self.scene.lines[adjacent]
                s0 = line.project(Point(self.poses[source][:2]))
                for group in self.groups[adjacent]:
                    for target in group:
                        lead = self.station[target] - s0
                        if (self.offset[target] == 0. and 2*cfg.station_step_m <= lead <= 4*cfg.station_step_m
                                and abs(float(wrap_angle(self.poses[target][2]-self.poses[source][2]))) < math.pi/6):
                            self.add_edge(source, target, [key, adjacent])
        for key in self.scene.lane_matches(np.zeros(3)):
            s0 = self.scene.lines[key].project(Point(0, 0))
            for group in self.groups[key]:
                for target in group:
                    if .1 <= self.station[target]-s0 <= cfg.station_step_m * (cfg.max_station_skip+1):
                        self.add_edge(0, target, [key])
        if explicit_goal is not None:
            self.add_explicit_goal(explicit_goal)
        return self

    def add_explicit_goal(self, xy):
        """Exact requested position; lane tangent supplies heading, never snap xy."""
        self.explicit_nodes = []
        for key in sorted(self.scene.lanes):
            if not self.scene.polygons[key].covers(Point(xy)):
                continue
            line = self.scene.lines[key]
            station = line.project(Point(xy))
            state = _line_state(line, station)
            # User goals may lie inside a lane but far from its modeled corridor.
            if np.linalg.norm(state[:2]-xy) > max(abs(x) for x in self.scene.config.lateral_offsets_m) + .25:
                continue
            state[:2] = xy
            target = self.add_node(state, key, station, 0.)
            self.explicit_nodes.append(target)
            for source in range(1, target):
                if self.node_lane[source] == key and .1 <= station-self.station[source] <= 3*self.scene.config.station_step_m:
                    self.add_edge(source, target, [key])

    def add_pose_goal(self, pose):
        """Exact logged endpoint, including heading, without snapping to a lane.

        An unmatched or blocked goal remains an unreachable node. The GT must
        not be relocated or suppressed just because the static graph cannot solve it.
        """
        pose = np.asarray(pose, dtype=float)
        matches = sorted(self.scene.lane_matches(pose))
        key = matches[0] if matches else ""
        station = self.scene.lines[key].project(Point(pose[:2])) if key else 0.
        target = self.add_node(pose.copy(), key, station, 0.)
        for source in range(1, target):
            source_lane = self.node_lane[source]
            delta = pose[:2] - self.poses[source][:2]
            forward = np.dot(delta, [math.cos(self.poses[source][2]), math.sin(self.poses[source][2])])
            if not .1 < forward <= 3*self.scene.config.station_step_m:
                continue
            permitted = {source_lane} | set(self.scene.lanes[source_lane]["outgoing"] + self.scene.adjacent(source_lane))
            for goal_lane in matches:
                if goal_lane in permitted:
                    self.add_edge(source, target, sorted({source_lane, goal_lane}))
        return target

    def solve(self, task):
        distance = np.full(len(self.poses), np.inf)
        parent = np.full(len(self.poses), -1, dtype=np.int64)
        distance[0] = 0.
        queue = [(0., 0)]
        while queue:
            cost, source = heapq.heappop(queue)
            if cost > distance[source]:
                continue
            for edge_id in self.adjacency[source]:
                edge = self.edges[edge_id]
                candidate = cost + task_cost(edge["metrics"], task)
                target = edge["target"]
                if candidate < distance[target] - 1e-10:
                    distance[target], parent[target] = candidate, edge_id
                    heapq.heappush(queue, (candidate, target))
        return distance, parent

    def choose_goal(self, distances, args, scene_id):
        xmin, xmax, ymin, ymax = self.scene.bounds
        margin = args.goal_margin_m
        candidates = []
        for node, pose in enumerate(self.poses[1:], 1):
            distance = np.linalg.norm(pose[:2])
            if (np.isfinite(distances[node]) and self.offset[node] == 0.
                    and args.goal_min_distance <= distance <= args.goal_max_distance
                    and pose[0] >= args.goal_min_forward_m
                    and xmin+margin <= pose[0] <= xmax-margin
                    and ymin+margin <= pose[1] <= ymax-margin):
                candidates.append(node)
        if args.goal_xy is not None:
            candidates = sorted(set(candidates) & set(getattr(self, "explicit_nodes", [])))
        else:
            # Local downstream exits: furthest eligible station on each lane.
            terminal = {}
            for node in candidates:
                key = self.node_lane[node]
                if key not in terminal or self.station[node] > self.station[terminal[key]]:
                    terminal[key] = node
            candidates = sorted(terminal.values(), key=lambda i: (self.node_lane[i], self.station[i]))
            # Prefer exit lanes, rather than internal connectors, when available.
            exits = [i for i in candidates if self.scene.lanes[self.node_lane[i]]["kind"] == "lane"
                     and not any(out in terminal for out in self.scene.lanes[self.node_lane[i]]["outgoing"])]
            candidates = exits or candidates
        if not candidates:
            return None
        digest = hashlib.sha256(f"{args.goal_seed}:{scene_id}".encode()).digest()
        return candidates[int.from_bytes(digest[:8], "little") % len(candidates)]

    def reference(self, target, solution, task):
        distance, parent = solution
        if not np.isfinite(distance[target]):
            return dict(status="infeasible_on_graph", cost=None, poses=np.empty((0, 3)), edge_ids=[], metrics={})
        node, ids = target, []
        while node != 0:
            edge_id = int(parent[node])
            if edge_id < 0:
                raise RuntimeError("broken reference predecessor")
            ids.append(edge_id)
            node = self.edges[edge_id]["source"]
        ids.reverse()
        poses = np.concatenate([self.edges[i]["poses"] if j == 0 else self.edges[i]["poses"][1:]
                                for j, i in enumerate(ids)])
        metrics = self.scene.assess(poses)
        checked_cost = task_cost(metrics, task)
        if not np.isfinite(checked_cost) or not np.isclose(checked_cost, distance[target], rtol=1e-6, atol=1e-6):
            raise RuntimeError("reference failed independent geometry/cost validation")
        return dict(status="optimal_on_graph", cost=float(distance[target]), poses=poses, edge_ids=ids, metrics=metrics)

    def save(self, path):
        offsets = np.r_[0, np.cumsum([len(e["poses"]) for e in self.edges])]
        np.savez_compressed(path, node_pose=np.asarray(self.poses), node_lane=np.asarray(self.node_lane),
                            edge_index=np.asarray([[e["source"], e["target"]] for e in self.edges], dtype=np.int64).reshape(-1, 2),
                            edge_pose_offsets=offsets,
                            edge_poses=np.concatenate([e["poses"] for e in self.edges]) if self.edges else np.empty((0, 3)),
                            edge_length_m=np.asarray([e["metrics"]["length_m"] for e in self.edges]),
                            edge_pedestrian_integral_m=np.asarray([e["metrics"]["pedestrian_integral_m"] for e in self.edges]),
                            edge_hazard_hit=np.asarray([e["metrics"]["hazard_hit"] for e in self.edges], dtype=bool))


def json_ready(value):
    """Strict JSON: unavailable infinite distances are null, never NaN literals."""
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, data):
    path.write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
