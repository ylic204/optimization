"""Metric-consistent ego-centric BEV rendering for the V8.5 selector.

The renderer is simulator independent.  nuPlan-specific code converts map and
tracked-object objects into the primitive dictionaries consumed here.  The
visual input is a single RGB BEV, while semantic layers are saved for audits
and never passed to the Student as privileged features.
"""

from dataclasses import asdict, dataclass
import json
import math

import numpy as np
from PIL import Image, ImageDraw


SEMANTIC_NAMES = (
    "drivable",
    "route",
    "lane_center",
    "crosswalk",
    "reference",
    "traffic_red",
    "traffic_yellow",
    "traffic_green",
    "vehicle",
    "pedestrian",
    "bicycle",
    "static_obstacle",
    "history",
    "ego",
    "goal",
)
SEMANTIC_INDEX = {name: index for index, name in enumerate(SEMANTIC_NAMES)}
STATIC_SEMANTIC_NAMES = SEMANTIC_NAMES + ("hazard",)


@dataclass(frozen=True)
class BevConfig:
    """Fixed physical support; never auto-scale individual scenes."""

    image_size: int = 288
    x_min: float = -16.0
    x_max: float = 64.0
    y_min: float = -40.0
    y_max: float = 40.0
    region_grid: int = 9
    history_seconds: float = 2.0
    history_samples: int = 5

    def __post_init__(self):
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not self.x_min < self.x_max or not self.y_min < self.y_max:
            raise ValueError("invalid BEV bounds")
        if self.image_size % self.region_grid != 0:
            raise ValueError("image_size must be divisible by region_grid")
        x_span = self.x_max - self.x_min
        y_span = self.y_max - self.y_min
        if not math.isclose(x_span, y_span, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("V8.5 BEV currently requires equal x/y spans")

    @property
    def bounds(self):
        return np.asarray(
            [self.x_min, self.x_max, self.y_min, self.y_max],
            dtype=np.float32,
        )

    @property
    def meters_per_pixel(self):
        return (self.x_max - self.x_min) / self.image_size

    def to_json(self):
        return json.dumps(asdict(self), sort_keys=True)


RGB = {
    "background": (247, 247, 244),
    "drivable": (221, 235, 245),
    "route": (184, 222, 242),
    "lane_center": (132, 139, 145),
    "crosswalk": (178, 181, 184),
    "reference": (139, 74, 186),
    "traffic_red": (224, 54, 54),
    "traffic_yellow": (235, 183, 42),
    "traffic_green": (46, 160, 67),
    "history": (56, 154, 92),
    "vehicle": (45, 108, 190),
    "pedestrian": (139, 88, 62),
    "bicycle": (207, 65, 139),
    "static_obstacle": (30, 30, 30),
    "ego": (235, 126, 35),
    "goal": (232, 178, 25),
    "hazard": (240, 168, 125),
}


def _points(values):
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return array.reshape(-1, 2)


def world_to_ego(points, ego_pose):
    """Transform global xy to x-forward/y-left coordinates at ego rear axle."""
    # UTM coordinates can be millions of meters. Subtract the origin before
    # converting to float32, otherwise lane details are rounded by decimeters.
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x, y, heading = np.asarray(ego_pose, dtype=np.float64)
    delta = points.astype(np.float64) - np.asarray([x, y])
    cosine, sine = math.cos(heading), math.sin(heading)
    rotation = np.asarray([[cosine, sine], [-sine, cosine]])
    return (delta @ rotation.T).astype(np.float32)


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def ego_to_pixel(points, config):
    """Return PIL-style (column, row) coordinates for ego-frame xy."""
    points = _points(points)
    column = (
        (config.y_max - points[:, 1])
        / (config.y_max - config.y_min)
        * (config.image_size - 1)
    )
    row = (
        (config.x_max - points[:, 0])
        / (config.x_max - config.x_min)
        * (config.image_size - 1)
    )
    return np.stack([column, row], axis=-1).astype(np.float32)


def pixel_to_ego(pixel, config):
    pixel = _points(pixel)
    y = config.y_max - (
        pixel[:, 0] / (config.image_size - 1)
    ) * (config.y_max - config.y_min)
    x = config.x_max - (
        pixel[:, 1] / (config.image_size - 1)
    ) * (config.x_max - config.x_min)
    return np.stack([x, y], axis=-1).astype(np.float32)


def oriented_box_corners(center, heading, length, width):
    local = np.asarray(
        [
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
        ],
        dtype=np.float32,
    )
    cosine, sine = math.cos(float(heading)), math.sin(float(heading))
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + np.asarray(center, dtype=np.float32)


def region_world_bounds(config):
    """Physical [xmin,xmax,ymin,ymax] for row-major visual regions."""
    x_edges = np.linspace(config.x_max, config.x_min, config.region_grid + 1)
    y_edges = np.linspace(config.y_max, config.y_min, config.region_grid + 1)
    bounds = []
    for row in range(config.region_grid):
        for column in range(config.region_grid):
            bounds.append(
                [
                    x_edges[row + 1],
                    x_edges[row],
                    y_edges[column + 1],
                    y_edges[column],
                ]
            )
    return np.asarray(bounds, dtype=np.float32)


def region_semantic_counts(semantic, config):
    patch = config.image_size // config.region_grid
    channels = []
    for row in range(config.region_grid):
        for column in range(config.region_grid):
            crop = semantic[
                :,
                row * patch : (row + 1) * patch,
                column * patch : (column + 1) * patch,
            ]
            channels.append(crop.reshape(semantic.shape[0], -1).mean(-1))
    return np.asarray(channels, dtype=np.float32)


def _draw_polygon(draw, points, fill):
    if len(points) >= 3:
        draw.polygon([tuple(value) for value in points], fill=fill)


def _draw_line(draw, points, fill, width):
    if len(points) >= 2:
        draw.line(
            [tuple(value) for value in points],
            fill=fill,
            width=max(1, int(width)),
            joint="curve",
        )


def _star(center, outer=5.0, inner=2.3):
    points = []
    for index in range(10):
        angle = -math.pi / 2.0 + index * math.pi / 5.0
        radius = outer if index % 2 == 0 else inner
        points.append(
            [center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)]
        )
    return np.asarray(points, dtype=np.float32)


def render_bev(
    map_polygons,
    map_lines,
    agents,
    histories,
    traffic_lines,
    reference_path,
    goal_xy,
    config=None,
    ego_size=(5.176, 2.297),
    static_scene=False,
    hazard_polygons=(),
    ego_center_xy=(0.0, 0.0),
):
    """Render one scene from ego-frame primitives.

    ``map_polygons`` keys: drivable, route, crosswalk.
    ``map_lines`` keys: lane_center and optional reference.
    Agent dictionaries require center, heading, length, width and type.
    History entries are ego-frame xy polylines.  Traffic-line keys are
    red/yellow/green/unknown; unknown is deliberately rendered as lane gray.
    ``static_scene`` suppresses route/reference/history/traffic layers even if
    supplied, and adds a hazard channel. ``ego_center_xy`` locates the body;
    the start coordinate remains the rear-axle origin.
    """
    config = config or BevConfig()
    names = STATIC_SEMANTIC_NAMES if static_scene else SEMANTIC_NAMES
    indices = {name: index for index, name in enumerate(names)}
    rgb = Image.new(
        "RGB", (config.image_size, config.image_size), RGB["background"]
    )
    rgb_draw = ImageDraw.Draw(rgb)
    semantic_images = [
        Image.new("L", (config.image_size, config.image_size), 0)
        for _ in names
    ]
    semantic_draw = [ImageDraw.Draw(image) for image in semantic_images]

    def polygon_layer(name, values, color=None):
        for value in values:
            pixel = ego_to_pixel(value, config)
            _draw_polygon(rgb_draw, pixel, RGB[color or name])
            _draw_polygon(semantic_draw[indices[name]], pixel, 255)

    def line_layer(name, values, width=2, color=None):
        for value in values:
            pixel = ego_to_pixel(value, config)
            _draw_line(rgb_draw, pixel, RGB[color or name], width)
            _draw_line(
                semantic_draw[indices[name]], pixel, 255, width
            )

    polygon_layer("drivable", map_polygons.get("drivable", []))
    for hole in map_polygons.get("road_holes", []):
        pixel = ego_to_pixel(hole, config)
        _draw_polygon(rgb_draw, pixel, RGB["background"])
        _draw_polygon(semantic_draw[indices["drivable"]], pixel, 0)
    if not static_scene:
        polygon_layer("route", map_polygons.get("route", []))
    polygon_layer("crosswalk", map_polygons.get("crosswalk", []))
    if static_scene:
        polygon_layer("hazard", hazard_polygons)
        # Map boundaries and arrows describe all lanes, never a chosen route.
        for value in map_polygons.get("drivable", []):
            points = ego_to_pixel(value, config)
            _draw_line(rgb_draw, np.concatenate([points, points[:1]]), (115, 129, 139),
                       max(1, config.image_size // 360))
    line_layer("lane_center", map_lines.get("lane_center", []), width=1)
    if static_scene:
        for line in map_lines.get("lane_center", []):
            line = np.asarray(line)
            if len(line) < 2:
                continue
            length = np.linalg.norm(np.diff(line, axis=0), axis=1)
            station = np.r_[0., np.cumsum(length)]
            for s in np.arange(5., station[-1], 10.):
                i = min(np.searchsorted(station, s, side="right")-1, len(length)-1)
                if length[i] < 1e-6:
                    continue
                direction = (line[i+1]-line[i])/length[i]
                tip = line[i] + (s-station[i])*direction
                normal = np.array([-direction[1], direction[0]])
                arrow = np.array([tip-.65*direction+.3*normal, tip, tip-.65*direction-.3*normal])
                _draw_line(rgb_draw, ego_to_pixel(arrow, config), (100, 112, 125),
                           max(1, config.image_size // 360))
    if not static_scene:
        line_layer("reference", map_lines.get("reference", []), width=3)
        if reference_path is not None and len(reference_path):
            line_layer("reference", [np.asarray(reference_path)[:, :2]], width=3)

    for status in (() if static_scene else ("red", "yellow", "green")):
        line_layer(
            f"traffic_{status}", traffic_lines.get(status, []), width=4
        )
    for value in ([] if static_scene else traffic_lines.get("unknown", [])):
        _draw_line(rgb_draw, ego_to_pixel(value, config), RGB["lane_center"], 3)

    if not static_scene:
        line_layer("history", histories, width=2)

    type_map = {
        "vehicle": "vehicle",
        "pedestrian": "pedestrian",
        "bicycle": "bicycle",
        "traffic_cone": "static_obstacle",
        "barrier": "static_obstacle",
        "czone_sign": "static_obstacle",
        "generic_object": "static_obstacle",
    }
    for agent in agents:
        layer = type_map.get(str(agent["type"]).lower(), "static_obstacle")
        corners = oriented_box_corners(
            agent["center"],
            agent["heading"],
            max(float(agent["length"]), 0.8),
            max(float(agent["width"]), 0.6),
        )
        pixel = ego_to_pixel(corners, config)
        _draw_polygon(rgb_draw, pixel, RGB[layer])
        if static_scene:
            _draw_line(rgb_draw, np.concatenate([pixel, pixel[:1]]), (27, 45, 62),
                       max(1, config.image_size // 360))
        _draw_polygon(semantic_draw[indices[layer]], pixel, 255)
        if layer in ("vehicle", "bicycle"):
            center = np.asarray(agent["center"], dtype=np.float32)
            front = center + 0.65 * float(agent["length"]) * np.asarray(
                [math.cos(agent["heading"]), math.sin(agent["heading"])]
            )
            _draw_line(
                rgb_draw,
                ego_to_pixel(np.stack([center, front]), config),
                (250, 250, 250),
                1,
            )

    ego_corners = oriented_box_corners(
        ego_center_xy, 0.0, float(ego_size[0]), float(ego_size[1])
    )
    ego_pixel = ego_to_pixel(ego_corners, config)
    _draw_polygon(rgb_draw, ego_pixel, RGB["ego"])
    _draw_polygon(semantic_draw[SEMANTIC_INDEX["ego"]], ego_pixel, 255)
    _draw_line(
        rgb_draw,
        ego_to_pixel([[0.0, 0.0], [ego_size[0] * 0.7, 0.0]], config),
        (255, 255, 255),
        2,
    )
    # A triangular arrow identifies heading; no text is baked into the image.
    if static_scene:
        tip = np.asarray([max(float(ego_size[0]) * 0.7, 2.0), 0.0])
        arrow = np.asarray([tip, tip + [-1.3, 0.7], tip + [-1.3, -0.7]])
        _draw_polygon(rgb_draw, ego_to_pixel(arrow, config), (255, 255, 255))

    if goal_xy is not None:
        center = ego_to_pixel([goal_xy], config)[0]
        radius = max(5., .75 * (config.image_size-1) / (config.x_max-config.x_min))
        goal = _star(center, outer=radius, inner=radius*.46) if static_scene else _star(center)
        _draw_polygon(rgb_draw, goal, RGB["goal"])
        _draw_polygon(semantic_draw[SEMANTIC_INDEX["goal"]], goal, 255)

    semantic = np.stack(
        [np.asarray(image, dtype=np.uint8) > 0 for image in semantic_images]
    ).astype(np.uint8)
    dynamic_indices = [
        SEMANTIC_INDEX[name]
        for name in ("vehicle", "pedestrian", "bicycle", "static_obstacle")
    ]
    traversable = semantic[SEMANTIC_INDEX["drivable"]].astype(bool)
    dynamic_obstacle = semantic[dynamic_indices].any(axis=0)
    return {
        "bev_rgb": np.asarray(rgb, dtype=np.uint8),
        "bev_semantic": semantic,
        "traversable": traversable,
        "dynamic_obstacle": dynamic_obstacle,
        "region_world_bounds": region_world_bounds(config),
        "region_semantic_counts": region_semantic_counts(semantic, config),
    }
