"""Static export regression tests, including absence of route/history/GT access."""

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import export_nuplan_bev_v85 as exporter
from bev_renderer_v85 import BevConfig, SEMANTIC_INDEX, ego_to_pixel, render_bev
from raw_record_bev_v85 import STATIC_INPUT_MODE
from static_bev_tasks_v85 import reachable_cells, select_goal, static_evaluation_layers


def arguments(*extra):
    return exporter.parse_args(["--data-root", "/unused", "--map-root", "/unused",
                                "--db-files", "unused.db", "--output-root", "/unused",
                                *extra])


def make_scene():
    names = ("LANE", "LANE_CONNECTOR", "CROSSWALK", "STOP_LINE",
             "ROADBLOCK", "ROADBLOCK_CONNECTOR", "DRIVABLE_AREA")
    layers = SimpleNamespace(**{name: name for name in names})
    lane = SimpleNamespace(polygon=SimpleNamespace(exterior=SimpleNamespace(
        coords=[[-36, -8], [36, -8], [36, 8], [-36, 8], [-36, -8]])),
        baseline_path=SimpleNamespace(linestring=SimpleNamespace(coords=[[-36, 0], [36, 0]])))

    def query(point, radius, requested):
        assert "DRIVABLE_AREA" not in requested
        return {layers.LANE: [lane]}

    def forbidden(*args, **kwargs):
        raise AssertionError("static export must not access route, histories, traffic or future GT")

    def actor(x, y, kind, length, width):
        return SimpleNamespace(center=SimpleNamespace(x=x, y=y, heading=0.),
                               box=SimpleNamespace(length=length, width=width),
                               tracked_object_type=SimpleNamespace(fullname=kind))

    scene = SimpleNamespace(
        map_api=SimpleNamespace(get_proximal_map_objects=query),
        get_ego_state_at_iteration=lambda _: SimpleNamespace(
            rear_axle=SimpleNamespace(x=0., y=0., heading=0.),
            center=SimpleNamespace(x=1.4, y=0.)),
        get_tracked_objects_at_iteration=lambda _: [actor(12, 4, "vehicle", 4.8, 2.),
            actor(8, -4, "pedestrian", .6, .6), actor(18, 4, "traffic_cone", .5, .5)],
        get_route_roadblock_ids=forbidden, get_past_tracked_objects=forbidden,
        get_ego_future_trajectory=forbidden, get_traffic_light_status_at_iteration=forbidden,
        get_time_point=lambda _: SimpleNamespace(time_us=1000000),
        ego_vehicle_parameters=SimpleNamespace(length=4.8, width=2.),
        log_name="log", token="scene", scenario_type="test",
    )
    state_module = ModuleType("nuplan.common.actor_state.state_representation")
    state_module.Point2D = lambda x, y: SimpleNamespace(x=x, y=y)
    map_module = ModuleType("nuplan.common.maps.maps_datatypes")
    map_module.SemanticMapLayer = layers
    return scene, {state_module.__name__: state_module, map_module.__name__: map_module}


class StaticBevTest(unittest.TestCase):
    def test_default_geometry_and_legacy_opt_in(self):
        args = arguments()
        self.assertEqual(args.planning_mode, "static_multitask")
        self.assertEqual((args.x_min, args.x_max, args.history_seconds), (-40., 40., 0.))
        legacy = arguments("--planning-mode", "route_tracking")
        self.assertEqual((legacy.x_min, legacy.x_max, legacy.history_seconds), (-16., 64., 2.))

    def test_renderer_ignores_route_history_and_lights_in_static_mode(self):
        line = np.array([[0, 0], [30, 0]])
        rendered = render_bev({"route": [np.array([[0, 0], [20, 0], [20, 5]])]},
                              {"reference": [line]}, [], [line], {"red": [line]},
                              np.column_stack([line, np.zeros(2)]), [20, 0], static_scene=True)
        self.assertEqual(rendered["bev_semantic"].shape, (16, 288, 288))
        for name in ("route", "reference", "history", "traffic_red"):
            self.assertFalse(rendered["bev_semantic"][SEMANTIC_INDEX[name]].any())

    def test_multitask_export_is_identical_image_and_has_no_future_or_route(self):
        scene, modules = make_scene()
        args = arguments("--goal-xy", "28", "0")
        config = BevConfig(x_min=-40, x_max=40, history_seconds=0, history_samples=0)
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", modules):
            self.assertTrue(exporter.export_iteration(scene, 0, directory, config, args))
            folder = Path(directory) / "log/scene/0000"
            paths = sorted(folder.glob("*.npz"))
            self.assertEqual(len(paths), 3)
            self.assertTrue((folder / "bev_preview.png").exists())
            records = []
            for path in paths:
                with np.load(path, allow_pickle=False) as data:
                    records.append({k: data[k] for k in data.files})
            self.assertEqual(len({str(r["task_text"].item()) for r in records}), 3)
            for record in records:
                self.assertEqual(str(record["input_mode"].item()), STATIC_INPUT_MODE)
                self.assertNotIn("reference_path_ego", record)
                self.assertNotIn("expert_trajectory_ego", record)
                self.assertNotIn("planning_time_s", record)
                np.testing.assert_array_equal(record["bev_rgb"], records[0]["bev_rgb"])
                np.testing.assert_array_equal(record["goal_xy"], [28, 0])
                self.assertEqual(record["region_semantic_counts"].shape, (81, 16))
                self.assertTrue(record["hazard_mask"].any())
                self.assertTrue(record["pedestrian_mask"].any())
                col, row = np.rint(ego_to_pixel([record["goal_xy"]], config)[0]).astype(int)
                self.assertTrue(record["planning_free"][row, col])
                self.assertTrue(record["hazard_free_reachable"][row, col])
                self.assertTrue(json.loads(record["task_spec_json"].item())["task_applicable"])

    def test_invalid_explicit_goal_is_skipped_not_replaced(self):
        scene, modules = make_scene()
        config = BevConfig(x_min=-40, x_max=40)
        for xy in ((80, 0), (12, 4), (20, 25)):
            args = arguments("--goal-xy", *map(str, xy))
            with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", modules):
                self.assertFalse(exporter.export_iteration(scene, 0, directory, config, args))
                self.assertFalse(list(Path(directory).rglob("*.npz")))

    def test_goal_sampling_is_reproducible_and_inside_reachable_support(self):
        scene, modules = make_scene()
        args = arguments("--no-preview")
        config = BevConfig(x_min=-40, x_max=40)
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", modules):
            exporter.export_iteration(scene, 0, directory, config, args)
            path = Path(directory) / "log/scene/0000/shortest.npz"
            with np.load(path) as record:
                goal = record["goal_xy"]
                layers = {key: record[key] for key in ("reachable", "hazard_free_reachable")}
            np.testing.assert_array_equal(select_goal(layers, config, "log/scene/0000", args), goal)
            self.assertGreaterEqual(np.linalg.norm(goal), args.goal_min_distance)
            self.assertLessEqual(np.linalg.norm(goal), args.goal_max_distance)

    def test_reachability_does_not_jump_disconnected_road_or_diagonal_gap(self):
        free = np.eye(3, dtype=bool)
        reached = reachable_cells(free, (0, 0))
        self.assertEqual(int(reached.sum()), 1)
        free = np.ones((5, 5), dtype=bool)
        free[:, 2] = False
        reached = reachable_cells(free, (2, 0))
        self.assertFalse(reached[:, 3:].any())

    def test_evaluator_clearance_inflates_obstacles_and_boundaries(self):
        config = BevConfig(x_min=-40, x_max=40)
        road = np.array([[-40, -40], [40, -40], [40, 40], [-40, 40]])
        rendered = render_bev({"drivable": [road]}, {}, [dict(center=[10, 0], heading=0,
                              length=2, width=2, type="vehicle")], [], {}, None, None,
                              config=config, static_scene=True)
        layers = static_evaluation_layers(rendered, config, 1.)
        for xy, free in (([10, 0], False), ([11.5, 0], False), ([15, 0], True), ([39.8, 0], False)):
            col, row = np.rint(ego_to_pixel([xy], config)[0]).astype(int)
            self.assertEqual(bool(layers["planning_free"][row, col]), free)

    def test_legacy_processor_rejects_static_records_with_actionable_message(self):
        from planning_data_bev_v85 import process_bev_raw_record
        with self.assertRaisesRegex(ValueError, "static_multitask records"):
            process_bev_raw_record({"input_mode": np.str_(STATIC_INPUT_MODE)})


if __name__ == "__main__":
    unittest.main()
