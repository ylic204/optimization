"""Camera-free BEV geometry and record-schema tests for V8.5."""

import json
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from bev_renderer_v85 import (
    BevConfig,
    SEMANTIC_INDEX,
    SEMANTIC_NAMES,
    ego_to_pixel,
    pixel_to_ego,
    render_bev,
)
from raw_record_bev_v85 import INPUT_MODE, save_bev_raw_record
from export_nuplan_bev_v85 import _reference_path_world
import export_nuplan_bev_v85 as exporter


class BevGeometryTest(unittest.TestCase):
    def test_metric_pixel_round_trip_and_region_bounds(self):
        config = BevConfig()
        points = np.asarray(
            [[0.0, 0.0], [63.0, 39.0], [-15.0, -39.0]],
            dtype=np.float32,
        )
        recovered = pixel_to_ego(ego_to_pixel(points, config), config)
        np.testing.assert_allclose(recovered, points, atol=1e-4)

        rendered = self._render(config)
        bounds = rendered["region_world_bounds"]
        self.assertEqual(bounds.shape, (81, 4))
        np.testing.assert_allclose(bounds[0], [55.11111, 64, 31.11111, 40])
        np.testing.assert_allclose(bounds[-1], [-16, -7.111111, -40, -31.11111])

    def test_renderer_has_fixed_shapes_and_semantics(self):
        rendered = self._render(BevConfig())
        self.assertEqual(rendered["bev_rgb"].shape, (288, 288, 3))
        self.assertEqual(
            rendered["bev_semantic"].shape,
            (len(SEMANTIC_NAMES), 288, 288),
        )
        self.assertEqual(rendered["region_semantic_counts"].shape, (81, 15))
        self.assertTrue(rendered["traversable"].any())
        self.assertTrue(rendered["dynamic_obstacle"].any())
        self.assertTrue(
            rendered["bev_semantic"][SEMANTIC_INDEX["reference"]].any()
        )

    def test_raw_record_is_explicitly_bev_only(self):
        config = BevConfig()
        rendered = self._render(config)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sample.npz"
            save_bev_raw_record(
                output_path=output,
                reference_path_ego=np.asarray([[0, 0, 0], [20, 0, 0]]),
                map_bounds=config.bounds,
                goal_xy=np.asarray([20, 0]),
                ego_state=np.zeros(8),
                task_text="safe progress",
                sample_id="log/scenario/0000",
                expert_trajectory_ego=np.asarray([[1, 0, 0], [2, 0, 0]]),
                bev_config_json=config.to_json(),
                **rendered,
            )
            with np.load(output, allow_pickle=False) as data:
                self.assertEqual(str(data["input_mode"].item()), INPUT_MODE)
                self.assertNotIn("image", data.files)
                self.assertNotIn("camera", data.files)
                config_data = json.loads(data["bev_config_json"].item())
                self.assertEqual(config_data["image_size"], 288)

    def test_route_reference_never_reverses_to_the_longer_lane_side(self):
        def edge(points):
            line = SimpleNamespace(coords=points)
            return SimpleNamespace(
                baseline_path=SimpleNamespace(linestring=line)
            )

        route = [
            (object(), [edge([[0, 0], [5, 0], [10, 0]])]),
            (object(), [edge([[10, 0], [15, 0], [20, 0]])]),
        ]
        reference = _reference_path_world(route, np.asarray([8, 0]))
        self.assertGreaterEqual(reference[0, 0], 8.0)
        self.assertTrue((np.diff(reference[:, 0]) >= 0.0).all())

    def test_export_with_nuplan_object_layer_contract_preserves_lane_mask_and_gt(self):
        # nuPlan exposes DRIVABLE_AREA as a raster, but rejects object queries
        # for it. Reproduce that public API contract without dataset downloads.
        names = (
            "LANE", "LANE_CONNECTOR", "CROSSWALK", "STOP_LINE",
            "ROADBLOCK", "ROADBLOCK_CONNECTOR", "DRIVABLE_AREA",
        )
        layers = SimpleNamespace(**{name: name for name in names})
        supported = set(names) - {"DRIVABLE_AREA"}

        def polygon(x_min, x_max):
            return SimpleNamespace(polygon=SimpleNamespace(
                exterior=SimpleNamespace(coords=[
                    [x_min, -2], [x_max, -2], [x_max, 2],
                    [x_min, 2], [x_min, -2],
                ])
            ))

        def query(point, radius, requested):
            unsupported = set(requested) - supported
            assert not unsupported, f"Object representation unavailable: {unsupported}"
            return {
                layers.LANE: [polygon(0, 10)],
                layers.LANE_CONNECTOR: [polygon(10, 20)],
            }

        state_module = ModuleType("nuplan.common.actor_state.state_representation")
        state_module.Point2D = lambda x, y: SimpleNamespace(x=x, y=y)
        map_module = ModuleType("nuplan.common.maps.maps_datatypes")
        map_module.SemanticMapLayer = layers
        scenario = SimpleNamespace(
            map_api=SimpleNamespace(get_proximal_map_objects=query),
            get_ego_state_at_iteration=lambda _: SimpleNamespace(
                rear_axle=SimpleNamespace(x=0., y=0., heading=0.)
            ),
            get_route_roadblock_ids=lambda: [],
            get_traffic_light_status_at_iteration=lambda _: [],
            get_time_point=lambda _: SimpleNamespace(time_us=1000000),
            ego_vehicle_parameters=SimpleNamespace(length=4.8, width=2.),
            log_name="log", token="scenario", scenario_type="test",
        )
        args = SimpleNamespace(future_samples=16, future_horizon=4., task_text="drive safely")
        times = np.arange(1, 17) * .25
        expert = np.column_stack([times, np.zeros((16, 2))])
        config = BevConfig()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            stack.enter_context(patch.dict("sys.modules", {
                state_module.__name__: state_module,
                map_module.__name__: map_module,
            }))
            replacements = {
                "_route_edges": [],
                "_reference_path_ego": np.asarray([[0, 0, 0], [20, 0, 0]]),
                "_future_expert": (expert, times),
                "_agents_and_histories": ([], []),
                "_traffic_lines": {},
                "_ego_state_vector": np.zeros(8),
            }
            for name, value in replacements.items():
                stack.enter_context(patch.object(exporter, name, return_value=value))
            self.assertTrue(exporter.export_iteration(scenario, 0, directory, config, args))
            with np.load(Path(directory) / "log/scenario/0000.npz", allow_pickle=False) as data:
                for xy, expected in (([5, 0], True), ([15, 0], True), ([5, 8], False)):
                    col, row = np.rint(ego_to_pixel([xy], config)[0]).astype(int)
                    self.assertEqual(bool(data["traversable"][row, col]), expected)
                np.testing.assert_allclose(data["expert_trajectory_ego"], expert)
                np.testing.assert_allclose(data["expert_time_s"], times)
                np.testing.assert_allclose(data["planning_time_s"], times)

    @staticmethod
    def _render(config):
        return render_bev(
            map_polygons={
                "drivable": [
                    np.asarray([[-10, -12], [50, -12], [50, 12], [-10, 12]])
                ],
                "route": [
                    np.asarray([[-5, -4], [45, -4], [45, 4], [-5, 4]])
                ],
                "crosswalk": [
                    np.asarray([[18, -10], [21, -10], [21, 10], [18, 10]])
                ],
            },
            map_lines={"lane_center": [np.asarray([[-10, 0], [50, 0]])]},
            agents=[
                {
                    "center": [15, 1.5],
                    "heading": 0.0,
                    "length": 4.8,
                    "width": 2.0,
                    "type": "vehicle",
                }
            ],
            histories=[np.asarray([[10, 1.5], [12, 1.5], [15, 1.5]])],
            traffic_lines={"red": [np.asarray([[17, -4], [17, 4]])]},
            reference_path=np.asarray([[0, 0, 0], [20, 0, 0], [40, 0, 0]]),
            goal_xy=np.asarray([40, 0]),
            config=config,
        )


if __name__ == "__main__":
    unittest.main()
