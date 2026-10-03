"""Camera-free BEV geometry and record-schema tests for V8.5."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

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
