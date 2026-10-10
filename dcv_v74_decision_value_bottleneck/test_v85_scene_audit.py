"""Object completeness, overlap diagnostics and non-selective scene indexing."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bev_renderer_v85 import BevConfig, render_bev
from bev_scene_audit_v85 import audit_scene, compare_references, write_dataset_index
from evaluate_static_bev_v85 import model_inputs
from test_v85_static_bev import arguments, make_scene, road_scene
import export_nuplan_bev_v85 as exporter


def agent(x, y, kind="vehicle", length=4., width=2.):
    return dict(center=[x,y], heading=0., length=length, width=width, type=kind)


def raster(agents):
    cfg = BevConfig(image_size=360, x_min=-12, x_max=36, y_min=-24, y_max=24)
    return render_bev({}, {}, agents, [], {}, None, [26,0], config=cfg, static_scene=True)


def ref(points, status="optimal_on_graph"):
    return dict(poses=np.c_[points, np.zeros(len(points))], status=status, cost=10.)


class SceneAuditTest(unittest.TestCase):
    def test_outside_center_and_overlap_are_not_missing_objects(self):
        agents = [agent(10,0), agent(10,0), agent(37,0), agent(100,0), agent(8,10,"new_category",1,1)]
        audit, feature = audit_scene(road_scene(agents=agents), raster(agents), [], [], "test")
        self.assertEqual(audit["raw_count"], 5)
        self.assertEqual(audit["converted_count"], 5)
        self.assertEqual(audit["submitted_to_renderer_count"], 5)
        self.assertEqual(audit["in_view_counts"]["vehicle"], 3)
        self.assertEqual(audit["visible_counts"]["vehicle"], 2)
        self.assertFalse(audit["objects"][2]["center_in_view"])
        self.assertTrue(audit["objects"][2]["footprint_in_view"])
        self.assertEqual(audit["warning_counts"]["fully_overdrawn"], 1)
        self.assertEqual(audit["warning_counts"]["unknown_type_rendered_as_static_obstacle"], 1)
        self.assertEqual(audit["objects"][3]["fill_pixels_before_overdraw"], 0)
        self.assertTrue(audit["objects"][1]["visible_region_ids"])
        self.assertIn("moderate_traffic", feature["tags"])

    def test_ego_overdraw_is_reported(self):
        agents = [agent(0,0,length=1,width=1)]
        audit, _ = audit_scene(road_scene(agents=agents), raster(agents), [], [], "test")
        self.assertEqual(audit["objects"][0]["visible_owned_pixels"], 0)
        self.assertIn("fully_overdrawn", audit["objects"][0]["warnings"])

    def test_multilabel_groups_and_unknown_reference_proximity(self):
        agents = [agent(6+i*4,8) for i in range(5)] + [agent(8,3,"pedestrian",.6,.6),
                  agent(12,8,"traffic_cone",.5,.5)]
        tasks = [dict(id="shortest")]
        references = [ref([[0,0],[26,0]], "not_computed")]
        _, features = audit_scene(road_scene(agents=agents), raster(agents), references, tasks, "test")
        self.assertIn("multi_vehicle", features["tags"])
        self.assertIn("pedestrian_present", features["tags"])
        self.assertIn("construction_present", features["tags"])
        self.assertIsNone(features["near_feasible_reference_counts"]["shortest"])
        references[0]["status"] = "optimal_on_graph"
        audit, features = audit_scene(road_scene(agents=agents), raster(agents), references, tasks, "test", near_distance_m=2.)
        self.assertEqual(features["near_feasible_reference_counts"]["shortest"]["pedestrian"], 1)
        self.assertGreater(audit["objects"][0]["distance_to_feasible_reference_sweep_m"]["shortest"], 2.)

    def test_reference_geometry_difference_and_objective_conflict_are_distinct(self):
        tasks = [dict(id="a"),dict(id="b")]
        same = [ref([[0,0],[10,0]]), ref([[0,0],[2,0],[10,0]])]
        report = compare_references(same,tasks,[[10,10],[10,10]])
        self.assertTrue(report["all_tasks_no_cost_conflict"])
        self.assertFalse(report["any_geometry_difference"])
        different = [same[0],ref([[0,0],[5,3],[10,0]])]
        report = compare_references(different,tasks,[[10,10],[10,10]])
        self.assertTrue(report["any_geometry_difference"])
        self.assertFalse(report["any_cost_conflict"])
        self.assertTrue(compare_references(different,tasks,[[10,np.inf],[12,10]])["any_cost_conflict"])
        different[1]["status"]="infeasible_on_graph"
        report = compare_references(different,tasks,[[10,np.inf],[np.inf,np.inf]])
        self.assertIsNone(report["all_tasks_no_cost_conflict"])
        self.assertIsNone(report["any_geometry_difference"])

    def test_full_export_indexes_sparse_scene_once_and_keeps_audit_out_of_inputs(self):
        scene,modules=make_scene()
        cfg=BevConfig(image_size=360,x_min=-12,x_max=36,y_min=-24,y_max=24)
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules",modules):
            self.assertTrue(exporter.export_iteration(scene,0,directory,cfg,arguments("--goal-xy","26","0")))
            root=Path(directory)
            audit=json.loads((root/"log/scene/0000/audit/objects.json").read_text())
            self.assertEqual(audit["raw_count"],2)
            self.assertEqual(audit["in_view_counts"]["pedestrian"],1)
            self.assertEqual(audit["in_view_counts"]["static_obstacle"],1)
            for _ in range(2):
                report=write_dataset_index(root)
                self.assertEqual(report["indexed_scenes"],1)
                self.assertEqual(report["scene_tag_counts"]["sparse_traffic"],1)
                self.assertEqual(len((root/"scene_index.jsonl").read_text().splitlines()),1)
            with np.load(root/"log/scene/0000/shortest.npz",allow_pickle=False) as record:
                self.assertEqual(int(record["schema_revision"]),7)
                self.assertNotIn("object_raster_audit",model_inputs(record))
                self.assertNotIn("scene_features",model_inputs(record))

    def test_invalid_object_geometry_is_not_silently_dropped(self):
        scene,_=make_scene()
        obj=scene.get_tracked_objects_at_iteration(0)[0]
        obj.box.width=float("nan")
        with self.assertRaisesRegex(ValueError,"refusing to silently drop"):
            exporter._agent_primitive(obj,[0.,0.,0.])


if __name__ == "__main__":
    unittest.main()
